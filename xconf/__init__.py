"""Minimal configuration system to populate dataclasses with a
mix of config files in TOML and command-line arguments"""
import textwrap
import argparse
import re
import sys
import toml
import logging
from enum import Enum
from functools import partial
import typing
from .vendor.py310_dataclasses import dataclass, is_dataclass, asdict
from .vendor import dacite
from .vendor import py310_dataclasses as dataclasses
from collections import defaultdict

log = logging.getLogger(__name__)

config = partial(dataclass, kw_only=True)

def load_config(filepath):
    log.debug(f'Loading config from {filepath}')
    with open(filepath, 'r') as fh:
        raw_config = toml.load(fh)
    return raw_config

class NestedDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(self.to_plain_dict())
    
    def to_plain_dict(self):
        out = dict()
        for key, value in self.items():
            if isinstance(value, self.__class__):
                out[key] = value.to_plain_dict()
            else:
                out[key] = value
        return out

def fail(message):
    log.error(message)
    sys.exit(1)

def convert_bool(x):
    if x in (True, False):
        return x
    if x.lower() in ('true', 't', '1'):
        return True
    elif x.lower() in ('false', 'f', '0'):
        return False
    else:
        raise ValueError(f"Cannot convert {x} to boolean")


def convert_list_of_str(x):
    if ',' in x:
        return [y.strip() for y in x.split(',')]
    else:
        return [x.strip()]

TYPE_HOOKS = {
    bool: convert_bool,
    int: int,
    float: float,
    list[str]: convert_list_of_str,
}

def field(*args, **kwargs):
    metadata = None
    if 'help' in kwargs:
        metadata = {'help': kwargs.pop('help')}
    return dataclasses.field(*args, metadata=metadata, **kwargs)

def format_field_type(field_type):
    if hasattr(field_type, '__args__'):
        if dacite.types.is_optional(field_type):
            return f'(optional) {field_type.__args__[0].__name__}'
        elif dacite.types.is_union(field_type):
            name = ''
        else:
            name = field_type.__name__
        members = ', '.join([x.__name__ for x in field_type.__args__])
        return f'{name}[{members}]'
    return field_type.__name__

def list_fields(cls, prefix='', help_suffix=''):
    for fld in dataclasses.fields(cls):
        name = fld.name
        field_help = fld.metadata.get('help', '')
        if not isinstance(fld.default, dataclasses._MISSING_TYPE):
            field_help += f' (default: {repr(fld.default)})'
        field_help += help_suffix
        field_type_str = format_field_type(fld.type)
        prefixed_name = prefix + name
        yield prefixed_name, f'{field_type_str}', field_help
        
        
        if dacite.types.is_union(fld.type) and not dacite.types.is_optional(fld.type):
            members = dacite.types.extract_generic(fld.type)
            for mtype in members:
                if dataclasses.is_dataclass(mtype):
                    for k, v, h in list_fields(mtype, prefix=f'{prefix}{name}.', help_suffix=help_suffix+f' <{mtype.__name__}>'):
                        yield k, v, h
                else:
                    # no need for additional docs for primitive types
                    continue
        elif dacite.types.is_generic_collection(fld.type):
            if dacite.types.extract_origin_collection(fld.type) is list:
                # already output above for collection
                continue
            else:
                key_type, val_type = dacite.types.extract_generic(fld.type)
                for k, v, h in list_fields(val_type, prefix=f'{prefix}{name}.<{key_type.__name__}>.'):
                    yield k, v, h
        else:
            if dataclasses.is_dataclass(fld.type):
                for k, v, h in list_fields(fld.type, prefix=f'{prefix}{name}.'):
                    yield k, v, h
            

class Dispatcher:
    def __init__(self, commands):
        self.commands = commands

    def configure_logging(self, level):
        logging.basicConfig(level=level)

    def main(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.set_defaults(command_cls=None)
        subps = parser.add_subparsers(title="subcommands", description="valid subcommands")
        names_to_subparsers = {}
        for command_cls in self.commands:
            if command_cls.name is None or command_cls.name in names_to_subparsers:
                raise Exception(f"Invalid command name for {command_cls}")
            subp = subps.add_parser(command_cls.name, add_help=False)
            names_to_subparsers[command_cls.name] = subp
            subp.set_defaults(command_cls=command_cls)
            default_config_file = f"{command_cls.name}.conf.toml"
            subp.add_argument("-c", "--config-file", help=f"Path to config file (default: {default_config_file})", default=default_config_file)
            subp.add_argument("-h", "--help", action='store_true', help="Print usage information")
            subp.add_argument("-v", "--verbose", action='store_true', help="Enable debug logging")
            subp.add_argument("vars", nargs='*', help="Config variables set with 'key.key.key=value' notation")
        
        args = parser.parse_args()
        if args.command_cls is None:
            parser.print_help()
            sys.exit(1)
        if args.help:
            print_help(args.command_cls, names_to_subparsers[args.command_cls.name])
            sys.exit(0)
        self.configure_logging('DEBUG' if args.verbose else 'INFO')
        command = args.command_cls.from_args(args)
        return command.main()

def print_help(cls, parser):
    print(f"{cls.name}: {cls.__doc__}")
    parser.print_help()
    print("configuration keys:")
    fields = list_fields(cls)
    for field_key, field_type, help_text in fields:
        print(f'  {field_key}')
        print(f'      {field_type}')
        if help_text:
            wrapped = textwrap.fill(
                help_text,
                initial_indent='    ',
                subsequent_indent='    '
            )
            print(wrapped)

def _get_config_data(config_file_path, cli_args):
    try:
        raw_config = load_config(config_file_path)
    except FileNotFoundError:
        raw_config = {}
    raw_config = NestedDefaultDict(raw_config)

    for non_flag_arg in cli_args:
        lhs, rhs = non_flag_arg.split('=', 1)
        key = lhs.split('.')
        base = raw_config
        for subkey in key[:-1]:
            base = base[subkey]
        base[key[-1]] = rhs
    
    log.debug(f'Config attributes provided: {raw_config}')
    raw_config = raw_config.to_plain_dict()
    return raw_config

@dataclass
class Command:
    @classmethod
    @property
    def name(cls):
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
        return name

    @classmethod
    def from_args(cls, parsed_args):
        raw_config = _get_config_data(parsed_args.config_file, parsed_args.vars)
        if parsed_args.verbose:
            import coloredlogs
            coloredlogs.install(level='DEBUG', logger=log)
            log.debug('Verbose logging enabled')
        else:
            log.setLevel(logging.INFO)
        try:
            return dacite.from_dict(
                cls,
                raw_config,
                config=dacite.Config(strict=True, type_hooks=TYPE_HOOKS, cast=[Enum])
            )
        except (dacite.UnexpectedDataError, dacite.MissingValueError) as e:
            log.error(f"Applying configuration failed with: {e}")
            from pprint import pformat
            log.error(f"File and arguments provided this configuration:\n\n{pformat(raw_config)}\n")
            sys.exit(1)

    def main(self):
        raise NotImplementedError("Subclasses must implement main()")
