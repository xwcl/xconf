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
from toml import encoder
from .vendor.py310_dataclasses import dataclass, is_dataclass, asdict
from .vendor import dacite
from .vendor.dacite import UnexpectedDataError, MissingValueError
from .vendor import py310_dataclasses as dataclasses
from collections import defaultdict

log = logging.getLogger(__name__)

config = partial(dataclass, kw_only=True)

def convert_bool(x):
    if x in (True, False):
        return x
    if x.lower() in ('true', 't', '1'):
        return True
    elif x.lower() in ('false', 'f', '0'):
        return False
    else:
        raise ValueError(f"Cannot convert {x} to boolean")


def convert_list(x, convert_type=lambda x: x):
    if ',' in x:
        return [convert_type(y.strip()) for y in x.split(',')]
    elif isinstance(x, str):
        return [convert_type(x.strip())]
    else:
        return x

TYPE_HOOKS = {
    bool: convert_bool,
    int: int,
    float: float,
    list[str]: partial(convert_list, convert_type=str),
    # list[int]: partial(convert_list, convert_type=int),
    # list[float]: partial(convert_list, convert_type=float),
}
from_dict = partial(dacite.from_dict, config=dacite.Config(strict=True, type_hooks=TYPE_HOOKS, cast=[Enum]))


class TomlEnumEncoder(encoder.TomlEncoder):
    """Encode to TOML, replacing values that are Enum subclasses with their value attribute"""

    def dump_value(self, v):
        if isinstance(v, Enum):
            return super().dump_value(v.value)
        return super().dump_value(v)

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

def field(*args, **kwargs):
    metadata = None
    if 'help' in kwargs:
        metadata = {'help': kwargs.pop('help')}
    return dataclasses.field(*args, metadata=metadata, **kwargs)

def type_name_or_choices(the_type):
    if dacite.types.is_subclass(the_type, Enum):
        return ', '.join([repr(x.value) for x in the_type])
    elif hasattr(the_type, '__name__'):
        return the_type.__name__
    elif dacite.types.is_union(the_type):
        members = dacite.types.extract_generic(the_type)
        return '(' + ', '.join([type_name_or_choices(m) for m in members]) + ')'
    else:
        return repr(the_type)

def format_field_type(field_type):
    if hasattr(field_type, '__args__'):
        if dacite.types.is_optional(field_type):
            return f'(optional) {format_field_type(field_type.__args__[0])}'
        elif dacite.types.is_union(field_type):
            name = ''
        else:
            name = type_name_or_choices(field_type)
        members = ', '.join([type_name_or_choices(x) for x in field_type.__args__])
        return f'{name}[{members}]'
    return type_name_or_choices(field_type)

def list_fields(cls, prefix='', help_suffix=''):
    for fld in dataclasses.fields(cls):
        for result in list_one_dataclass_field(fld, prefix, help_suffix):
            yield result

def list_one_dataclass_field(fld : dataclasses.Field, prefix, help_suffix):
    name = fld.name
    field_help = fld.metadata.get('help', '')
    fld_type = fld.type
    default = fld.default
    if not isinstance(fld.default_factory, dataclasses._MISSING_TYPE):
        default = fld.default_factory()
    for result in list_one_field(name, fld_type, field_help, prefix, help_suffix, default=default):
        yield result

def list_one_field(name, fld_type, field_help, prefix, help_suffix, default=dataclasses._MISSING_TYPE):
    if not isinstance(default, dataclasses._MISSING_TYPE):
        if isinstance(default, Enum):
            default_value = repr(default.value)
        else:
            default_value = repr(default)
        field_help += f' (default: {default_value})'
    field_help += help_suffix
    field_type_str = format_field_type(fld_type)
    prefixed_name = prefix + name
    yield prefixed_name, f'{field_type_str}', field_help

    if dacite.types.is_union(fld_type): # and not dacite.types.is_optional(fld.type):
        members = dacite.types.extract_generic(fld_type)
        for mtype in members:
            if dataclasses.is_dataclass(mtype):
                for k, v, h in list_fields(mtype, prefix=f'{prefix}{name}.', help_suffix=help_suffix+f' <{mtype.__name__}>'):
                    yield k, v, h
            elif dacite.types.is_generic_collection(mtype):
                collection_entry_type, = dacite.types.extract_generic(mtype)
                for k, v, h in list_fields(collection_entry_type, prefix=f'{prefix}{name}[#].', help_suffix=help_suffix+f' <{format_field_type(collection_entry_type)}>'):
                    yield k, v, h
            else:
                # no need for additional docs for primitive types
                continue
    elif dacite.types.is_generic_collection(fld_type):
        if dacite.types.extract_origin_collection(fld_type) is list:
            # already output above for collection
            pass
        else:
            # dict types
            key_type, val_type = dacite.types.extract_generic(fld_type)
            if dataclasses.is_dataclass(val_type):
                # value is a dataclass, recurse into its fields
                for k, v, h in list_fields(val_type, prefix=f'{prefix}{name}.<{key_type.__name__}>.'):
                    yield k, v, h
            else:
                if dacite.types.is_union(val_type):
                    members = dacite.types.extract_generic(val_type)
                    if any(dataclasses.is_dataclass(m) for m in members):
                        for mtype in members:
                            if dataclasses.is_dataclass(val_type):
                                for k, v, h in list_fields(val_type, prefix=f'{prefix}{name}.<{key_type.__name__}>.', help_suffix=help_suffix+f' <{format_field_type(mtype)}>'):
                                    yield k, v, h
                            else:
                                yield f"{prefixed_name}.<{format_field_type(key_type)}>", format_field_type(mtype), field_help
                    else:
                        yield f"{prefixed_name}.<{format_field_type(key_type)}>", format_field_type(val_type), field_help
                elif dacite.types.is_generic_collection(val_type):
                    # TODO: when the value itself is generic, recurse into it
                    pass
    else:
        if dataclasses.is_dataclass(fld_type):
            for k, v, h in list_fields(fld_type, prefix=f'{prefix}{name}.'):
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
            default_config_file = command_cls.default_config_name
            subp.add_argument(
                "-c", "--config-file",
                action="append",
                default=[],
                help=f"Path to config file, repeat to merge multiple (default: {default_config_file})",
            )
            subp.add_argument("-h", "--help", action='store_true', help="Print usage information")
            subp.add_argument("-v", "--verbose", action='store_true', help="Enable debug logging")
            subp.add_argument("--dump-config", action='store_true', help="Dump final configuration state as TOML and exit")
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
        if args.dump_config:
            print(toml.dumps(command.config_to_dict(), encoder=TomlEnumEncoder()))
            sys.exit(0)
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

def _get_config_data(default_config_name, config_file_paths, cli_args):
    if len(config_file_paths):
        raw_config = {}
        for config_file_path in config_file_paths:
            loaded_config = load_config(config_file_path)
            raw_config.update(loaded_config)
    else:
        try:
            raw_config = load_config(default_config_name)
        except FileNotFoundError:
            raw_config = {}

    for non_flag_arg in cli_args:
        lhs, rhs = non_flag_arg.split('=', 1)  # something.foo.bar=value
        key = lhs.split('.') # [something, foo, bar]
        base = raw_config
        for idx, subkey in enumerate(key):
            if idx == len(key) - 1:
                # instead of creating containers, last key will assign a value
                last_key = True
            else:
                last_key = False
            # if key includes index, it's a key and a hint there's a list
            idx_match = re.match(r'(.+)\[(\d+)\]', subkey)
            if idx_match is not None:
                subkey, idx_str = idx_match.groups()
                idx = int(idx_str)

            # not using defaultdict any more, so explicitly create
            # the containers down to the level where we assign the rhs
            if idx_match is not None:
                if subkey not in base:
                    base[subkey] = []
                # lists have to be padded to size with dicts (nested lists are unsupported)
                while len(base[subkey]) <= idx:
                    base[subkey].append({})
            elif subkey not in base:
                base[subkey] = {}

            if last_key and idx_match is not None:
                base[subkey][idx] = rhs
            elif last_key:
                base[subkey] = rhs
            elif idx_match is not None:
                base = base[subkey][idx]
            else:
                base = base[subkey]


    log.debug(f'Config attributes provided: {raw_config}')
    return raw_config

@dataclass
class Command:
    @classmethod
    @property
    def name(cls):
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
        return name
    @classmethod
    @property
    def default_config_name(cls):
        return f"{cls.name}.conf.toml"

    def config_to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_args(cls, parsed_args):
        raw_config = _get_config_data(cls.default_config_name, parsed_args.config_file, parsed_args.vars)
        try:
            return from_dict(cls, raw_config)
        except (UnexpectedDataError, MissingValueError) as e:
            log.error(f"Applying configuration failed with: {e}")
            from pprint import pformat
            log.error(f"File and arguments provided this configuration:\n\n{pformat(raw_config)}\n")
            sys.exit(1)

    def main(self):
        raise NotImplementedError("Subclasses must implement main()")
