"""Minimal configuration system to populate dataclasses with a
mix of config files in TOML and command-line arguments"""
import textwrap
import pathlib
import argparse
import re
import sys
import toml
import logging
from enum import Enum
from functools import partial
import typing
from toml import encoder
from .vendor import dacite
from .vendor.dacite import UnexpectedDataError, MissingValueError
from collections import defaultdict
import dataclasses

__all__ = (
    'field',
    'config',
    'config_to_dict',
    'dict_to_toml',
    'config_to_toml',
    'ConfigMismatch',
    'Command',
)

log = logging.getLogger(__name__)

config = partial(dataclasses.dataclass, kw_only=True)

def convert_bool(x):
    if x in (True, False):
        return x
    if x.lower() in ('true', 't', '1'):
        return True
    elif x.lower() in ('false', 'f', '0'):
        return False
    else:
        raise ValueError(f"Cannot convert {repr(x)} to boolean")


def convert_list(x, convert_type=lambda x: x):
    if isinstance(x, str):
        # handle escaped commas by temporarily replacing with an escaped value
        x = x.replace(r'\,', '\u0001')
        out = []
        for y in x.split(','):
            val = y.replace('\u0001', ',')
            try:
                val = convert_type(val)
                out.append(val)
            except ValueError:
                raise ValueError(f"Could not apply type conversion to {repr(val)} from list {repr(x)}")
    else:
        # type conversion was already handled and x is a list at this point
        assert isinstance(x, list)
        out = x
    return out

TYPE_HOOKS = {
    bool: convert_bool,
    int: int,
    float: float,
    list[str]: partial(convert_list, convert_type=str),
    list[int]: partial(convert_list, convert_type=int),
    list[float]: partial(convert_list, convert_type=float),
}
from_dict = partial(dacite.from_dict, config=dacite.Config(strict=True, type_hooks=TYPE_HOOKS, cast=[Enum]))
from_dict.__doc__ = dacite.from_dict.__doc__


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

fields = dataclasses.fields

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
        try:
            default = fld.default_factory()
        except TypeError:
            raise TypeError(f"Could not instantiate field {fld} from default factory")
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

    if dacite.types.is_union(fld_type) or dacite.types.is_generic_collection(fld_type):
        if dacite.types.is_union(fld_type):
            members = dacite.types.extract_generic(fld_type)
        else:
            members = [fld_type]
        for mtype in members:
            if dataclasses.is_dataclass(mtype):
                for k, v, h in list_fields(mtype, prefix=f'{prefix}{name}.', help_suffix=help_suffix+f' <{mtype.__name__}>'):
                    yield k, v, h
            elif dacite.types.is_generic_collection(mtype):
                orig_type = dacite.types.extract_origin_collection(mtype)
                res = dacite.types.extract_generic(mtype)
                if orig_type is dict: # mapping
                    collection_key_type, collection_value_type = res
                    if dataclasses.is_dataclass(collection_value_type):
                        for k, v, h in list_fields(collection_value_type, prefix=f'{prefix}{name}.<{collection_key_type.__name__}>.', help_suffix=help_suffix+f' <{format_field_type(collection_value_type)}>'):
                            yield k, v, h
                    else:
                        pass # output above with primitive types
                elif orig_type is list:
                    collection_entry_type, = res
                    if dataclasses.is_dataclass(collection_entry_type):
                        for k, v, h in list_fields(collection_entry_type, prefix=f'{prefix}{name}[#].', help_suffix=help_suffix+f' <{format_field_type(collection_entry_type)}>'):
                            yield k, v, h
                    else:
                        pass # output above with primitive types
                else:
                    pass # unhandled parameterized generic collection (tuples? sets?)
            else:
                # no need for additional docs for primitive types
                continue
    else:
        if dataclasses.is_dataclass(fld_type):
            for k, v, h in list_fields(fld_type, prefix=f'{prefix}{name}.'):
                yield k, v, h

def add_subparser_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-c", "--config-file",
        action="append",
        default=[],
        help=f"Path to config file, repeat to merge multiple, last one wins for repeated top-level keys",
    )
    parser.add_argument("-h", "--help", action='store_true', help="Print usage information")
    parser.add_argument("-v", "--verbose", action='store_true', help="Enable debug logging")
    parser.add_argument("--dump-config", action='store_true', help="Dump final configuration state as TOML and exit")
    parser.add_argument("vars", nargs='*', help="Config variables set with 'key.key.key=value' notation")
    return parser

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
            if command_cls.get_name() is None or command_cls.get_name() in names_to_subparsers:
                raise Exception(f"Invalid command name for {command_cls}")
            subp = subps.add_parser(command_cls.get_name(), add_help=False)
            names_to_subparsers[command_cls.get_name()] = subp
            subp.set_defaults(command_cls=command_cls)
            add_subparser_arguments(subp)

        args = parser.parse_args()
        if args.command_cls is None:
            parser.print_help()
            sys.exit(1)
        if args.help:
            print_help(args.command_cls, names_to_subparsers[args.command_cls.get_name()])
            sys.exit(0)
        self.configure_logging('DEBUG' if args.verbose else 'INFO')
        command = args.command_cls.from_args(args)
        if args.dump_config:
            print(config_to_toml(command))
            sys.exit(0)
        return command.main()

def print_help(cls, parser):
    print(f"{sys.argv[0]}:")
    parser.print_help()
    print("\nconfiguration keys:")
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

def _get_config_data(default_config_path, config_file_paths, cli_args):
    raw_config = {}
    if len(config_file_paths):
        for config_file_path in config_file_paths:
            loaded_config = load_config(config_file_path)
            raw_config.update(loaded_config)
    elif default_config_path is not None:
        try:
            raw_config = load_config(default_config_path)
        except FileNotFoundError:
            pass

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

def config_to_dict(inst):
    return dataclasses.asdict(inst)

def dict_to_toml(config_dict):
    return toml.dumps(config_dict, encoder=TomlEnumEncoder())

def config_to_toml(inst):
    config_dict = config_to_dict(inst)
    return dict_to_toml(config_dict)

class ConfigMismatch(Exception):
    def __init__(self, original_exception: Exception, raw_config: dict):
        self.original_exception = original_exception
        self.raw_config = raw_config

@dataclasses.dataclass
class Command:
    @classmethod
    def get_name(cls):
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
        return name

    @classmethod
    def get_default_config_prefix(cls) -> pathlib.Path:
        return pathlib.Path('./')

    @classmethod
    def get_default_config_path(cls) -> pathlib.Path:
        return cls.get_default_config_prefix() / (cls.get_name() + ".conf")

    def config_to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_config(
        cls,
        config_path_or_paths: typing.Union[str,list[str]]=None,
        config_dict: dict= None,
        settings_strs: list[str]=None,
    ):
        '''Initialize a class instance using config files from disk or a dictionary
        of options

        Parameters
        ----------
        config_path :
        '''

        config_paths = []
        if isinstance(config_path_or_paths, str):
            config_paths.append(config_path_or_paths)
        elif isinstance(config_path_or_paths, list):
            config_paths.extend(config_path_or_paths)
        if settings_strs is None:
            settings_strs = []
        raw_config = _get_config_data(cls.get_default_config_path(), config_paths, settings_strs)
        if config_dict is not None:
            for key, value in config_dict.items():
                if key in raw_config:
                    old_val = raw_config[key]
                    log.info(f"Using provided value {value} for {key} which was set to {old_val} in the loaded config files")
            raw_config.update(config_dict)
        try:
            instance = from_dict(cls, raw_config)
        except (UnexpectedDataError, MissingValueError) as e:
            raise ConfigMismatch(e, raw_config)
        return instance

    @classmethod
    def run(cls):
        parser = argparse.ArgumentParser(add_help=False)
        add_subparser_arguments(parser)
        args = parser.parse_args()
        if args.help:
            print_help(cls, parser)
            sys.exit(0)
        logging.basicConfig(level='WARN')
        logging.getLogger('__main__').setLevel('DEBUG' if args.verbose else 'INFO')
        command = cls.from_args(args)
        if args.dump_config:
            print(config_to_toml(command))
            sys.exit(0)
        command.main()

    @classmethod
    def from_args(cls, parsed_args):
        try:
            return cls.from_config(config_path_or_paths=parsed_args.config_file, settings_strs=parsed_args.vars)
        except ConfigMismatch as e:
            log.error(f"Applying configuration failed with: {e.original_exception}")
            from pprint import pformat
            log.error(f"File and arguments provided this configuration:\n\n{pformat(e.raw_config)}\n")
            sys.exit(1)

    def main(self):
        raise NotImplementedError("Subclasses must implement main()")
