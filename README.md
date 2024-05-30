# `xconf` - Dataclasses and TOML for command configuration

## Demo

An example of how to use `xconf.Command`, `xconf.field`, and `xconf.config` is in `xconf_demo.py`. Run it to see its configuration keys.

```
$ python xconf_demo.py demo_command -h
demo_command: Demo command
usage: xconf_demo.py demo_command [-c CONFIG_FILE] [-h] [-v] [--dump-config] [vars ...]

positional arguments:
  vars                  Config variables set with 'key.key.key=value' notation

optional arguments:
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        Path to config file (default: demo_command.conf)
  -h, --help            Print usage information
  -v, --verbose         Enable debug logging
  --dump-config         Dump final configuration state as TOML and exit

configuration keys:
  collections
      dict[str, ExtendedThingie]
  collections.<str>.name
      str
     <ExtendedThingie>
  collections.<str>.extended
      bool
     <ExtendedThingie>
  either_one
      [int, str]
  should_bar
      bool
     (default: False)
  should_foo
      bool
    Whether demo should foo
  number_list
      list[int]
    List of favorite numbers
  float_list
      list[float]
    List of favorite floating-point numbers
  str_list
      list[str]
    List of favorite strings
  sequence
      list[ExtendedThingie]
     (default: [])
  sequence[#].name
      str
     <ExtendedThingie>
  sequence[#].extended
      bool
     <ExtendedThingie>
```

### Default config file

The command name, `demo_command`, is generated from the class name and used to find a default configuration file (`demo_command.conf`) in the current directory.

### Providing arguments at the command line

Any configuration key from the help output can be supplied on the command line in a `dotted.name=value` format.

For lists of primitive types (`str`, `int`, `float`), you can just use commas to separate the values on the right hand side of the `=`. Example: `number_list=1,2,3`.

To override a single entry in a list, use `some_name[#]` or `dotted[#].name=value` where `#` is an integer index will work. Example: `number_list[0]=99`

String values are bare (i.e. no quotation marks around `value`). Boolean values are case-insensitive `true`, `t`, or 1 for True, `false`, `f`, or 0 for False.

### Structuring the command

See `xconf_demo.py` for an example. Note that commands must subclass `xconf.Command` *and* apply the `@xconf.config` decorator. Options are defined by a hierarchy of dataclasses. (For uninteresting reasons, they aren't *strictly speaking* `import dataclass` dataclasses.)

## License

All code outside `xconf/vendor/` is provided under the terms of the [MIT License](./LICENSE), except for `xconf_demo.py` and `demo_command.conf`, which are released into the public domain for you to build off of.

Note that code under `xconf/vendor/` is used under the terms of the licenses listed there.
