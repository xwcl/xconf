# `xconf` - Dataclasses and TOML for command configuration

## Demo

An example of how to use `xconf.Command`, `xconf.field`, and `xconf.config` is in `demo.py`. Run it to see its configuration keys.

```
$ python demo.py demo_command -h
demo_command: Demo command
usage: demo.py demo_command [-c CONFIG_FILE] [-h] [-v] [vars ...]

positional arguments:
  vars                  Config variables set with 'key.key.key=value' notation

optional arguments:
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        Path to config file (default: demo_command.conf.toml)
  -h, --help            Print usage information
  -v, --verbose         Enable debug logging
configuration keys:
  collections
      dict[str, ExtendedThingie]
  collections.<str>.name
      str
  collections.<str>.extended
      bool
  either_one
      [int, str]
  should_bar
      bool
  should_foo
      bool
    Whether demo should foo
```

### Default config file

The command name, `demo_command`, is generated from the class name and used to find a default configuration file in the current directory.

### Structuring the command

These options are defined by a hierarchy of dataclasses. (For uninteresting reasons, they aren't *strictly speaking* `import dataclass` dataclasses.)

```python
@xconf.config
class Thingie:
    name : str = xconf.field()

@xconf.config
class ExtendedThingie(Thingie):
    extended : bool = xconf.field()

@xconf.config
class DemoCommand(xconf.Command):
    """Demo command"""
    collections : dict[str, ExtendedThingie] = xconf.field()
    either_one : typing.Union[int, str] = xconf.field()
    should_bar : bool = xconf.field()
    should_foo : bool = xconf.field(help="Whether demo should foo")

    def main(self):
        print('Got these collections:', self.collections)
        print('either_one =', self.either_one)
        print('should_bar =', self.should_bar)
        print('should_foo =', self.should_foo)
```

Note that commands must subclass `xconf.Command` *and* apply the `@xconf.config` decorator.
