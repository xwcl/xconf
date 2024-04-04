import xconf
import typing
import logging

log = logging.getLogger()

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
    should_bar : bool = xconf.field(default=False)
    should_foo : bool = xconf.field(help="Whether demo should foo")
    number_list : list[int] = xconf.field(help="List of favorite numbers")
    sequence : list[ExtendedThingie] = xconf.field(default_factory=list)

    def main(self):
        print('Got these collections:', self.collections)
        print('either_one =', self.either_one)
        print('should_bar =', self.should_bar)
        print('should_foo =', self.should_foo)
        print('number_list =', self.number_list)
        print('sequence =', self.sequence)

logging.basicConfig(level='DEBUG')
d = xconf.Dispatcher([DemoCommand])
d.main()
