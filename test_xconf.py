import xconf
import xconf_demo
import pytest

def test_xconf_demo():
    with pytest.raises(xconf.ConfigMismatch):
        xconf_demo.DemoCommand.from_config(config_path_or_paths=[], settings_strs=[])
    inst = xconf_demo.DemoCommand.from_config(config_path_or_paths=['./demo_command.conf.toml'], settings_strs=[])
    inst = xconf_demo.DemoCommand.from_config(config_path_or_paths=['./demo_command.conf.toml'], settings_strs=['number_list=1,2,3,4,5'])
    assert inst.number_list == [1, 2, 3, 4, 5]
    inst = xconf_demo.DemoCommand.from_config(config_path_or_paths=['./demo_command.conf.toml'], settings_strs=['should_foo=1'])
    assert inst.should_foo
    inst = xconf_demo.DemoCommand.from_config(config_path_or_paths=['./demo_command.conf.toml'], settings_strs=['should_foo=true'])
    assert inst.should_foo
    inst = xconf_demo.DemoCommand.from_config(config_path_or_paths=['./demo_command.conf.toml'], settings_strs=['should_foo=True'])
    assert inst.should_foo
    inst = xconf_demo.DemoCommand.from_config(config_path_or_paths=['./demo_command.conf.toml'], settings_strs=['float_list=4,8'])
    assert inst.float_list == [4.0, 8.0]
