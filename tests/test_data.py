def test_import_data_module():
    import importlib
    mod = importlib.import_module('my_tool.data')
    assert hasattr(mod, 'create_params')
    assert hasattr(mod, 'generate_event_data')
