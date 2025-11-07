def test_import_analysis_module():
    import importlib
    mod = importlib.import_module('my_tool.analysis')
    assert hasattr(mod, 'run_analysis')
