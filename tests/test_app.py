from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_loads_and_scenario_prediction():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.metric and '%' in app.metric[0].value
    app.selectbox(key='prediction_model').select('Random Forest').run(timeout=30)
    assert not app.exception and app.metric


def test_initial_result_all_models_and_downloads_are_available():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
    assert not app.exception
    assert app.metric and '%' in app.metric[0].value
    for name in ['Logistic Regression', 'Decision Tree', 'Prevalence Baseline', 'BMA']:
        app.selectbox(key='prediction_model').select(name).run(timeout=30)
        assert not app.exception
    assert len(app.tabs) == 6
    assert len(app.get('download_button')) >= 9


def test_extrapolation_warning_for_extreme_temperature():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
    app.number_input[0].set_value(80.0)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert any('outside training range' in warning.value for warning in app.warning)
