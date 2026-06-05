from streamlit.testing.v1 import AppTest


def test_dashboard_renders_starter_mixture() -> None:
    app = AppTest.from_file("app.py")

    app.run(timeout=60)

    assert not app.exception
    assert len(app.title) == 1
    assert "SCINTILLA" in app.title[0].value
    assert {metric.label for metric in app.metric} >= {"Cs-137", "Co-60"}
    assert app.radio[0].options == [
        "Synthetic mixture",
        "Demo CSV upload",
        "HPGe benchmark CSV upload",
    ]
