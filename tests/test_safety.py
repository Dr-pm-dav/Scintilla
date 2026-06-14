from scintilla.safety import safety_brief


def test_safety_brief_handles_public_benchmark_labels() -> None:
    brief, sources = safety_brief(["214Bi", "137Cs"])

    assert "214Bi" in brief
    assert "137Cs" in brief
    assert sources

