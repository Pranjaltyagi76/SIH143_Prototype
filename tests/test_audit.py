"""Phase 9 tests for the pre-demo audit.

An audit that silently checks nothing is worse than no audit, because it
produces a clean report that nobody re-examines. The first version of the
language check did exactly that -- a mangled regex hunting for a byte that never
occurs -- and reported a confident PASS over zero literals (P-24).

So these tests plant violations and assert the audit catches them.
"""

from __future__ import annotations

from scripts.audit import (
    BANNED,
    Audit,
    _string_literals,
    check_language,
    check_seed_threading,
)


def test_string_literal_extraction_finds_literals():
    """The check that silently did nothing."""
    src = 'msg = "this vessel is guilty"\nn = 1  # culprit was here\n'
    literals = list(_string_literals(src))
    assert "this vessel is guilty" in literals


def test_string_literal_extraction_ignores_comments_and_identifiers():
    """Internal vocabulary is not the control. Banning 'culprit' as a variable
    name would make the audit cry wolf, which is worse than not running it."""
    src = 'culprit_mmsi = "219000000"  # the culprit was removed from AIS\n'
    literals = list(_string_literals(src))
    assert not any("culprit was" in lit for lit in literals)


def test_string_literal_extraction_handles_triple_quotes_and_fstrings():
    src = 'a = f"""a {x} guilty verdict"""\nb = \'single\'\n'
    literals = list(_string_literals(src))
    assert any("guilty" in lit for lit in literals)
    assert "single" in literals


def test_malformed_source_does_not_crash_the_audit():
    assert list(_string_literals("def broken(:\n")) == []


def test_language_check_passes_on_the_real_repository():
    a = Audit()
    check_language(a)
    assert not a.failed, a.failed


def test_language_check_catches_a_planted_violation(tmp_path, monkeypatch):
    """Proves the check bites, rather than trusting that it does."""
    import scripts.audit as audit

    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "bad.py").write_text(
        'REASON = "this vessel is guilty"\n', encoding="utf-8"
    )
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    a = Audit()
    audit.check_language(a)
    assert a.failed
    assert "guilty" in a.failed[0][1]


def test_seed_check_ignores_the_generator_type():
    """np.random.Generator is a TYPE annotation, not a source of randomness.
    Flagging it made the audit report a failure on correct code."""
    a = Audit()
    check_seed_threading(a)
    assert not a.failed, a.failed


def test_banned_list_covers_the_determinations_we_refuse_to_make():
    joined = " ".join(BANNED)
    for concept in ("guilty", "culprit", "polluter", "confirmed"):
        assert concept in joined
