"""Symbolic identities of the paper (sympy)."""
import pytest

sympy = pytest.importorskip('sympy')


def test_symbolic_identities():
    import symbolic_checks
    failed = [k for k, v in symbolic_checks.ok.items() if not v]
    assert not failed, failed
