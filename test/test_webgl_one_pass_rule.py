"""Which LayerNorm/softmax form WebGL runs, at the shapes it was measured at.

Each case is a point from the measurement recorded above `_one_pass` in `_core.py`, with
the form that was actually faster there. The rule is checked against the data, not the
other way round: if a later measurement moves the line, these cases move with it.
"""
import pytest

from webtorch import _core as C


@pytest.mark.parametrize("rows,width,one_pass_faster", [
    (1, 768, True), (2, 768, True), (4, 768, False), (8, 768, False), (519, 768, False),
    (16, 173, True), (64, 173, False), (6228, 173, False),
    (4, 1024, False), (8, 1024, False),
    (1, 4096, False), (512, 4096, False),
])
def test_the_rule_picks_the_measured_winner(rows, width, one_pass_faster):
    assert C._one_pass(rows, width) is one_pass_faster


def test_the_choice_depends_on_width_not_only_on_rows():
    # Sixteen rows of a 173-wide softmax are one-pass work; one row of a 4096-wide one is not.
    assert C._one_pass(16, 173) and not C._one_pass(1, 4096)
