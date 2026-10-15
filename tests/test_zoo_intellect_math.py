"""BoundedMathRubric: the scores of math_verify, and an enforceable bound on
hangs that the alarm cannot interrupt."""
import asyncio
import time

import pytest
import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer

import tasks.zoo.intellect_math as zoo
from tasks.zoo.intellect_math import BoundedMathRubric, bounded_verify_response


@pytest.fixture
def rubric():
    r = BoundedMathRubric(parser=vf.Parser(extract_boxed_answer))
    yield r
    for proc in (getattr(r.executor, "_processes", None) or {}).values():
        proc.kill()
    r.executor.shutdown(wait=False)


def score_one(rubric, response, answer):
    completion = [vf.AssistantMessage(content=f"The answer is \\boxed{{{response}}}.")]
    return asyncio.run(rubric.correct_answer(rubric.parser, completion, answer))


def test_scores_match_math_verify(rubric):
    assert score_one(rubric, "2", "2") == 1.0
    assert score_one(rubric, "8", "11") == 0.0
    assert score_one(rubric, "\\frac{1}{2}", "0.5") == 1.0


def test_prose_fallback_scores_without_hanging(rubric):
    # No \boxed{} in the completion: the non-strict extractor falls back to
    # the full text, the input that can hang the scorer. Must return,
    # whatever the score.
    prose = ("## Step 1: To find $P(A_n)$ we compute $1-(1-a)^n$ and then "
             "simplify $\\left(\\frac{1-a}{a}\\right)^2$ carefully.\n") * 40
    completion = [vf.AssistantMessage(content=prose)]
    t0 = time.monotonic()
    reward = asyncio.run(rubric.correct_answer(
        rubric.parser, completion, " \\left(\\frac{1-a}{a}\\right)^2 "))
    assert time.monotonic() - t0 < 60
    assert reward in (0.0, 1.0)


def _alarm_immune_hang(conn, response, answer, max_chars, timeout_seconds):
    # Simulate a C-level loop SIGALRM can't interrupt: block the signal.
    import signal
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    while True:
        time.sleep(0.1)


def test_alarm_immune_hang_is_killed(monkeypatch):
    monkeypatch.setattr(zoo, "_child_verify", _alarm_immune_hang)
    monkeypatch.setattr(zoo, "KILL_AFTER_SECONDS", 3.0)
    t0 = time.monotonic()
    reward, elapsed = bounded_verify_response("x", "y", 50_000, 1)
    assert reward == 0.0
    assert time.monotonic() - t0 < 15  # killed at the bound instead of hanging


def test_verify_survives_child_crash(monkeypatch):
    def _dies(conn, *a):
        raise SystemExit(1)
    monkeypatch.setattr(zoo, "_child_verify", _dies)
    reward, _ = bounded_verify_response("x", "y", 50_000, 1)
    assert reward == 0.0
