# grader.py
# Scores the agent's performance on each task from 0.0 to 1.0
#
# HOW GRADING WORKS:
# Each task has a known "worst possible score" and a "target (good) score".
# We normalize the agent's actual score into the 0.0–1.0 range.
#
# Score 0.0 = agent did terribly (stockouts every day, or did nothing)
# Score 1.0 = agent perfectly managed inventory across the episode
#
# The grader is DETERMINISTIC for Task 1 (fixed demand),
# and bounded for Tasks 2 and 3 (random demand).

from dataclasses import dataclass
from typing import List


@dataclass
class GradeResult:
    task_id: int
    task_name: str
    raw_score: float        # total accumulated reward from the episode
    grade: float            # normalized 0.0–1.0
    passed: bool            # True if grade >= 0.5
    summary: str


# ── Score bounds per task (pre-calculated based on max days × demand) ────────
# These are the min/max total rewards an agent can realistically get.

SCORE_BOUNDS = {
    1: {
        "name": "Easy - Stable Demand",
        "min_score": -80.0,    # if agent never restocks: 20 days × (5 unmet × -2)
        "target_score": 85.0,  # if agent sells 5 units/day for 20 days × +1 each
    },
    2: {
        "name": "Medium - Variable Demand",
        "min_score": -150.0,
        "target_score": 120.0,
    },
    3: {
        "name": "Hard - Demand Spikes",
        "min_score": -250.0,
        "target_score": 200.0,
    },
}


def grade_episode(task_id: int, total_reward: float) -> GradeResult:
    """
    Convert a raw episode reward into a normalized 0.0–1.0 grade.

    Formula:
        grade = (total_reward - min_score) / (target_score - min_score)
        grade = clamp(grade, 0.0, 1.0)

    This means:
      - If agent scores exactly min_score → grade = 0.0
      - If agent scores exactly target_score → grade = 1.0
      - Anything above target_score → grade = 1.0 (capped)
      - Anything below min_score → grade = 0.0 (capped)
    """
    bounds = SCORE_BOUNDS.get(task_id)
    if bounds is None:
        raise ValueError(f"Unknown task_id: {task_id}. Valid: 1, 2, 3")

    min_s = bounds["min_score"]
    target_s = bounds["target_score"]
    name = bounds["name"]

    # Normalize
    grade = (total_reward - min_s) / (target_s - min_s)

    # strict clamp BEFORE rounding
    grade = max(0.0001, min(0.9999, grade))

    # round
    grade = round(grade, 4)

    # strict clamp AGAIN after rounding (very important)
    grade = max(0.0001, min(0.9999, grade))

    passed = grade >= 0.5

    if grade >= 0.8:
        summary = "Excellent! Agent managed inventory very well."
    elif grade >= 0.5:
        summary = "Acceptable. Agent passed with some stockouts or overstock."
    elif grade >= 0.2:
        summary = "Poor. Agent struggled to balance supply and demand."
    else:
        summary = "Failed. Agent barely restocked or caused constant stockouts."

    return GradeResult(
        task_id=task_id,
        task_name=name,
        raw_score=round(total_reward, 2),
        grade=grade,
        passed=passed,
        summary=summary,
    )


def run_grader(task_id: int, env, agent_fn) -> GradeResult:
    """
    Run one full episode and return the grade.

    Parameters:
        task_id   - which task (1, 2, or 3)
        env       - the InventoryRestockEnvironment instance
        agent_fn  - a function(observation) -> RestockAction

    Returns:
        GradeResult with grade between 0.0 and 1.0
    """
    obs = env.reset(task_id, seed=task_id)

    while not obs.done:
        action = agent_fn(obs)
        obs = env.step(action)

    total_reward = env.state.total_reward
    return grade_episode(task_id, total_reward)
