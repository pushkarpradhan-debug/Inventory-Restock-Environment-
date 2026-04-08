# grader.py
# Scores the agent's performance on each task from 0.01 to 0.99
#
# HOW GRADING WORKS:
# Each task has a known "worst possible score" and a "target (good) score".
# We normalize the agent's actual score into the 0.01–0.99 range.
#
# Score 0.01 = agent did terribly (stockouts every day, or did nothing)
# Score 0.99 = agent perfectly managed inventory across the episode
#
# The grader is DETERMINISTIC for Task 1 (fixed demand),
# and bounded for Tasks 2 and 3 (random demand).
 # grader.py
from dataclasses import dataclass
from typing import Dict


@dataclass
class GradeResult:
    task_id: int
    task_name: str
    raw_score: float
    grade: float
    passed: bool
    summary: str


SCORE_BOUNDS = {
    1: {
        "name": "Easy - Stable Demand",
        "min_score": -80.0,
        "target_score": 85.0,
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
    bounds = SCORE_BOUNDS.get(task_id)
    if bounds is None:
        raise ValueError(f"Unknown task_id: {task_id}. Valid: 1, 2, 3")

    min_s = bounds["min_score"]
    target_s = bounds["target_score"]
    name = bounds["name"]

    # Normalize
    grade = (total_reward - min_s) / (target_s - min_s)

    # STRICT SAFE RANGE (avoid 0 and 1 completely)
    grade = max(0.011, min(0.989, grade))

    grade = round(grade, 4)

    # clamp again after rounding
    grade = max(0.011, min(0.989, grade))

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


# ✅ NEW FUNCTION (THIS FIXES YOUR FAILURE)
def run_grader(env, agent_fn) -> Dict[str, float]:
    results = {}

    for task_id in [1, 2, 3]:
        env_local = type(env)()

        obs = env_local.reset(task_id=task_id, seed=task_id)

        while not obs.done:
            action = agent_fn(obs)
            obs = env_local.step(action)

        total_reward = env_local.state.total_reward
        grade_result = grade_episode(task_id, total_reward)

        results[f"task_{task_id}"] = float(grade_result.grade)

    return results
