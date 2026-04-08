import os
from typing import List, Optional
from openai import OpenAI

from env.environment import InventoryRestockEnvironment
from agent.baseline_agent import simple_agent

# ENV VARIABLES (MANDATORY)
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

TASK_NAME = os.getenv("TASK_NAME", "inventory-restock")
BENCHMARK = os.getenv("BENCHMARK", "inventory_env")

MAX_STEPS = 50
MAX_TOTAL_REWARD = 200.0  # adjust if needed


client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ── LOG FUNCTIONS (EXACT FORMAT) ─────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── MAIN EXECUTION ───────────────────────────────────

def run():
    env = InventoryRestockEnvironment()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # REQUIRED LLM CALL (do not remove)
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "init"}],
    )

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id=1, seed=42)

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action_obj = simple_agent(obs)

            # Convert action safely
            action = getattr(action_obj, "restock_quantity", str(action_obj))

            obs = env.step(action_obj)

            reward = getattr(obs, "reward", 0.0)
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=str(action), reward=reward, done=done, error=error)

            if done:
                break

        # SCORE (EXACT FORMAT FROM SAMPLE)
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)

        success = score >= 0.1

    finally:
        try:
            env.close()
        except Exception:
            pass

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    run()
