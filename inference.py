import os
from openai import OpenAI
from env.environment import InventoryRestockEnvironment
from agent.baseline_agent import simple_agent
from env.grader import run_grader

# ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

iHF_TOKEN = HF_TOKEN or "dummy"

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def run():
    print("[START]")
    try:
        env = InventoryRestockEnvironment()

        # REQUIRED LLM CALL (once is enough)
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "initialize"}],
        )

        for task_id in [1, 2, 3]:
            print(f"\n[RUNNING TASK {task_id}]")

            result = run_grader(task_id, env, simple_agent)

            print(
                f"[RESULT] Task {task_id} | Score: {result.grade} | Passed: {result.passed}"
            )

    except Exception as e:
        print(f"[ERROR] {str(e)}")

    finally:
        print("[END]")


if __name__ == "__main__":
    run()
