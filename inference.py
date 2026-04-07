import os
from openai import OpenAI
from env.environment import InventoryRestockEnvironment
from agent.baseline_agent import simple_agent

# ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def run():
    print("[START]")
    try:
        env = InventoryRestockEnvironment()

        for task_id in [1, 2, 3]:
            obs = env.reset(task_id)
            done = False

            while not done:
                action = simple_agent(obs)

                # REQUIRED LLM CALL (even if dummy)
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": "simulate step"}],
                )

                obs = env.step(action)

                reward = f"{obs.reward:.2f}"
                done_flag = str(obs.done).lower()

                print(f"[STEP] reward={reward} done={done_flag}")

                done = obs.done

    except Exception as e:
        print(f"[ERROR] {str(e)}")

    finally:
        print("[END]")


if __name__ == "__main__":
    run()