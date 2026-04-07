---
title: Inventory Restock Environment
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 📦 Inventory Restock Environment (OpenEnv)

> A real-world reinforcement learning environment for inventory optimization under uncertainty.

---

## 🚀 Live Demo

👉 https://pushkarpradhan-inventory-restock-env-fin.hf.space/docs

---

## ⚡ Quick Test (30 seconds)

1. Open `/docs`
2. Click `/reset`
3. Try:
   - `/step` → manual decision
   - `/auto_step` → AI agent decision
4. View `/state`

---

## 🧠 Why This Matters

Inventory management is a critical real-world problem:

- 📉 Stockouts → lost revenue
- 📦 Overstock → wasted capital
- ⏳ Lead time → delayed decisions
- 📊 Uncertainty → complex planning

This project simulates these challenges in a **controlled RL environment**.

---

## 🎯 Key Highlights

- 🔹 Real-world simulation (not toy problem)
- 🔹 Multi-difficulty tasks (Easy → Hard)
- 🔹 Lead-time aware decision making
- 🔹 Manual + AI agent control
- 🔹 Fully deployed API (FastAPI + Docker + Hugging Face)

---
---

## 🧾 What This Does (Plain English)

Imagine you run a small shop.

Every day:

* Customers arrive and demand products
* You must decide how much stock to order
* Orders take time to arrive (lead time)

If you order too little → you lose sales
If you order too much → you waste money

This environment simulates that decision-making process and evaluates how well an agent handles these trade-offs.

---

## 🧠 Problem Motivation

In real-world retail systems:

* Demand is **uncertain and fluctuating**
* Deliveries have **delays (lead time)**
* Overstock ties up **capital and storage**
* Understock leads to **lost revenue and customer dissatisfaction**

This project models these challenges in a controlled simulation, enabling **decision strategies to be tested and evaluated**.

---

## ⚙️ Environment Design

### 🔹 State (Observation)

| Field         | Description                             |
| ------------- | --------------------------------------- |
| current_stock | Units currently available               |
| daily_demand  | Demand for the day                      |
| lead_time     | Delay for incoming orders               |
| pending_order | Units already ordered but not delivered |
| day           | Current timestep                        |

---

### 🔹 Action

| Field            | Description                     |
| ---------------- | ------------------------------- |
| restock_quantity | Number of units to order (0–50) |

---

### 🔹 Reward Function

| Component         | Value | Description                    |
| ----------------- | ----- | ------------------------------ |
| Sale reward       | +1    | Reward per unit sold           |
| Stockout penalty  | -2    | Heavy penalty per unmet demand |
| Overstock penalty | -0.1  | Light penalty per excess unit  |

> The reward function strongly prioritizes **avoiding stockouts**, while still discouraging unnecessary overstock.

---

## 🎯 Task Difficulty Levels

### 🟢 Task 1: Easy — Stable Demand

* Constant demand
* Short lead time
* Predictable behavior

---

### 🟡 Task 2: Medium — Variable Demand

* Fluctuating demand
* Moderate uncertainty
* Requires adaptive strategy

---

### 🔴 Task 3: Hard — Demand Spikes

* Sudden spikes in demand
* Long lead time
* High penalty for mistakes

> This task tests **robustness, planning, and foresight**, not just reactivity.

---

## 🤖 Baseline Agent Strategy

The baseline agent uses a **rule-based inventory control policy** inspired by real-world systems.

### Core Idea:

> Maintain sufficient stock to satisfy **future demand during lead time**, with a safety buffer.

---

### 📌 Logic Summary

1. Estimate future demand:

   ```
   demand_forecast = daily_demand × lead_time
   ```

2. Add safety buffer:

   ```
   target_stock = demand_forecast + buffer
   ```

3. Restock only when necessary:

   ```
   if current_stock < target_stock:
       order = target_stock - current_stock
   ```

---

### 💡 Key Strengths

* ✔ Accounts for **lead time delays**
* ✔ Handles **uncertainty and demand spikes**
* ✔ Balances stockout vs overstock trade-offs
* ✔ Simple, interpretable, and efficient
* ✔ Avoids over-engineering while maintaining robustness

---

## 📊 Baseline Performance

| Task   | Grade    | Status |
| ------ | -------- | ------ |
| Easy   | ~1.0     | ✅ PASS |
| Medium | ~0.8–0.9 | ✅ PASS |
| Hard   | ~0.5–0.6 | ✅ PASS |

> The agent demonstrates **consistent performance across all difficulty levels**, successfully handling high-uncertainty scenarios.

---

## 🧪 How to Run

### ▶️ Run full simulation

```bash
python inference.py
```

---

### ▶️ Run grading

```python
from env.grader import run_grader
from env.environment import InventoryRestockEnvironment
from agent.baseline_agent import simple_agent

env = InventoryRestockEnvironment()

for task in [1, 2, 3]:
    print(run_grader(task, env, simple_agent))
```

---

## 🏗️ Project Structure

```
inventory_project/
│
├── env/                # Environment logic
│   ├── environment.py
│   ├── models.py
│   ├── grader.py
│
├── agent/              # Baseline agent
│   ├── baseline_agent.py
│
├── inference.py        # Simulation runner
├── deployment/         # API + Docker setup
```

---

## 🌟 Key Highlights

* ✅ Clean and modular architecture (env / agent / deployment separation)
* ✅ Fully OpenEnv-compatible design
* ✅ Real-world problem modeling
* ✅ Handles uncertainty and delayed effects
* ✅ Strong baseline without complex ML
* ✅ Focus on clarity, correctness, and robustness

---

## 🔮 Future Improvements

* Adaptive agents using reinforcement learning
* Multi-product inventory systems
* Demand forecasting using time-series models
* Cost-aware optimization strategies

---

## 🔁 Reproducibility

This environment is fully deterministic for evaluation:
- Fixed seed per task ensures consistent results
- Grading outputs are reproducible across runs

## 🚀 Deployment

This project is deployed using:
- FastAPI for environment serving
- Docker for containerization
- Hugging Face Spaces for hosting

## 🏁 Conclusion

This project demonstrates how **simple, well-designed logic** can effectively solve a complex decision-making problem under uncertainty.

> Instead of relying on heavy ML models, the focus is on **understanding system dynamics and designing robust policies**.

---

## 🙌 Acknowledgment

Developed as part of an OpenEnv hackathon to explore reinforcement learning environments and decision-making systems.

---
