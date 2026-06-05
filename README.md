# Investigating Reward Design and Alignment Failures in Reinforcement Learning

**Author:** Aditya  
**Affiliations:** BS in Data Science and Applications, IIT Madras | BE (Hons) in AI & Machine Learning, Chandigarh University

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Stable Baselines3](https://img.shields.io/badge/Stable%20Baselines3-2.x-purple.svg)](https://stable-baselines3.readthedocs.io/)

## Technical Report
**[Read the full Technical Report (PDF)](report.pdf)**

This repository contains the experimental codebase and evaluation data for an empirical investigation into the alignment problem and reward mis-specification in reinforcement learning.

Using Proximal Policy Optimization (PPO), the project studies how continuous-state agents behave when their mathematically defined reward functions diverge from the intended human objective across two environments: `MountainCar-v0` and `LunarLander-v3`.

### Key Findings Overview
- The alignment tax: heavily shaped, dense rewards can create local optima that cause policy collapse and avoidance of the terminal state.
- Specification gaming: agents can learn a hovering exploit by optimizing a proxy reward, producing poor true-task performance under the real environment objective.
- See the technical report for the full learning curves and trajectory efficiency analysis.

---

## Repository Structure

```text
rl_reward_research/
├── analysis/                 # Evaluation and Matplotlib visualization pipelines
│   ├── plot_results.py
│   └── plot_lunar_lander.py
├── experiments/              # Training pipelines and callbacks
│   ├── train_mountain_car.py
│   └── train_lunar_lander.py
├── logs/                     # Evaluation outputs and generated figures
│   ├── lunar_lander/
│   └── mountain_car/
├── report.pdf                # Full academic write-up and methodology
├── requirements.txt          # Python dependencies
├── src/                      # Source code for environment manipulation
│   └── wrappers.py
└── README.md
```

## Local Installation & Usage
This project was built and tested natively on Apple Silicon. A virtual environment is recommended.

```bash
# Clone the repository
git clone https://github.com/Optimus2007/rl-alignment-and-reward-hacking.git
cd rl-alignment-and-reward-hacking

# Create and activate a virtual environment
python3 -m venv env
source env/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

## Reproducing the Experiments
To run the training loops:

```bash
python experiments/train_mountain_car.py
python experiments/train_lunar_lander.py
```

To parse the generated evaluation arrays and render the Matplotlib figures:

```bash
python analysis/plot_results.py
python analysis/plot_lunar_lander.py
```
