# Resilience Analysis of Recurrent Neural Networks in Predicting High-Turbulence Chaotic Systems under Stochastic Noise
**Author:** Bruno Priantti | **Quantyma Research** **Field:** Deep Learning, Dynamical Systems, Chaos Theory

---

## Abstract
This research evaluates the predictive performance and topological preservation of Recurrent Neural Network architectures (RNN, GRU, and LSTM) under high-turbulence conditions. Using the Lorenz-63 system as a benchmark with 20% additive Gaussian noise, we assess the ability of these models to maintain the integrity of the strange attractor despite significant observational uncertainty.

---

## Setup, Hardware & Execution

### 1. Hardware & OS Environment (Original Workstation)
The results were obtained using this specific configuration to ensure reproducibility:
- **Host Model:** Lenovo ThinkPad E14 Gen 5 (21JS000SBO)
- **OS:** Debian GNU/Linux 13.4 (Bookworm)
- **CPU:** AMD Ryzen 7 7730U (8-Core / 16-Threads)
- **GPU:** AMD Radeon Graphics (Barcelo)
- **RAM:** 16GB LPDDR4
- **Environment:** Local Research Station

### 2. Installation & Execution Script
Run the commands below to configure the environment and start the experiment:

```bash
# Update system, install dependencies, configure Python environment and run
sudo apt update && sudo apt install curl git -y && \
curl -fsSL [https://pyenv.run](https://pyenv.run) | bash && \
export PATH="$HOME/.pyenv/bin:$PATH" && \
eval "$(pyenv init -)" && \
eval "$(pyenv virtualenv-init -)" && \
pyenv install 3.11.8 && \
pyenv local 3.11.8 && \
curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 - && \
export PATH="$HOME/.local/bin:$PATH" && \
poetry config virtualenvs.in-project true && \
poetry install && \
poetry run jupyter notebook src/experiment.ipynb
