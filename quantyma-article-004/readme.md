# Scenario-Based Dynamic Modeling Framework for Multi-Vector Hybrid Microgrids in Isolated Amazonian Defense Outposts
**Author:** Bruno Priantti | **Quantyma Research** **Field:** Complex Systems, Dynamic Systems, Time-Series Analysis, Time Series

---
## Abstract
Comprehensive operational planning of hybrid microgrids in isolated equatorial environments, such as the Amazon rainforest, requires simulation tools capable of capturing environmental and logistical constraints across multiple energy vectors. Traditional deterministic approaches often obscure operational stress by relying on time-averaged assumptions. This paper presents a scenario-based dynamic modeling framework for multi-vector hybrid microgrids, integrating photovoltaic generation, lithium-ion battery storage, hydrogen-based seasonal storage, and a diesel backup generator under localized boundary conditions. A non-homogeneous Markov chain combined with solar geometry models is used to generate stochastic irradiance profiles, while a multi-layer demand model represents the volatile load of a remote Special Border Platoon. The framework is evaluated over an annual horizon using a rule-based dispatch strategy, revealing system-level stress patterns such as battery depletion events, hydrogen saturation, and fuel scarcity under constrained logistics. The source code is released as an open computational environment for microgrid design and control research.


---

## Setup, Hardware & Execution

### 1. Hardware & OS Environment (Original Workstation)
The results were obtained using this specific configuration to ensure reproducibility:
- **Host Model:** Lenovo ThinkPad E14 Gen 5 (21JS000SBO)
- **OS:** Debian GNU/Linux 13.4 (Bookworm)
- **CPU:** AMD Ryzen 7 7730U (8-Core / 16-Threads)
- **GPU:** AMD Radeon Graphics (Barcelo)
- **RAM:** 24GB LPDDR4
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
