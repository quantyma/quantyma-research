# Anomaly Detection in Industrial Gearbox Systems Using Acoustic Signals Under Highly Noise Background
**Author:** Bruno Priantti | **Quantyma Research** **Field:** Anomaly Detection, Signal PreProcessing, Industrial Machine Monitoring, Time Series

---
## Abstract
This research investigates robust anomaly detection in industrial gearbox systems under high environmental noise conditions. Using a public industrial benchmark dataset, the proposed framework combines signal conditioning techniques with unsupervised subspace learning to preserve fault-related acoustic patterns despite significant observational uncertainty. The methodology employs bandpass filtering and Hilbert Transform envelope extraction to isolate amplitude-modulated components associated with gear meshing irregularities, followed by Principal Component Analysis (PCA) for anomaly detection through reconstruction error analysis. Experimental results demonstrated complete separation between normal and anomalous operating conditions, achieving AUC, Precision, and Recall scores of 1.0 across multiple noise intensities. The findings indicate that combining physically grounded signal processing with PCA-based modeling provides an accurate, interpretable, and noise-resilient solution for predictive maintenance in industrial environments.


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
