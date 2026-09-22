# SCAS: Structural Change Alert Signal for European Energy Generation Networks

**Author:** Bruno Priantti | **Quantyma Research**
**Field:** Complex Systems, Temporal Networks, Time-Series Analysis, Energy Systems

---

## Abstract

This study presents the Structural Change Alert Signal (SCAS), a network-based framework for identifying periods of potential structural change in interconnected energy generation systems. The framework combines temporal correlation networks, global network metrics, and Matrix Profile anomaly detection to monitor changes in the relationships among energy generation profiles. The analysis uses monthly generation data from ten European countries and five generation categories: Fossil, Nuclear, Hydro, Wind, and Solar. After log-differencing and standardization, 50 time series are organized into overlapping 12-month windows to construct temporal correlation networks. Density, clustering, modularity, and assortativity are computed for each network, and Matrix Profile is applied to identify anomalous temporal patterns. The resulting anomalies are combined into a binary SCAS. Across 102 monthly windows from 2016 to 2024, SCAS alerts coincide with 24 of 29 documented events (82.8%) within a two-month tolerance window. The source code and analysis are publicly available for reproducibility and further research.

---

## Data and Experimental Setup

### 1. Dataset

The analysis uses the Real-E benchmark constructed from the ENTSO-E Transparency Platform.

The study uses the **O20 version** of the dataset and the **Actual Generation per Production Type** subset at the national level (CTY).

The original production types are aggregated into five generation categories:

* Fossil
* Nuclear
* Hydro
* Wind
* Solar

Ten European countries are selected:

* Austria
* Belgium
* Czech Republic
* France
* Germany
* Italy
* Netherlands
* Poland
* Spain
* Switzerland

This produces **50 country-category time series**.

The original 15-minute generation data are aggregated to monthly totals before the temporal network analysis.

---

### 2. Temporal Preprocessing

Each generation series is transformed using the following pipeline:

```text
15-minute generation data
        ↓
Monthly aggregation
        ↓
Country × generation category organization
        ↓
log1p transformation
        ↓
First-order differencing
        ↓
Z-score normalization
        ↓
50 standardized time series
```

For each series:

$$
Y_t = \log(1+X_t)
$$

$$
\Delta Y_t = Y_t-Y_{t-1}
$$

$$
Z_t =
\frac{\Delta Y_t-\mu_{\Delta Y}}
{\sigma_{\Delta Y}}
$$

The resulting standardized series are used to construct the temporal correlation networks.

---

### 3. Temporal Network Construction

The 50 standardized time series are divided into overlapping windows of 12 months with a step size of one month.

Using 113 monthly observations:

$$
W = 113-12+1=102
$$

temporal windows are obtained.

For each window, pairwise Pearson correlations are calculated and converted into a signed weighted network using:

$$
|r_{ij}| \geq 0.5.
$$

Each node represents a country-generation category pair, while edge weights correspond to the original Pearson correlation coefficients.

The resulting sequence contains 102 temporal correlation networks.

---

### 4. Network Metrics

Four global network metrics are calculated for each temporal window:

| Metric        | Description                     |
| ------------- | ------------------------------- |
| Density       | Overall network connectivity    |
| Clustering    | Local connectivity and cohesion |
| Modularity    | Community organization          |
| Assortativity | Degree-based mixing structure   |

These metrics produce four temporal signals describing the evolution of the network structure.

---

### 5. Anomaly Detection

Matrix Profile is independently applied to each network metric.

The configuration used in the experiment is:

```text
Method: Matrix Profile
Subsequence length: 6 months
Anomaly type: Discords
Selected anomalies: Top 5 per metric
Metrics: 4
Total candidate anomalies: 20
```

The highest-distance subsequences are treated as candidate structural anomalies.

---

### 6. Structural Change Alert Signal

The detected anomalies are converted into the **Structural Change Alert Signal (SCAS)**.

SCAS is defined as a binary temporal signal:

$$
S_t =
\begin{cases}
1, & \text{if an anomaly is detected at or near }t,\\
0, & \text{otherwise}.
\end{cases}
$$

When an anomaly is detected, the signal remains active for the subsequent two months.

The SCAS therefore provides a compact temporal representation of periods associated with anomalous network configurations.

---

### 7. Event Validation

The SCAS is evaluated against **29 documented events** between 2016 and 2024.

The events are grouped into six categories:

* Geopolitical
* Market
* Climate
* Infrastructure
* Policy
* Sanitary

An event is considered temporally matched when it occurs within the predefined two-month tolerance window of an SCAS alert.

The resulting event match rate is:

$$
\frac{24}{29}=82.8\%.
$$

A random baseline is additionally used to assess how frequently comparable matches can occur under randomly positioned event dates.

---

## Computational Environment

### Hardware

The analysis was developed and executed in a local research environment.

The computational workflow is based on Python and Jupyter Notebook.

### Software Stack

```text
Python
├── pandas
├── numpy
├── scipy
├── networkx
├── stumpy
├── matplotlib
└── jupyter
```

The complete computational environment and analysis scripts are maintained in the project repository.

---

## Reproducibility

The implementation, data-processing pipeline, network construction, anomaly detection, SCAS generation, and event-validation procedures are available at:

https://github.com/quantyma/quantyma-research/tree/main/quantyma-article-005

The repository is organized to allow the complete analysis pipeline to be reproduced from preprocessing through event validation.



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

