# **`README.md`**

## Quantifying Semantic Shift in Financial NLP: A Robust Evaluation Framework

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.0205-b31b1b.svg)](https://arxiv.org/abs/2510.00205v1)
[![Conference](https://img.shields.io/badge/Conference-ICAIF%20'25-9cf)](https://icaif.acm.org/2025/)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/quantifying_semantic_shift_financial_nlp)
[![Discipline](https://img.shields.io/badge/Discipline-Financial%20NLP-00529B)](https://github.com/chirindaopensource/quantifying_semantic_shift_financial_nlp)
[![Primary Data](https://img.shields.io/badge/Data-Financial%20News%20%7C%20Stock%20Returns-lightgrey)](https://github.com/chirindaopensource/quantifying_semantic_shift_financial_nlp)
[![Core Method](https://img.shields.io/badge/Method-Regime--Based%20Robustness%20Testing-orange)](https://github.com/chirindaopensource/quantifying_semantic_shift_financial_nlp)
[![Key Metrics](https://img.shields.io/badge/Metrics-FCAS%20%7C%20PCS%20%7C%20TSV%20%7C%20NLICS-red)](https://github.com/chirindaopensource/quantifying_semantic_shift_financial_nlp)
[![Models](https://img.shields.io/badge/Models-LSTM%20%7C%20Transformers-blueviolet)](https://github.com/chirindaopensource/quantifying_semantic_shift_financial_nlp)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg?style=flat&logo=OpenAI&logoColor=white)](https://openai.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
--

**Repository:** `https://github.com/chirindaopensource/quantifying_semantic_shift_financial_nlp`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Quantifying Semantic Shift in Financial NLP: Robust Metrics for Market Prediction Stability"** by:

*   Zhongtian Sun
*   Chenghao Xiao
*   Anoushka Harit
*   Jongmin Yu

The project provides a complete, end-to-end computational framework for replicating the paper's novel evaluation suite for financial NLP models. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from rigorous data validation and regime-based partitioning, through multi-architecture model training and feature engineering, to the computation of four novel diagnostic metrics and a comprehensive suite of analytical studies.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callables](#key-callables)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "Quantifying Semantic Shift in Financial NLP: Robust Metrics for Market Prediction Stability." The core of this repository is the iPython Notebook `quantifying_semantic_shift_financial_nlp_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial data validation to the final generation of all analytical tables and figures.

The paper introduces a structured evaluation framework to quantify the robustness of financial NLP models under the stress of macroeconomic regime shifts. It argues that standard metrics like MSE are insufficient and proposes four complementary diagnostic metrics to provide a multi-faceted view of model stability. This codebase operationalizes this advanced evaluation suite, allowing users to:
-   Rigorously validate and cleanse time-series financial news and market data.
-   Systematically partition data into distinct macroeconomic regimes (e.g., Pre-COVID, COVID).
-   Perform chronological train-validation-test splits to prevent lookahead bias.
-   Train multiple model architectures (LSTM, Text Transformer, Feature-Enhanced MLP) on a per-regime basis.
-   Compute the four novel diagnostic metrics: **FCAS**, **PCS**, **TSV**, and **NLICS**.
-   Quantify semantic drift between regimes using **Jensen-Shannon Divergence**.
-   Conduct a full suite of analyses, including case studies, ablation studies, and cross-sector generalization tests.

## Theoretical Background

The implemented methods are grounded in time-series econometrics, natural language processing, and deep learning.

**1. Regime-Based Evaluation:**
The framework's foundation is the acknowledgment that financial markets are non-stationary. The data-generating process changes over time, particularly during major economic events. The methodology explicitly partitions the data into distinct macroeconomic regimes, $R = \{r_1, ..., r_K\}$, and evaluates models within each regime $r_k$. This allows for a precise measurement of performance degradation under structural breaks.

**2. The Four Diagnostic Metrics:**
The paper introduces four metrics to create a "Robustness Profile" beyond simple prediction error:
-   **Financial Causal Attribution Score (FCAS):** Measures if a model's prediction direction aligns with simple causal keywords in the source text.
    $$
    \text{FCAS} = \mathbb{E}[\mathbb{I}(\text{sign}(\text{prediction}) = \text{sign}(\text{causal\_cue}))]
    $$
-   **Patent Cliff Sensitivity (PCS):** Measures the magnitude of change in a model's prediction when the input text is subjected to a controlled semantic perturbation (e.g., "growth" -> "decline").
    $$
    \text{PCS} = \mathbb{E}[|f_\theta(\mathbf{x}) - f_\theta(\tilde{\mathbf{x}})|]
    $$
-   **Temporal Semantic Volatility (TSV):** Measures the drift in the underlying meaning of the text corpus over time, calculated as the average Euclidean distance between embeddings of consecutive news articles.
    $$
    \text{TSV} = \frac{1}{N-1} \sum_{i=1}^{N-1} \|\phi(\mathbf{x}_{i+1}) - \phi(\mathbf{x}_i)\|_2
    $$
-   **NLI-based Logical Consistency Score (NLICS):** Uses a large language model (LLM) to perform Natural Language Inference, assessing whether the model's prediction is a logical entailment of the source news text.
    $$
    \text{NLICS} = \mathbb{E}[\text{EntailmentScore}(\text{text}, \text{Hypothesis}(\text{prediction}))]
    $$

**3. Semantic Drift Quantification:**
The linguistic shift between any two regimes is quantified using the **Jensen-Shannon (J-S) Divergence** between their respective vocabulary probability distributions. This provides a formal measure of how much the language used in financial news has changed.
$$
D_{JS}(P, Q) = \frac{1}{2}D_{KL}(P || M) + \frac{1}{2}D_{KL}(Q || M), \quad M = \frac{1}{2}(P+Q)
$$

## Features

The provided iPython Notebook (`quantifying_semantic_shift_financial_nlp_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Phase Architecture:** The entire pipeline is broken down into 35 distinct, modular tasks, each with its own orchestrator function, covering validation, partitioning, feature engineering, training, inference, and a full suite of analyses.
-   **Configuration-Driven Design:** All experimental parameters are managed in an external `config.yaml` file, allowing for easy customization and replication without code changes.
-   **Multi-Architecture Support:** Complete training and evaluation pipelines for three distinct model types: a baseline LSTM, a fine-tuned Text Transformer (DistilBERT), and a hybrid Feature-Enhanced MLP.
-   **Idempotent and Resumable Pipelines:** Computationally expensive steps, such as model training and LLM-based evaluations, are designed to be idempotent (resumable), saving checkpoints and caching results to prevent loss of progress and redundant computation.
-   **Production-Grade Metric Implementation:** Includes a highly performant, asynchronous, and cached implementation for the NLICS metric and a full-pipeline replication for the computationally intensive PCS metric.
-   **Comprehensive Analysis Suite:** Implements all analyses from the paper, including J-S divergence, t-SNE visualization, stock-specific case studies, control experiments, and a full N x N cross-sector generalization matrix.
-   **Automated Reporting:** Programmatic generation of all key tables and figures from the paper, as well as a final, synthesized analytical report.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation & Cleansing (Tasks 1-3):** Ingests and rigorously validates the raw data and `config.yaml`, performs a deep data quality audit, and standardizes all data.
2.  **Data Partitioning (Tasks 4-6):** Partitions the data by macroeconomic regime and performs chronological train/val/test splits.
3.  **Feature Engineering (Tasks 7-9):** Generates TF-IDF, sentence embedding, and combined feature sets.
4.  **Model Training (Tasks 10-15):** Orchestrates the training of all 12 model-regime pairs with early stopping and checkpointing.
5.  **Inference & Evaluation (Tasks 16-24):** Generates predictions on all test sets and computes the full suite of five performance and diagnostic metrics.
6.  **Analysis & Ablation (Tasks 25-35):** Executes all higher-level analyses, including semantic drift calculation, visualizations, case studies, and ablation studies.

## Core Components (Notebook Structure)

The `quantifying_semantic_shift_financial_nlp_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callables

The project is designed around a single, top-level user-facing interface function:

-   **`execute_quantifying_semantic_shift_study`:** This master orchestrator function runs the entire automated research pipeline from end-to-end. It handles all data processing, model training, and analysis. It also generates the necessary files for the optional, human-in-the-loop entailment model comparison. A single call to this function reproduces the entire computational portion of the project.

## Prerequisites

-   Python 3.9+
-   An OpenAI API key set as an environment variable (`OPENAI_API_KEY`) for the NLICS metric.
-   Core dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `pyyaml`, `torch`, `transformers`, `sentence-transformers`, `openai`, `matplotlib`, `seaborn`, `tqdm`, `ipython`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/quantifying_semantic_shift_financial_nlp.git
    cd quantifying_semantic_shift_financial_nlp
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set Environment Variable:**
    ```sh
    export OPENAI_API_KEY="your_api_key_here"
    ```

## Input Data Structure

The pipeline requires a single `pandas.DataFrame` and a `config.yaml` file. The script includes a helper function to generate a synthetic, structurally correct DataFrame for testing purposes. The required schema is:
-   **Index:** A `pandas.MultiIndex` with three levels:
    -   `date` (`DatetimeIndex`): The trading date.
    -   `ticker` (`object`): The stock ticker.
    -   `sector` (`object`): The GICS sector.
-   **Columns:**
    -   `Open`, `High`, `Low`, `Close`, `Adj Close` (`float64`): Standard market data.
    -   `Volume` (`int64`): Daily trading volume.
    -   `aggregated_text` (`object`/`str`): Concatenated daily news text. An empty string is a valid value.
    -   `target_return` (`float64`): The forward-looking, next-day adjusted close return.

## Usage

The `quantifying_semantic_shift_financial_nlp_draft.ipynb` notebook provides a complete, step-by-step guide. The primary workflow is to call the top-level orchestrator from a `main.py` script or the final cell of the notebook:

```python
# main.py
from pathlib import Path
import pandas as pd
import yaml

# Assuming all pipeline functions are in `pipeline.py`
from pipeline import execute_quantifying_semantic_shift_study

# Load configuration
with open("config.yaml", 'r') as f:
    study_config = yaml.safe_load(f)

# Load data (or generate synthetic data)
raw_df = pd.read_pickle("data/financial_data.pkl")

# Run the entire study
final_artifacts = execute_quantifying_semantic_shift_study(
    raw_df=raw_df,
    study_config=study_config
)
```

## Output Structure

The `execute_quantifying_semantic_shift_study` function creates a `results/` directory and returns a dictionary of artifact paths:

```
{
    'data_splits': Path('results/data_splits.pkl'),
    'training_results': Path('results/training_results.pkl'),
    'enriched_predictions': Path('results/enriched_predictions.pkl'),
    'robustness_profile': Path('results/robustness_profile.csv'),
    'js_divergence_matrix': Path('results/js_divergence_matrix.csv'),
    'nli_benchmark_for_annotation': Path('results/nli_benchmark_for_annotation.csv'),
    ...
}
```

## Project Structure

```
quantifying_semantic_shift_financial_nlp/
│
├── quantifying_semantic_shift_financial_nlp_draft.ipynb # Main implementation notebook
├── config.yaml                                          # Master configuration file
├── requirements.txt                                     # Python package dependencies
├── LICENSE                                              # MIT license file
└── README.md                                            # This documentation file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can easily modify all experimental parameters, including regime dates, model architectures, feature engineering settings, and LLM prompts, without altering the core Python code.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Additional Model Architectures:** Integrating other models like FinBERT or more advanced transformer architectures.
-   **Alternative Diagnostic Metrics:** Implementing other measures of model robustness, such as influence functions or prediction confidence calibration.
-   **Automated Retraining Triggers:** Building a system that uses the computed drift metrics (like TSV or J-S Divergence) to automatically trigger model retraining when a significant regime shift is detected.
-   **Dynamic Feature Selection:** Exploring methods for dynamically adjusting feature importance based on the detected market regime.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@inproceedings{sun2025quantifying,
  author    = {Sun, Zhongtian and Xiao, Chenghao and Harit, Anoushka and Yu, Jongmin},
  title     = {Quantifying Semantic Shift in Financial NLP: Robust Metrics for Market Prediction Stability},
  booktitle = {Proceedings of the 6th ACM International Conference on AI in Finance},
  series    = {ICAIF '25},
  year      = {2025},
  publisher = {ACM}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Professional-Grade Implementation of the "Quantifying Semantic Shift" Framework.
GitHub repository: https://github.com/chirindaopensource/quantifying_semantic_shift_financial_nlp
```

## Acknowledgments

-   Credit to **Zhongtian Sun, Chenghao Xiao, Anoushka Harit, and Jongmin Yu** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, SciPy, Scikit-learn, PyTorch, HuggingFace, and Jupyter**, whose work makes complex computational analysis accessible and robust.

--

*This README was generated based on the structure and content of `quantifying_semantic_shift_financial_nlp_draft.ipynb` and follows best practices for research software documentation.*
