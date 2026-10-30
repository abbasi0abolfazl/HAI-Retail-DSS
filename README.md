# HAI-Retail DSS (Hybrid Human-AI Retail Decision Support System)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-41CD52.svg?logo=qt&logoColor=white)](https://riverbankcomputing.com/software/pyqt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[🇮🇷 مستندات فارسی (Persian Documentation)](README.fa.md) • [Report Issues](https://github.com/abbasi0abolfazl/HAI-Retail-DSS/issues)

An interactive, simulation-driven Decision Support System (DSS) exploring **Hybrid Human-AI Interaction (HAI)**, Explainable AI (XAI), Fuzzy Risk Management, and Algorithmic Fairness in retail pricing and inventory allocation.

---

## 📌 Conceptual Framework

In high-stakes retail environments, fully autonomous AI systems often fail due to corner-case inventory risks, unmodeled externalities (e.g., weather anomalies), or margin disparities. Conversely, pure human decision-making suffers from cognitive bias and slow response times. 

**HAI-Retail DSS** models a collaborative human-machine equilibrium:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Dynamic Retail State                             │
│     POS Sales Volatility  •  Inventory Level  •  External Weather Impact    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AI Decision Engine                               │
│           MDP Formulation:  V(s) = R(s) + γ · Σ P(s'|s,a) V(s')             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────┐                             ┌───────────────────────┐
│   Fuzzy Risk Module   │                             │   Fairness & XAI      │
│  Downscales extreme   │                             │  Penalizes margin     │
│  actions on low stock │                             │  disparities (L_bias) │
└───────────┬───────────┘                             └───────────┬───────────┘
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Human-in-the-Loop (HITL) Gateway                        │
│          • Trust Calibration: Base Reward scaled by Operator Trust          │
│          • Human Override: Expert intervention with switching penalty       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Final Action & KPI                              │
│       Adjusted Price Action (%)  •  Net Regularized Reward  •  Risk Status  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Formulations

1. **Reinforcement / Value Evaluation:**
   $$V(s) = R(s) + \gamma \sum_{s'} P(s' \mid s, a) V(s')$$

2. **Fairness & Margin Bias Regularization:**
   $$L_{\text{bias}} = \beta \cdot \max(0, \, |m_i - m_j| - \tau)$$
   Where $m_i, m_j$ denote profit margins across retail categories, $\tau$ represents the tolerance threshold, and $\beta$ is the penalty weight.

3. **Adjusted System Reward:**
   $$R_{\text{adj}} = (R_{\text{base}} \cdot \text{Trust}) + 5 \cdot \text{Adapt} - L_{\text{bias}} - \mathbb{I}_{\text{human}} \cdot \text{Penalty}$$

---

## ✨ Features

- **Interactive PyQt5 Desktop Interface:**
  - Modern, responsive styling with clean form controls (`QDoubleSpinBox`, ablation switches).
  - Dual-axis interactive Matplotlib chart comparing **AI Recommendation** vs. **Final Decision** vs. **Net Reward**.
- **Real-Time Ablation Study Toggles:**
  - Toggle **Fuzzy Risk Control** (on/off)
  - Toggle **Explainable AI (XAI)** rationale generation
  - Toggle **Bias Mitigation** penalty enforcement
  - Toggle **Human-in-the-Loop** override mechanism
- **Telemetry & Live Math Display:**
  - Formulas automatically update their parameter values in real time as sliders are adjusted.
  - Live summary metrics: Average Reward, Accumulated Bias Penalty, Human Intervention Ratio, and Active Module Count.

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.10 or higher
- Qt5 libraries

### 1. Clone & Setup Environment

```bash
git clone https://github.com/abbasi0abolfazl/HAI-Retail-DSS.git
cd HAI-Retail-DSS

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application

```bash
python main.py
```

---

## 📂 Repository Structure

```
HAI-Retail-DSS/
├── main.py             # Complete PyQt5 UI, Matplotlib canvas, and simulation engine
├── pyproject.toml      # Project metadata and dependencies
├── requirements.txt    # Frozen dependencies
├── LICENSE             # MIT License
└── README.md           # Documentation
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
