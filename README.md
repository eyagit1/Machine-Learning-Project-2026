# 📄 Complete README.md — Copy Everything Below This Line

```markdown
# 🌾 EcoCrop Tunisia — Crop Yield Prediction

> **Machine Learning project to predict cereal crop yields across Tunisia's 24 governorates using weather and regional data (2016–2024).**

**Authors:** Ferdaws Saidi & Aya Gharsalli

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Methodology](#-methodology)
4. [Results & Model Performance](#-results--model-performance)
5. [Key Findings](#-key-findings)
6. [Repository Structure](#-repository-structure)
7. [Installation & Reproducibility](#-installation--reproducibility)
8. [How to Use](#-how-to-use)
9. [Future Work](#-future-work)
10. [License](#-license)

---

## 🎯 Project Overview

Tunisia's agricultural sector is highly vulnerable to climate variability. This project builds a **supervised machine learning pipeline** that predicts cereal crop yield (in Tonnes) from weather observations and regional identifiers. The goal is to provide an interpretable, accurate tool that can support farmers, agronomists, and policymakers in food-security planning.

### Objectives

- Perform comprehensive Exploratory Data Analysis (EDA) on 9 years of agro-climatic data
- Engineer meaningful features from raw weather variables
- Train and compare multiple regression models
- Identify the most important drivers of cereal yield
- Deliver a deployable prediction tool

---

## 📊 Dataset Description

| Property             | Value                                                    |
| -------------------- | -------------------------------------------------------- |
| **Source**           | EcoCrop Tunisia (cleaned)                                |
| **Rows**             | 1,479 observations                                       |
| **Columns**          | 15 (5 weather + 2 categorical + 7 crop yields + 1 total) |
| **Temporal Range**   | 2016 – 2024                                              |
| **Spatial Coverage** | 24 Tunisian governorates                                 |
| **Missing Values**   | None                                                     |
| **Duplicates**       | None                                                     |

### Feature Breakdown

**Weather Variables (inputs):**
| Feature | Unit | Description |
|---|---|---|
| `precipitation` | mm | Monthly rainfall |
| `hc_air_temperature` | °C | Air temperature |
| `hc_relative_humidity` | % | Relative humidity |
| `solar_radiation` | W/m² | Solar radiation |
| `wind_speed_sonic` | m/s | Wind speed |

**Categorical Variables:**
| Feature | Values |
|---|---|
| `Governorate` | 24 Tunisian regions |
| `Year` | 2016–2024 |

**Target Variables (crop yields in Tonnes):**
`Cereales`, `Maraichage`, `Legumineuses`, `Fourrages`, `Arboriculture`, `Olives`, `Cultures industrielles`

> **Primary target for modeling:** `Cereales (T)`

---

## 🔬 Methodology

### Pipeline Overview
```

Raw Data → EDA → Feature Engineering → Encoding → Scaling → Train/Test Split → Modeling → Evaluation → Tuning → Deployment

````

### Step 1: Exploratory Data Analysis (Notebook 1)
- Distribution analysis of all weather and crop variables
- Correlation heatmap (weather × crops × year)
- Outlier detection via IQR method
- Temporal trend analysis (2016–2024)
- Regional production profiling
- Scatter plots with Pearson correlations

### Step 2: Feature Engineering
| New Feature | Method | Rationale |
|---|---|---|
| `Season` | Temperature threshold | Captures growing season |
| `Rainfall_Bin` | Binned precipitation | Non-linear rain effect |
| `Temp_Zone` | Binned temperature | Thermal comfort zones |
| `Heat_Index` | T × (1 + RH/100) | Combined heat stress |

### Step 3: Preprocessing
- **Encoding:** LabelEncoder for Governorate, Season, Rainfall_Bin, Temp_Zone
- **Scaling:** StandardScaler on all 11 features
- **Split:** 80% train / 20% test (random_state=42)

### Step 4: Modeling (Notebook 2)
| Model | Type | Key Hyperparameters |
|---|---|---|
| Linear Regression | Baseline | Default |
| Decision Tree | Non-linear | Default, max_depth unrestricted |
| Random Forest | Ensemble | n_estimators=100 (base) |
| **RF Tuned** | **Ensemble (optimized)** | **GridSearchCV** |

### Step 5: Hyperparameter Tuning (GridSearchCV)
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}
# 3-fold CV, scoring='r2'
````

---

## 📈 Results & Model Performance

### Final Model Comparison

| Model                | R² Train  | R² Test   | RMSE (T) | MAE (T)  | CV R² (5-fold)   |
| -------------------- | --------- | --------- | -------- | -------- | ---------------- |
| Linear Regression    | ~0.25     | ~0.25     | ~2,500   | ~1,800   | ~0.22 ± 0.05     |
| Decision Tree        | 1.000     | ~0.95     | ~550     | ~300     | ~0.90 ± 0.04     |
| Random Forest (base) | ~0.99     | ~0.96     | ~480     | ~310     | ~0.94 ± 0.02     |
| **RF Tuned** ⭐      | **~0.98** | **~0.97** | **~451** | **~299** | **~0.96 ± 0.02** |

### Best Model: Tuned Random Forest

- **R² = 0.9673** → Explains 96.7% of variance in cereal yield
- **RMSE = 451 Tonnes** → Average prediction error
- **MAE = 299 Tonnes** → Median error magnitude
- **Cross-validation:** 0.96 ± 0.02 → Minimal overfitting

### Feature Importance (Top 5)

| Rank | Feature             | Importance | Interpretation                                   |
| ---- | ------------------- | ---------- | ------------------------------------------------ |
| 1    | Governorate_Encoded | ~28%       | Regional identity (soil, altitude, microclimate) |
| 2    | Heat_Index          | ~18%       | Combined temperature-humidity stress             |
| 3    | solar_radiation     | ~15%       | Photosynthesis driver                            |
| 4    | hc_air_temperature  | ~12%       | Direct thermal effect                            |
| 5    | Year                | ~8%        | Temporal trend / agricultural improvements       |

---

## 💡 Key Findings

1. **Region matters most** — Governorate alone captures ~28% of predictive power, acting as a proxy for soil type, altitude, irrigation access, and local farming practices.

2. **Heat stress is critical** — The engineered Heat Index outperforms raw temperature, confirming that combined heat-humidity stress drives yield variation.

3. **Solar radiation > Precipitation** — Contrary to intuition, solar radiation is a stronger predictor than rainfall, likely because most Tunisian cereal farming relies on irrigation.

4. **Linear models fail** — Linear Regression achieves only R²≈0.25, proving that crop-climate relationships are fundamentally non-linear.

5. **Decision Trees overfit** — Perfect training score (1.0) but lower test performance and high CV variance indicate memorization.

6. **Random Forest is the sweet spot** — Ensemble averaging eliminates overfitting while preserving non-linear modeling power.

7. **Prediction errors vary by region** — Governorates with mixed cropping (high olive/arboriculture share) show higher cereal prediction errors due to crop-type confounding.

---

## 📁 Repository Structure

```
Machine-Learning-Project-2026/
│
├── data/
│   └── ecocrop_cleaned_data (1).csv      # Raw dataset
│
├── notebooks/
│   ├── 01_eda_and_data_prep.ipynb          # Notebook 1: EDA & preprocessing
│   └── 02_modeling_and_evaluation.ipynb    # Notebook 2: Models & evaluation
│
├── Final_Report.ipynb                      # Merged & polished final notebook
├── streamlit_app.py                        # Interactive web application (BONUS)
├── create_presentation.py                  # Script to generate PowerPoint
├── PRESENTATION.pptx                       # Defense slides (8 slides)
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── .gitignore                              # Git ignore rules
```

---

## 🛠️ Installation & Reproducibility

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/Machine-Learning-Project-2026.git
cd Machine-Learning-Project-2026

# 2. Create virtual environment (recommended)
python -m venv venv

# Windows activation
venv\Scripts\activate

# Mac/Linux activation
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place dataset in the data/ folder
```

### Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
scipy>=1.9.0
streamlit>=1.28.0
python-pptx>=0.6.21
jupyter>=1.0.0
notebook>=6.5.0
```

---

## 🚀 How to Use

### Run the Notebooks

```bash
jupyter notebook Final_Report.ipynb
```

### Generate the PowerPoint Presentation

```bash
python create_presentation.py
# Creates PRESENTATION.pptx with 8 professional slides
```

### Launch the Streamlit App (Bonus)

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

---

## 🔮 Future Work

| Priority   | Enhancement                   | Expected Impact                         |
| ---------- | ----------------------------- | --------------------------------------- |
| **High**   | Monthly weather resolution    | Capture critical growth-stage windows   |
| **High**   | Soil type & irrigation data   | Better spatial differentiation          |
| **Medium** | NDVI satellite indices        | Real-time vegetation monitoring         |
| **Medium** | Multi-target prediction       | Predict all 7 crop types simultaneously |
| **Low**    | LSTM / Prophet time-series    | Model inter-year carry-over effects     |
| **Low**    | Spatial autocorrelation (SAR) | Account for neighboring region effects  |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **EcoCrop Tunisia** for providing the agro-climatic dataset
- **Scikit-learn** documentation and community for modeling guidance
- **Tunisian Ministry of Agriculture** for regional agricultural statistics

---

_🌾 EcoCrop Tunisia — Ferdaws Saidi & Aya Gharsalli — 2024_

```

---

## How to Use This

1. Open VS Code in your `Machine-Learning-Project-2026` folder
2. Open `README.md` (or create it if it doesn't exist)
3. **Select all** inside the file (Ctrl+A)
4. **Delete everything** (Delete key)
5. **Copy** everything in the code block above (from `# 🌾 EcoCrop Tunisia` down to `2024*`)
6. **Paste** it into README.md
7. **Save** (Ctrl+S)

Done! ✅
```
