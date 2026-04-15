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

# My Colab Notebooks

## Notebook 1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eya_gharsalli/Machine-Learning-Project-2026/blob/main/NOTEBOOK1.ipynb)

## Notebook 2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eya_gharsalli/Machine-Learning-Project-2026/blob/main/NOTEBOOK2.ipynb)
**Target Variables (crop yields in Tonnes):**
```
