# Customer Churn Analysis & Retention Forecasting Dashboard
🚀 Live Demo: https://customer-churn-analysis-cdputwmhckphydt9duwvib.streamlit.app/
## Project Overview

Customer retention is often more valuable than customer acquisition, but many organizations struggle to identify which users are at risk of leaving and when intervention should occur.

This project combines machine learning, survival analysis, and interactive visualization to identify churn risk, forecast retention trends, and explain the behavioral drivers behind customer attrition.

The final solution is deployed as a Streamlit dashboard that allows users to explore churn patterns, retention forecasts, and feature importance metrics through an interactive interface.

---

## Business Problem

Customer churn directly impacts revenue, customer lifetime value, and growth forecasting.

The goal of this project was to answer three key questions:

1. Which customers are most likely to churn?
2. When is churn most likely to occur?
3. What customer behaviors contribute most to churn risk?

---

## Project Components

### Feature Engineering Pipeline

Developed a dedicated feature engineering workflow to transform raw customer activity data into model-ready features.

Examples include:

* Behavioral engagement metrics
* Usage frequency indicators
* Retention window calculations
* Temporal activity trends
* Aggregated customer-level statistics

---

### Early Churn Prediction System

Built machine learning models designed to identify customers at risk of churning during the earliest stages of their lifecycle.

Objectives:

* Detect high-risk customers early
* Support proactive retention strategies
* Improve intervention timing

---

### Churn Timing Risk Analysis

Implemented survival analysis techniques to estimate the probability of customer churn over time.

Methods include:

* Cox Proportional Hazards Modeling
* Hazard Ratio Analysis
* Survival Curve Estimation

This component helps answer not only whether a customer may churn, but when churn is most likely to occur.

---

### Retention Forecasting

Generated forward-looking retention forecasts to estimate future customer retention trends.

Outputs include:

* Retention curves
* Retention change windows
* Cohort-level retention projections
* Long-term retention estimates

---

### Explainable AI (SHAP)

Integrated SHAP (SHapley Additive exPlanations) to improve model interpretability.

This allows users to:

* Understand feature importance
* Analyze churn drivers
* Explain model predictions
* Identify actionable retention insights

---

## Dashboard Features

The Streamlit dashboard provides:

* Customer churn risk analysis
* Retention forecasting visualizations
* Survival analysis outputs
* Hazard ratio interpretation
* SHAP feature importance exploration
* Interactive charts and filtering

---

## Technical Stack

### Languages & Libraries

* Python
* Pandas
* NumPy
* Scikit-Learn
* Lifelines
* SHAP
* Plotly
* Streamlit

### Deployment Architecture

GitHub is used to store source code and application logic.

Large serialized artifacts, precomputed analytical outputs, and model-generated datasets are stored separately on Hugging Face Hub to avoid GitHub file size limitations.

Architecture:

GitHub (Application Code)
→ Streamlit Dashboard
→ Hugging Face Dataset Storage
→ Precomputed Outputs & Analytical Artifacts

This approach keeps the repository lightweight while allowing the dashboard to load large analytical assets at runtime.

---

## Repository Structure

```text
app.py
APPS_Feature_Engineering.py
Churn_Timing_risk.py
Early Churn Prediction System.py
Retention Forecasting.py
requirements.txt
```

Large data artifacts are hosted externally through Hugging Face Hub and loaded dynamically when the application starts.

---

## Future Improvements

* Automated model retraining pipeline
* Cloud-based deployment
* Real-time churn monitoring
* Experiment tracking
* Additional forecasting models

---

## Author

Lauren Cunningham

Data Science | Machine Learning | Predictive Analytics
