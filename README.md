# Real Estate Investment Advisor

## Project Overview
The **Real Estate Investment Advisor** is a machine learning project designed to help screen residential property listings based on their relative price positioning within a City and Property Type benchmark.

The project uses machine learning classification to identify properties priced below the benchmark median and explores regression techniques for property price prediction.

**Important:** The classification output indicates relative affordability against the benchmark. It does not establish profitability, investment quality, or future returns.

## Objectives
- Perform Exploratory Data Analysis (EDA) on Indian housing data.
- Identify patterns and relationships in property prices.
- Build a machine learning classification model for property screening.
- Evaluate regression models for property price prediction.
- Track machine learning experiments using MLflow.
- Deploy the classification model through Streamlit.

## Dataset
- **Dataset:** `india_housing_prices.csv`
- **Number of records:** 250,000
- **Target for classification:** `Good_Investment`
- **Target for regression:** `Price_in_Lakhs`

The classification target was created using the median price per square foot for each City and Property Type benchmark.

- `1`: Property price per square foot is below the benchmark median.
- `0`: Property price per square foot is at or above the benchmark median.

The benchmark was derived from training data only to avoid test-label leakage.

## Tech Stack
- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Experiment Tracking:** MLflow
- **Deployment:** Streamlit
- **Environment:** Google Colab, Jupyter Notebook

## Project Workflow
1. Data loading and preprocessing
2. Exploratory Data Analysis (20 analysis questions)
3. Feature engineering and target creation
4. Train-test split
5. Classification model development
6. Model evaluation and feature importance analysis
7. Regression model development and evaluation
8. MLflow experiment tracking
9. Streamlit deployment of the classification model

## Classification Model

**Algorithm:** Random Forest Classifier

**Model Configuration**
- Number of estimators: 100
- Maximum depth: 15
- Minimum samples per leaf: 50
- Random state: 42

**Preprocessing**
- Numerical features: StandardScaler
- Categorical features: OneHotEncoder with `handle_unknown="ignore"`
- Model: RandomForestClassifier

**Dataset Split**
- Training: 200,000 records
- Testing: 50,000 records

### Classification Results

| Metric | Score |
|---|---:|
| Accuracy | 71.27% |
| Precision | 70.39% |
| Recall | 73.30% |
| F1-Score | 71.82% |

The classifier performed better than the dummy baseline, demonstrating useful predictive signal for the defined screening target.

### Feature Importance
The leading features based on the model's feature importance were:

| Feature | Importance |
|---|---:|
| Size_in_SqFt | 49.90% |
| Locality | 21.96% |
| Amenities | 13.78% |

These three features account for approximately 85.64% of the model's reported feature importance.

Permutation importance using F1-score decrease also highlighted `Size_in_SqFt` as the dominant feature.

## Regression Model

**Algorithm:** Random Forest Regressor

**Target:** `Price_in_Lakhs`

### Regression Results

| Metric | Model Result |
|---|---:|
| MAE | 122.3245 Lakhs |
| RMSE | 141.2112 Lakhs |
| R² | -0.0003 |

The regression model performed approximately at the mean-prediction baseline and was marginally worse.

**Conclusion:** The regression model did not demonstrate meaningful property-specific predictive performance and was not deployed.

## MLflow Experiment Tracking

MLflow was used to track model experiments under the experiment:

`Real_Estate_Investment_Advisor`

Experiments recorded:

- Random Forest Classification
- Random Forest Regression

The classification experiment includes model parameters, evaluation metrics, and the logged model.

The regression experiment includes evaluation metrics, baseline comparisons, and model evaluation notes.

## Deployment
The classification model was deployed using **Streamlit**.

The application uses the trained classification pipeline to provide property screening predictions.

The regression model was not included in the deployed application because of its poor predictive performance.

## Limitations
- The classification target represents relative price positioning, not actual investment returns.
- Property prices and market conditions can change over time.
- Model performance may vary across cities and localities.
- Feature importance does not imply causation.
- The regression model did not outperform the simple mean baseline.
- Predictions should not replace independent property valuation, due diligence, or professional financial advice.

## Conclusion
This project demonstrates an end-to-end machine learning workflow, from data exploration and feature engineering to model evaluation, experiment tracking, and deployment.

The classification model provides a preliminary property screening approach, while the regression evaluation highlights the importance of comparing complex models against simple baselines.

## Author
**Vedanshi Chaudhary**
