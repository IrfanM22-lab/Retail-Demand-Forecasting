# Retail Demand Forecasting: Machine Learning & LSTM Model Comparison

## Project Overview

This project develops an end-to-end **retail demand forecasting pipeline** using historical store-product sales data and a combination of operational, commercial, calendar and environmental variables.

The objective is to determine how accurately future daily demand can be predicted and to compare the performance of four different modelling approaches:

- Linear Regression
- Random Forest
- XGBoost
- Long Short-Term Memory (LSTM)

The project places particular emphasis on **time-series-aware modelling**, including chronological train/test splitting, historical lag features, rolling statistics and leakage prevention.

Rather than assuming that the most complex model will perform best, the project evaluates each approach on a held-out future period and uses the results to select the strongest model for this dataset.

## Business Problem

Accurate demand forecasting is an important component of retail decision-making.

Poor demand forecasts can contribute to:

- Overstocking
- Stock shortages
- Inefficient replenishment
- Poor inventory allocation
- Missed sales opportunities
- Inefficient operational planning

The central question addressed by this project is:

**How accurately can future daily demand be predicted using historical demand patterns together with operational, commercial and contextual information, and which modelling approach provides the strongest forecasting performance?**

The resulting forecasts could potentially support inventory planning, replenishment decisions, promotion planning and short-term operational planning.

## Dataset

The dataset contains **76,000 store-product-day observations** covering:

- **5 stores**
- **20 products**
- **760 daily dates**
- **100 Store-Product time series**
- Date range: **2022-01-01 to 2024-01-30**

Each observation represents one combination of:

**Store × Product × Date**

The target variable is:

**Demand**

The dataset also contains operational, commercial and contextual variables relating to inventory, sales, orders, pricing, promotions, competitors, weather, seasonality and epidemic conditions.

### Dataset Structure

The expected panel structure is:

5 stores × 20 products × 760 dates = 76,000 observations

The notebook validates this structure before beginning the modelling process.

# Project Workflow

The project follows an end-to-end forecasting workflow:

Load Dataset  
↓  
Data Inspection  
↓  
Data Quality Validation  
↓  
Exploratory Data Analysis  
↓  
Feature Engineering  
↓  
Leakage Prevention  
↓  
Chronological Train/Test Split  
↓  
Feature Scaling & Encoding  
↓  
Model Training  
↓  
Model Evaluation  
↓  
Error Analysis  
↓  
Feature Importance  
↓  
Final Model Selection  
↓  
Business Recommendations

# 1\. Data Loading & Inspection

The notebook begins by loading the dataset and parsing the Date column as a datetime variable.

Initial inspection covers:

- Dataset dimensions
- Data types
- Sample observations
- Descriptive statistics
- Date coverage
- Store and product structure

This establishes the structure of the forecasting problem before any modelling decisions are made.

# 2\. Data Quality Validation

Several checks are performed to ensure that the dataset is suitable for time-series feature engineering.

The notebook verifies:

- Missing values
- Duplicate records
- Number of Store-Product combinations
- Observations per Store-Product series
- Chronological ordering
- Completeness of the panel

### Results

The dataset contains:

- **0 missing values**
- **0 duplicate rows**
- **100 Store-Product combinations**
- **760 observations per Store-Product series**

This provides a complete and consistent panel for constructing historical demand features.

# 3\. Exploratory Data Analysis

Exploratory Data Analysis is used to understand the underlying demand patterns before modelling.

The analysis investigates:

- Overall demand trends
- Day-of-week effects
- Monthly variation
- Weather conditions
- Promotions
- Epidemic periods
- Seasonality
- Store-level differences
- Product-category differences
- Correlations between numerical variables

The purpose of this stage is to identify patterns that may provide predictive value and inform the subsequent feature-engineering strategy.

## Demand Over Time

The overall daily demand series is visualised to identify:

- Long-term movement
- Recurring patterns
- Peaks and troughs
- Potential seasonal behaviour
- Unusual periods

Because the dataset is a time series, understanding how demand changes through time is essential before selecting a modelling strategy.

## Calendar Effects

Demand is compared across:

- Days of the week
- Months

These comparisons help determine whether recurring calendar patterns exist.

Calendar features are subsequently incorporated into the modelling dataset so that models have explicit access to this temporal structure.

## Contextual Demand Drivers

Demand distributions are examined across:

- Weather conditions
- Promotions
- Epidemic status
- Seasonality

These variables represent potentially useful contextual information beyond historical demand.

However, the notebook treats these relationships as **associations rather than causal relationships**.

## Store & Product Differences

Demand distributions are also compared across stores and product categories.

This is important because different stores and products may naturally operate at different demand levels.

The analysis therefore supports treating the dataset as a **panel of multiple Store-Product time series**, rather than assuming that every series follows an identical demand pattern.

# 4\. Feature Engineering

Feature engineering transforms the raw dataset into variables that better represent the information available when generating a future demand forecast.

The engineered features fall into several categories.

### Calendar Features

Examples include:

- Day of week
- Month
- Quarter
- Weekend indicator

### Historical Features

Historical lag and rolling features are generated within each Store-Product series.

Examples include:

- 1-day lag
- 7-day lag
- 7-day rolling mean
- 14-day rolling mean

Historical information is also created for operational variables such as:

- Demand
- Units Sold
- Inventory Level

These features allow the models to use historical operational information without directly using the target day's realised outcome.

# Forecasting Leakage Prevention

One of the key methodological considerations in this project is **data leakage**.

The dataset contains:

- Units Sold
- Inventory Level

These variables can be highly informative about demand. However, their **same-day realised values** would not necessarily be available when generating a forecast for that day.

Using them directly could therefore create an unrealistic forecasting scenario.

For this reason, their historical versions are used through lagged and rolling features instead.

This follows the core forecasting principle:

**A forecasting model should only use information that would genuinely be available when the forecast is generated.**

# 5\. Train/Test Split

A standard random train/test split is inappropriate for this forecasting problem because it could allow observations from the future to appear in the training data.

Instead, the project uses a **chronological split**.

The most recent approximately 20% of the observations are reserved as the test period.

### Split Results

| Dataset  | Observations |
| -------- | ------------ |
| Training | 59,700       |
| Testing  | 14,900       |
| Features | 58           |

The test period covers:

**2023-09-04 → 2024-01-30**

This allows the models to be evaluated on a future period that was not used during training.

# 6\. Machine Learning Models

Four different modelling approaches are compared.

| Model             | Purpose                            |
| ----------------- | ---------------------------------- |
| Linear Regression | Interpretable baseline             |
| Random Forest     | Non-linear ensemble model          |
| XGBoost           | Gradient-boosted tree model        |
| LSTM              | Sequence-based deep learning model |

The comparison is designed to determine whether increased model complexity actually produces better forecasting performance.

## Linear Regression

Linear Regression provides an interpretable baseline.

It assumes that the relationship between the engineered predictors and demand can be represented through a linear combination of the input variables.

Its purpose is not necessarily to provide the best forecast, but to establish a benchmark against which more complex models can be evaluated.

## Random Forest

Random Forest is used to capture non-linear relationships and interactions between predictors.

The model combines multiple decision trees to produce a more robust ensemble prediction.

This allows the project to test whether non-linear relationships provide meaningful improvements over the Linear Regression baseline.

## XGBoost

XGBoost is a gradient-boosted decision-tree algorithm designed for structured/tabular data.

It builds trees sequentially, with later trees focusing on reducing the errors produced by earlier trees.

This makes XGBoost particularly suitable for the engineered tabular representation used in this project, which contains:

- Historical demand features
- Rolling statistics
- Calendar features
- Store/product information
- Operational variables
- Commercial variables
- Contextual variables

## LSTM

A Long Short-Term Memory network is included as the deep-learning approach.

Unlike the classical tabular models, the LSTM processes observations as sequences.

The model uses a:

**14-day historical sequence → next-day demand**

forecasting setup.

The architecture includes:

- LSTM layer — 64 units
- Dropout
- LSTM layer — 32 units
- Dropout
- Dense layer — 16 units
- Single continuous output

The LSTM receives historical and contextual features rather than relying exclusively on raw demand values.

### LSTM Sequence Structure

The sequence preparation produces:

**73,200 sequence examples**

with each example representing:

14 historical days × 20 features  
↓  
Next-day Demand

The resulting evaluation set contains **14,900 test sequences**, aligned with the same future forecasting period used for the tabular model comparison.

# 7\. Model Evaluation

The models are evaluated using four regression metrics.

### Mean Absolute Error — MAE

Measures the average absolute difference between predicted and actual demand.

**Lower is better.**

MAE is particularly intuitive because it represents the average magnitude of the forecasting error in demand units.

### Root Mean Squared Error — RMSE

RMSE places greater emphasis on larger forecasting errors.

**Lower is better.**

This makes it useful when large forecasting mistakes are particularly important from a business perspective.

### Mean Absolute Percentage Error — MAPE

MAPE expresses forecasting error as a percentage of actual demand.

**Lower is better.**

MAPE should be interpreted carefully because percentage-based errors can become unstable when actual values are very small.

### R² — Coefficient of Determination

R² measures the proportion of variation in the target explained by the model.

**Higher is better.**

# Model Performance

The final test-set results are:

| Model             | MAE       | RMSE      | MAPE       | R²        |
| ----------------- | --------- | --------- | ---------- | --------- |
| Linear Regression | 25.34     | 32.68     | 36.79%     | 0.453     |
| Random Forest     | 22.48     | 29.79     | 32.84%     | 0.545     |
| **XGBoost**       | **17.18** | **23.35** | **25.16%** | **0.721** |
| LSTM              | 29.82     | 38.35     | 44.36%     | 0.247     |

# Best Performing Model: XGBoost

**XGBoost achieved the strongest performance across all four evaluation metrics.**

It produced:

- **MAE:** 17.18
- **RMSE:** 23.35
- **MAPE:** 25.16%
- **R²:** 0.721

This means that, under the project's evaluation setup, XGBoost provides the most accurate predictions among the four tested approaches.

The R² value of **0.721** indicates that the model explains approximately **72.1% of the variation in demand** in the held-out test set.

# Why Did XGBoost Perform Best?

The results demonstrate an important machine-learning principle:

**Greater model complexity does not automatically produce better predictions.**

XGBoost benefits from the structured tabular representation created during feature engineering.

The dataset provides the model with explicit historical signals such as:

- Recent demand
- Weekly demand history
- Rolling demand averages
- Historical operational information
- Store/product information
- Calendar variables
- Commercial conditions
- Contextual variables

Much of the temporal information that an LSTM would normally need to learn from sequences has therefore already been represented explicitly as engineered features.

The result is a strong tabular representation that XGBoost can exploit effectively.

# Why Did the LSTM Underperform?

The LSTM achieved:

- **MAE:** 29.82
- **RMSE:** 38.35
- **MAPE:** 44.36%
- **R²:** 0.247

This does **not** mean that LSTMs are inherently unsuitable for demand forecasting.

Several factors may explain the result:

### 1\. Strong engineered features

The tabular models receive explicit lag and rolling features, reducing the amount of temporal structure they need to discover themselves.

### 2\. Rich tabular representation

The classical models receive the encoded feature matrix, including store/product information and other engineered variables.

### 3\. Number of observations per individual series

Although the dataset contains 76,000 observations overall, these observations are distributed across 100 Store-Product series.

### 4\. Different feature representations

The LSTM and tabular models do not receive exactly the same representation.

The LSTM receives a curated 20-feature sequential input, while the tabular models use the broader encoded feature matrix.

Therefore, the experiment should not be interpreted as a perfectly controlled comparison of model architectures alone.

# Forecast Diagnostics

Model performance is also assessed visually.

## Actual vs Predicted Demand

The actual-vs-predicted plots compare observed demand against model predictions.

The diagonal reference represents perfect predictions.

A stronger model should generally produce predictions that remain closer to this reference line across the demand range.

## Residual Analysis

Residuals are calculated as:

Residual = Actual Demand − Predicted Demand

Therefore:

- Positive residual → underprediction
- Negative residual → overprediction
- Residual near zero → accurate prediction

Residual distributions help identify whether models exhibit systematic errors that may not be obvious from aggregate metrics alone.

# XGBoost Feature Importance

Feature importance is used to investigate which predictors XGBoost relies on most heavily.

The analysis focuses on the top 15 features.

Historical demand and rolling variables can provide particularly valuable information because recent demand behaviour often contains strong signals about near-term demand.

However, feature importance should **not** be interpreted as causal evidence.

A highly important feature means that the fitted model relies heavily on that feature; it does not prove that changing that feature will independently cause demand to change.

# Business Applications

A demand forecasting solution of this type could support several retail planning activities.

### Inventory Planning

Forecasted demand can help businesses estimate future product requirements and reduce the risk of excess inventory or shortages.

### Replenishment

Short-term forecasts can support decisions about when and how much inventory should be replenished.

### Promotion Planning

Forecasts can help businesses understand expected demand around promotional periods.

### Store-Level Planning

Because the dataset contains multiple stores and products, forecasts can potentially support more granular store-product planning.

### Operational Planning

Improved demand visibility can support staffing, ordering and other short-term operational decisions.

The model should ultimately be viewed as **decision support rather than an autonomous decision-making system**.

# Limitations

Several limitations should be considered when interpreting the results.

### Same-Day Information Assumptions

Same-day Units Sold and Inventory Level are excluded from the direct predictor set because they may not be available at forecast generation time.

Other contextual variables are retained under the assumptions of the dataset.

In a production environment, realised weather observations should ideally be replaced with weather forecasts available when the prediction is generated.

### LSTM Comparison

The LSTM and tabular models use different feature representations.

A future version could investigate learned Store/Product embeddings and provide a more comparable feature representation.

### Statistical Forecasting Baselines

The project does not currently include traditional statistical forecasting methods such as SARIMA or ETS.

Adding these models would provide an additional benchmark.

### MAPE

MAPE can be sensitive to low actual demand values.

For this reason, MAE and RMSE should remain important when evaluating the practical forecasting performance.

### External Variables

Additional variables such as holidays, local events, marketing expenditure and detailed supply-chain information could potentially improve future forecasts.

### Hyperparameter Tuning

The models use the configurations implemented in the notebook rather than an extensive time-series hyperparameter optimisation process.

Future work could use rolling or expanding-window validation for model tuning.

# Future Improvements

Several improvements could make this forecasting pipeline more suitable for a production environment.

1. **Time-aware hyperparameter tuning** for XGBoost.
2. Add **SARIMA/ETS** as traditional forecasting benchmarks.
3. Experiment with **Store/Product embeddings** for the LSTM.
4. Incorporate holidays and local events.
5. Use forecasted rather than realised weather information.
6. Evaluate multiple forecasting horizons, such as:
   - 1 day
   - 7 days
   - 14 days
7. Evaluate model performance separately by store, product and category.
8. Add prediction intervals to communicate forecast uncertainty.
9. Develop a business dashboard showing actual demand, forecast demand and forecast error.
10. Investigate model retraining strategies for changing demand patterns.

# Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Visualisation

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- Random Forest
- Linear Regression
- XGBoost

### Deep Learning

- TensorFlow
- Keras
- LSTM

### Environment

- Google Colab
- Jupyter Notebook

# Project Structure

demand-forecasting-ml/  
│  
├── Demand_Forecasting_Portfolio_Ready_v2.ipynb  
├── demand_forecasting.csv  
└── README.md

# Notebook Structure

The notebook is organised into the following major sections:

1\. Setup & Imports  
2\. Load the Dataset  
3\. Display & Inspect the Dataset  
4\. Data Quality & Time-Series Structure  
5\. Exploratory Data Analysis  
6\. Feature Engineering & Data Preparation  
7\. Modelling  
├── Linear Regression  
├── Random Forest  
├── XGBoost  
└── LSTM  
8\. Model Comparison & Evaluation  
9\. Final Findings & Business Recommendations

Each section includes explanations of the methodology, interpretation of outputs and conclusions so that the notebook can be understood without relying solely on the code.

# Skills Demonstrated

This project demonstrates practical experience in:

- Time-series forecasting
- Regression modelling
- Exploratory data analysis
- Feature engineering
- Lag features
- Rolling statistics
- Categorical encoding
- Feature scaling
- Time-aware train/test splitting
- Data leakage prevention
- Classical machine learning
- Gradient boosting
- Deep learning
- LSTM sequence modelling
- Model evaluation
- Residual analysis
- Feature importance
- Business interpretation
- Communicating analytical findings

# Key Takeaway

The primary finding from this project is that **XGBoost provided the strongest forecasting performance on the held-out future test period**, outperforming Linear Regression, Random Forest and the LSTM model across MAE, RMSE, MAPE and R².

More importantly, the project demonstrates that effective forecasting is not simply about selecting the most sophisticated algorithm.

The combination of:

**data validation + leakage prevention + time-aware splitting + meaningful feature engineering + model comparison + error analysis + business interpretation**

is what produces a robust forecasting workflow.

This project therefore demonstrates both the **technical modelling skills** and the **analytical reasoning** required to turn a forecasting problem into an actionable machine-learning solution.

------------------------------------------------------------------------

## Author

**Irfan Moosa**

BSc in Information Technology --- Computer Science & Business Management

This project forms part of a data analytics, machine learning and
cybersecurity portfolio focused on applying technical skills to
practical business and technology problems.
