# Patient Dropout Prediction Model

This project contains a Snowflake notebook that builds an **XGBoost classifier** using Python to predict patient dropout from clinical trials.

## Overview

The model uses two simple features to predict whether a patient will dropout from a clinical trial:
- **Age**: Patient age (18-99 years)
- **Gender**: Patient gender (MALE/FEMALE)

**Tech Stack:**
- Python with Snowpark for data access
- XGBoost for gradient boosting classification
- scikit-learn for preprocessing and metrics
- pandas for data manipulation
- matplotlib/seaborn for visualization

## Files

- `patient_dropout_prediction.ipynb`: Snowflake notebook with complete ML pipeline

## Data Source

The notebook uses existing training data from:
- **Database**: INFORMATICS_SANDBOX
- **Schema**: ML_TEST
- **Table**: DOR_ANALYSIS_FF

This table should contain the following columns:
- `age`: Patient age (numeric)
- `gender`: Patient gender (MALE/FEMALE)
- `patient_dropout`: Dropout indicator (1 = dropout, 0 = completed)

## Notebook Structure

The notebook is organized into the following sections:

1. **Import Libraries and Setup**: Import required Python libraries and establish Snowpark session
2. **Load Training Data**: Load data from DOR_ANALYSIS_FF using Snowpark
3. **Exploratory Data Analysis**: Analyze data distribution with visualizations (by age group, gender)
4. **Data Preprocessing**: Encode categorical variables (Gender → binary)
5. **Train/Test Split**: Split data 80/20 with stratification
6. **Train XGBoost Model**: Train XGBoost classifier with feature importance
7. **Make Predictions**: Generate predictions on both train and test sets
8. **Model Evaluation - Test Set**: Calculate accuracy, precision, recall, F1, AUC, confusion matrix
9. **ROC Curve and AUC**: Visualize model performance across thresholds
10. **Predict on New Patients**: Score new patients with risk categorization
11. **Summary and Next Steps**: Provides comprehensive improvement suggestions
12. **Model Persistence**: Optional model saving with joblib

## How to Use

1. Open the notebook in Snowflake (Snowsight UI or Snowflake Notebooks)
2. Ensure you have the necessary permissions to:
   - Create databases and schemas
   - Create tables and views
   - Use Snowflake ML features (SNOWFLAKE.ML.CLASSIFICATION)
3. Run each cell sequentially from top to bottom
4. Review the results and predictions

## Model Details

- **Algorithm**: XGBoost Classifier (Python implementation)
- **Data Source**: INFORMATICS_SANDBOX.ML_TEST.DOR_ANALYSIS_FF  
- **Features**: Age (numeric), Gender (categorical - encoded as binary)
- **Training Set**: 80% stratified split
- **Test Set**: 20% stratified split for validation
- **Evaluation Metrics**: 
  - Accuracy, Precision, Recall, F1 Score
  - ROC-AUC score and curve
  - Confusion Matrix (with heatmap visualization)
  - Classification Report
  - Feature Importance

## Making Predictions

The model can predict dropout probability for new patients. The notebook includes a risk categorization:
- **High Risk**: Dropout probability ≥ 0.7
- **Medium Risk**: Dropout probability 0.4-0.7
- **Low Risk**: Dropout probability < 0.4

## Requirements

- Snowflake account with Snowflake Notebooks support
- Python packages (typically pre-installed in Snowflake Notebooks):
  - `snowflake-snowpark-python`
  - `xgboost`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
- Access to INFORMATICS_SANDBOX.ML_TEST.DOR_ANALYSIS_FF table
- Appropriate privileges to:
  - Read from DOR_ANALYSIS_FF
  - Execute Python code in Snowflake Notebooks
- No feature store or model registry required (as per design)

## Future Enhancements

Potential improvements to the model:
1. Add additional features (medical history, trial duration, previous trials)
2. Try ensemble methods (Random Forest, XGBoost)
3. Implement hyperparameter tuning
4. Address class imbalance with SMOTE or class weights
5. Deploy to production with monitoring
6. Integrate with feature store and model registry

## Notes

- **Python-based approach**: Uses XGBoost Python library instead of Snowflake SQL ML
- **Follows MEDPACE_ML_HOL patterns**: Implements the same workflow as reference notebooks (03 & 04)
- **No Feature Store/Model Registry**: As requested, doesn't use Snowflake ML Registry
- **Data Access**: Uses Snowpark Python API to load data, then converts to pandas for ML
- **Source Data**: Uses existing data from INFORMATICS_SANDBOX.ML_TEST.DOR_ANALYSIS_FF (read-only)
- **Visualizations**: Includes matplotlib/seaborn charts for EDA and model evaluation
- **Gender encoding**: Uses pandas string methods to handle case variations
- **Model Saving**: Optional joblib-based model persistence included

