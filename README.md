# AI/ML-Based Predictive Maintenance and Vehicle Health Monitoring System

## Module 3 – Connectivity Project

---

## 1. Project Overview

This project implements an Artificial Intelligence and Machine Learning based predictive maintenance and vehicle health monitoring system.

The primary objective is to use vehicle/engine sensor information to estimate the likelihood of an engine condition requiring maintenance and convert that prediction into an understandable vehicle health score and maintenance alert.

The project is based on the first project option specified in the Module 3 project brief:

**Project Title:** AI/ML-Based Predictive Maintenance and Vehicle Health Monitoring System

**Goal:** Predict vehicle/component failure before it occurs.

The project brief identifies several AI/ML technologies and models, including:

- Random Forest
- XGBoost
- LSTM
- Logistic Regression
- Decision Tree
- Transformer
- GRU
- Explainable AI (XAI)

The implementation in this repository focuses on classical machine learning models that can be trained efficiently on the selected public engine-sensor dataset:

1. Logistic Regression
2. Random Forest
3. XGBoost

Explainable AI is incorporated through:

- Random Forest feature importance
- SHAP-based local explanation

The system converts the machine-learning prediction into:

- Failure probability
- Vehicle health score
- Vehicle status
- Maintenance alert
- Major contributing sensor factors

---

# 2. Problem Statement

Modern vehicles contain multiple sensors that continuously monitor the condition and operation of different components.

Sensor values such as:

- Engine RPM
- Oil pressure
- Fuel pressure
- Coolant pressure
- Oil temperature
- Coolant temperature

can contain useful information about the operating condition of an engine.

Traditional maintenance approaches are often based on fixed service intervals or reactive repair after a component fails.

A predictive maintenance system attempts to identify abnormal or potentially problematic conditions before a serious failure occurs.

This project demonstrates a machine-learning pipeline that takes sensor measurements as input and predicts the engine condition.

The prediction is then translated into an easy-to-understand monitoring output.

Example:

    Vehicle Health: 72%
    Failure Probability: 68%
    Status: WARNING

    Major contributing factors:
    1. High vibration
    2. Rising temperature
    3. Abnormal oil pressure

The exact factors available in this implementation depend on the selected dataset.

---

# 3. Project Objectives

The main objectives of the project are:

1. Obtain a suitable automotive/engine predictive-maintenance dataset.
2. Load and inspect the sensor data.
3. Perform exploratory data analysis.
4. Identify missing values and prepare the data.
5. Separate sensor features from the target variable.
6. Divide the dataset into training and testing sets.
7. Train multiple machine-learning classification models.
8. Compare the performance of the models.
9. Select a practical model for vehicle-health prediction.
10. Calculate failure probability.
11. Convert failure probability into a vehicle health score.
12. Generate a maintenance status.
13. Generate a maintenance alert.
14. Identify important sensor features.
15. Demonstrate Explainable AI using feature importance and SHAP.
16. Save the trained model.
17. Save prediction results as CSV files.
18. Package project outputs for download.

---

# 4. Project Architecture

The architecture described in the project brief can be represented as:

    Vehicle Sensors
           |
           v
    Sensor Measurements
           |
           v
    Data Preprocessing
           |
           v
    Machine Learning Model
           |
           v
    Failure Probability
           |
           v
    Vehicle Health Score
           |
           v
    Maintenance Alert

The implemented notebook follows this general workflow.

Detailed pipeline:

    Public Engine Sensor Dataset
              |
              v
        Data Loading
              |
              v
    Exploratory Data Analysis
              |
              v
    Missing Value Handling
              |
              v
       Train/Test Split
              |
              v
    +-----------------------+
    | Machine Learning      |
    | Models                |
    |                       |
    | Logistic Regression   |
    | Random Forest         |
    | XGBoost               |
    +-----------------------+
              |
              v
       Model Evaluation
              |
              v
    Failure Probability
              |
              v
    Vehicle Health Score
              |
              v
       Status Generation
              |
              v
     Maintenance Alert
              |
              v
        XAI Analysis

---

# 5. Dataset

## Dataset Name

Predictive Maintenance Engine Sensor Dataset

## Dataset Source

Hugging Face:

https://huggingface.co/datasets/mukherjee78/predictive-maintenance-engine-data

The notebook is configured to download the dataset directly.

Dataset CSV URL used by the notebook:

https://huggingface.co/datasets/mukherjee78/predictive-maintenance-engine-data/resolve/main/raw_data.csv

The notebook therefore does not require manually downloading the CSV before execution.

---

# 6. Dataset Size

The selected dataset contains approximately 19,535 engine records.

Each record represents an engine observation containing sensor measurements and an engine-condition target.

The target column used by the notebook is:

    Engine Condition

The implementation treats this as a binary classification problem.

The exact class distribution is displayed automatically when the notebook is executed.

---

# 7. Dataset Features

The selected public dataset provides the following sensor features used by the model:

1. Engine rpm
2. Lub oil pressure
3. Fuel pressure
4. Coolant pressure
5. lub oil temp
6. Coolant temp

Target:

    Engine Condition

These features are used as machine-learning inputs.

---

# 8. Important Dataset Scope Note

The original project brief lists the following possible input parameters:

- Engine temperature
- RPM
- Oil pressure
- Vibration
- Battery voltage
- Coolant temperature
- Fuel consumption
- Vehicle speed
- Operating hours

The selected public dataset does NOT contain every one of these parameters.

In particular, this implementation does not have direct columns for:

- Vibration
- Battery voltage
- Vehicle speed
- Operating hours
- Fuel consumption

Therefore, these values are NOT artificially generated or fabricated.

The notebook uses the sensor columns that are actually present in the selected public dataset.

This is important for reproducibility and scientific integrity.

If a future version of the project uses a richer vehicle telemetry dataset containing these additional parameters, the feature set can be expanded.

---

# 9. Why the Problem is Treated as Classification

The dataset contains an engine-condition target.

The machine-learning task is therefore formulated as binary classification.

The model learns a relationship between:

    Sensor measurements

and:

    Engine Condition

During prediction, the model provides probabilities for the possible classes.

The probability associated with the maintenance/fault condition is used as the project's demonstration failure probability.

---

# 10. Technologies Used

## Programming Language

Python

## Development Environment

Google Colab / Jupyter Notebook

## Libraries

The notebook uses:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- SHAP
- Joblib

Packages are installed automatically where necessary.

---

# 11. Machine Learning Models

## 11.1 Logistic Regression

Logistic Regression is used as a baseline classification model.

It estimates the probability that an observation belongs to a particular class.

Advantages:

- Simple
- Fast
- Easy to interpret
- Useful as a baseline
- Provides probability estimates

The Logistic Regression pipeline contains:

1. Median imputation
2. Standard scaling
3. Logistic Regression classifier

---

# 12. Random Forest

Random Forest is one of the main models in the project.

Random Forest combines multiple decision trees.

Each tree learns patterns in the training data and the forest combines their predictions.

Advantages:

- Handles nonlinear relationships
- Works well with numerical sensor data
- Can capture feature interactions
- Provides feature importance
- Generally requires less feature scaling than linear models
- Provides probability estimates

The implementation uses a Random Forest with multiple decision trees and balanced class weighting.

Random Forest is also used for the project's primary vehicle-health demonstration.

---

# 13. XGBoost

XGBoost is another tree-based machine-learning model included in the project.

XGBoost uses gradient boosting to construct an ensemble of decision trees.

Advantages:

- Strong performance on tabular data
- Captures nonlinear relationships
- Handles feature interactions
- Provides probability estimates
- Widely used for structured-data classification

The notebook trains XGBoost alongside Logistic Regression and Random Forest so their performance can be compared.

---

# 14. Data Preprocessing

Before training, the dataset is separated into:

    X = input features

and:

    y = target

The target is:

    Engine Condition

Missing values are handled using median imputation.

For Logistic Regression, standardization is applied because linear models can benefit from features being on comparable numerical scales.

Tree-based models such as Random Forest and XGBoost do not require standard scaling in the same way.

The preprocessing operations are implemented using Scikit-learn pipelines.

This helps ensure that the same preprocessing procedure is used during training and prediction.

---

# 15. Train/Test Split

The dataset is divided into:

    80% training data
    20% testing data

A fixed random seed is used so that the experiment can be reproduced.

Stratification is also used to preserve approximately the same class distribution between the training and testing sets.

---

# 16. Model Evaluation

The project evaluates the models using multiple metrics.

## Accuracy

Accuracy measures the fraction of predictions that are correct.

    Accuracy =
    Correct Predictions / Total Predictions

Accuracy alone may not always be sufficient for maintenance applications, particularly when one class is more important than the other.

---

# 17. Precision

Precision answers:

"Of the observations predicted as the maintenance/fault class, how many actually belong to that class?"

    Precision =
    True Positives /
    (True Positives + False Positives)

---

# 18. Recall

Recall answers:

"Of all actual maintenance/fault observations, how many did the model detect?"

    Recall =
    True Positives /
    (True Positives + False Negatives)

Recall is particularly relevant when missing a potentially problematic engine condition is costly.

---

# 19. F1 Score

F1 combines precision and recall.

    F1 =
    2 × Precision × Recall /
    (Precision + Recall)

It provides a balance between precision and recall.

---

# 20. ROC-AUC

ROC-AUC measures how well the model separates the two classes across different classification thresholds.

The notebook calculates ROC-AUC using the model's probability output.

---

# 21. Confusion Matrix

The notebook generates a confusion matrix for the Random Forest model.

The matrix contains:

- True Negatives
- False Positives
- False Negatives
- True Positives

The confusion matrix provides a direct view of the types of errors made by the classifier.

---

# 22. ROC Curve

The notebook also generates an ROC curve.

The curve uses:

- False Positive Rate
- True Positive Rate

The ROC-AUC value is calculated from the probability predictions.

---

# 23. Vehicle Health Score

The project converts predicted failure probability into a simple health score.

The demonstration formula is:

    Vehicle Health Score =
    100 × (1 − Failure Probability)

For example:

    Failure Probability = 0.28

then:

    Health Score = 100 × (1 − 0.28)
                 = 72%

Therefore:

    Vehicle Health = 72%

This health score is intended as a project demonstration metric rather than a certified automotive diagnostic measurement.

---

# 24. Failure Probability

The Random Forest model generates class probabilities.

The probability associated with the maintenance/fault class is used as the project's demonstration failure probability.

Example:

    Failure Probability = 68%

The value is model-derived.

It should not be interpreted as an exact real-world probability of mechanical failure without proper calibration, validation, field data, and domain-specific thresholds.

---

# 25. Vehicle Status

The notebook converts failure probability into three demonstration status levels.

## HEALTHY

Failure probability:

    < 40%

Meaning:

    No immediate maintenance action according to the
    demonstration threshold.

---

## WARNING

Failure probability:

    40% to below 70%

Meaning:

    The system recommends inspection according to the
    demonstration threshold.

---

## CRITICAL

Failure probability:

    70% or higher

Meaning:

    The system indicates a high-risk condition according to
    the demonstration threshold.

These thresholds are project demonstration thresholds.

They should be calibrated using real fleet data and engineering requirements before being used in an actual vehicle.

---

# 26. Maintenance Alert

The system generates a simple maintenance recommendation.

For HEALTHY:

    NO IMMEDIATE ACTION

For WARNING or CRITICAL:

    INSPECTION REQUIRED

This makes the machine-learning output easier to understand for a non-technical user.

---

# 27. Explainable AI (XAI)

Machine-learning models can sometimes be difficult to interpret.

For a predictive-maintenance system, it is useful to know which sensor variables contributed strongly to a prediction.

The project therefore includes two XAI approaches.

---

# 28. Random Forest Feature Importance

Random Forest provides feature importance values.

The notebook:

1. Extracts feature importance.
2. Sorts the features.
3. Displays them in a table.
4. Generates a feature-importance graph.

This allows the project to identify which sensor measurements were most influential globally within the Random Forest model.

---

# 29. SHAP

The notebook also includes an optional SHAP explanation.

SHAP stands for:

    SHapley Additive exPlanations

SHAP can provide a local explanation for an individual prediction.

The purpose is to show how individual sensor features influence the model's output for a particular observation.

The notebook generates a SHAP contribution table.

This provides a stronger explanation mechanism than relying only on global feature importance.

---

# 30. Single-Vehicle Prediction

The notebook includes a reusable function:

    predict_vehicle_health()

The function accepts sensor values and returns:

- Predicted condition
- Failure probability
- Vehicle health score
- Status
- Maintenance alert

Example structure:

    sensor_values = {
        'Engine rpm': ...,
        'Lub oil pressure': ...,
        'Fuel pressure': ...,
        'Coolant pressure': ...,
        'lub oil temp': ...,
        'Coolant temp': ...
    }

The model then produces a monitoring result.

---

# 31. Example Monitoring Output

A typical project-style output can look like:

    ========================================
            VEHICLE HEALTH MONITOR
    ========================================

    Vehicle Health: 72.0%
    Failure Probability: 28.0%
    Status: HEALTHY
    Maintenance Alert: NO IMMEDIATE ACTION

The actual numbers depend on the input sensor measurements and trained model.

---

# 32. Maintenance Alert Summary

The notebook also generates a structured maintenance alert.

Example:

    Vehicle Health: 72.0%
    Failure Probability: 28.0%
    Status: HEALTHY
    Maintenance Alert: NO IMMEDIATE ACTION

    Major contributing factors:
    - Engine rpm
    - Coolant temp
    - Lub oil pressure

The contributing factors shown by the basic alert function are based on global Random Forest feature importance.

The optional SHAP section provides a more specific local explanation for an individual observation.

---

# 33. Exploratory Data Analysis

The notebook performs exploratory analysis before model training.

It includes:

- Dataset shape
- First rows
- Feature names
- Missing-value analysis
- Target distribution
- Descriptive statistics
- Feature histograms
- Correlation matrix

These visualizations help understand the dataset before training the models.

---

# 34. Generated Graphs

The notebook generates several visualizations.

These include:

1. Sensor feature distributions
2. Correlation matrix
3. Confusion matrix
4. ROC curve
5. Random Forest feature importance

Additional graphs can be added later.

---

# 35. Generated Files

The notebook can generate the following files:

    predictive_maintenance_random_forest.pkl

    vehicle_health_predictions.csv

    model_comparison_results.csv

    feature_importance.csv

    SHAP_explanation.csv

    project_summary.txt

The trained model file can be reused for later inference without retraining the model.

---

# 36. Model File

The trained Random Forest model is saved as:

    predictive_maintenance_random_forest.pkl

The model is serialized using Joblib.

This makes it possible to load the trained model later.

Example:

    import joblib

    model = joblib.load(
        "predictive_maintenance_random_forest.pkl"
    )

---

# 37. Prediction CSV

The monitoring predictions are saved as:

    vehicle_health_predictions.csv

The file contains sensor information together with:

- Actual condition
- Failure probability
- Vehicle health score
- Status
- Maintenance alert

This file can be used for further analysis.

---

# 38. Model Comparison CSV

The model comparison output is saved as:

    model_comparison_results.csv

It contains evaluation metrics for:

- Logistic Regression
- Random Forest
- XGBoost

Metrics include:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

---

# 39. Feature Importance CSV

The feature importance results are saved as:

    feature_importance.csv

This file contains the sensor features and their corresponding Random Forest importance values.

---

# 40. SHAP Explanation CSV

The SHAP explanation is saved as:

    SHAP_explanation.csv

This contains feature-level SHAP contributions for the selected sample.

Positive and negative contributions indicate different directions of influence on the model's output.

---

# 41. Output ZIP

The final Colab cell can collect the generated project files into one ZIP archive.

The ZIP can contain:

    Graphs
    CSV prediction files
    Model file
    SHAP results
    Feature importance
    Project summary
    Notebook

This makes it easier to submit or share the complete project.

---

# 42. Google Colab

The notebook is designed to run in Google Colab.

Steps:

1. Open Google Colab.
2. Upload the `.ipynb` notebook.
3. Select a Python runtime.
4. Run the cells from top to bottom.
5. Wait for dataset download.
6. Wait for model training.
7. Review the evaluation results.
8. Review the generated graphs.
9. Run the SHAP section if required.
10. Run the final ZIP-generation cell.
11. Download the generated project ZIP.

No local Python installation is required when running the notebook through Google Colab.

---

# 43. Installation

The notebook automatically installs the required additional packages.

The primary packages are:

    numpy
    pandas
    matplotlib
    scikit-learn
    xgboost
    shap
    joblib

The notebook contains the installation command:

    !pip -q install xgboost shap

Google Colab already provides many of the remaining scientific Python packages.

---

# 44. Repository Structure

A recommended GitHub repository structure is:

    AI-Predictive-Maintenance/
    |
    |-- README.md
    |
    |-- notebooks/
    |     |
    |     `-- Predictive_Maintenance_Vehicle_Health_Monitoring_Colab.ipynb
    |
    |-- data/
    |     |
    |     `-- DATASET_LINK.txt
    |
    |-- models/
    |     |
    |     `-- predictive_maintenance_random_forest.pkl
    |
    |-- outputs/
    |     |
    |     |-- model_comparison_results.csv
    |     |-- vehicle_health_predictions.csv
    |     |-- feature_importance.csv
    |     `-- SHAP_explanation.csv
    |
    `-- requirements.txt

The raw dataset does not need to be committed to the repository because the notebook downloads it directly from the public dataset source.

---

# 45. Running the Project

## Step 1 – Open the Notebook

Open:

    Predictive_Maintenance_Vehicle_Health_Monitoring_Colab.ipynb

in Google Colab.

## Step 2 – Install Packages

Run the installation cell.

## Step 3 – Load Dataset

Run the dataset-loading cell.

The notebook downloads the CSV automatically.

## Step 4 – Inspect Dataset

Run the EDA cells.

Review:

- Dataset shape
- Missing values
- Feature distributions
- Target distribution
- Correlations

## Step 5 – Train Models

Run the model-training cells.

The notebook trains:

    Logistic Regression
    Random Forest
    XGBoost

## Step 6 – Evaluate Models

Review:

    Accuracy
    Precision
    Recall
    F1
    ROC-AUC

## Step 7 – Inspect XAI

Review:

    Random Forest Feature Importance
    SHAP Explanation

## Step 8 – Test Vehicle Health

Use the single-vehicle prediction function.

## Step 9 – Generate Project Outputs

Run the final output-generation cell.

## Step 10 – Download ZIP

The final cell creates and downloads:

    Predictive_Maintenance_Project_Outputs.zip

---

# 46. Reproducibility

The project uses a fixed random state:

    RANDOM_STATE = 42

This makes the train/test split and model training more reproducible.

However, exact results can still vary slightly depending on:

- Library versions
- Dataset updates
- Runtime environment
- Hardware
- XGBoost version
- Scikit-learn version

---

# 47. Limitations

This project is an educational/prototype implementation.

It is NOT a production automotive diagnostic system.

Important limitations include:

1. The selected dataset is not a complete vehicle telemetry dataset.
2. The dataset does not contain every parameter listed in the project brief.
3. Real vehicle failure prediction requires real-world fleet data.
4. Failure probabilities should ideally be calibrated.
5. Maintenance thresholds should be determined using engineering validation.
6. Sensor noise can affect predictions.
7. Sensor calibration can affect predictions.
8. Dataset bias can affect generalization.
9. Model performance on this dataset does not guarantee performance on real vehicles.
10. The health score is a project-level derived metric.
11. The demonstration status thresholds are not automotive safety standards.
12. The current implementation does not use LSTM, GRU or Transformer models.
13. The current implementation does not process live CAN-bus or OBD-II streams.
14. The system does not directly control any vehicle component.
15. The system should not be used as the sole basis for safety-critical maintenance decisions.

---

# 48. Future Improvements

The project can be extended significantly.

## 48.1 Add More Sensor Data

Future versions can include:

- Vibration
- Battery voltage
- Vehicle speed
- Operating hours
- Fuel consumption
- Additional engine temperatures
- Exhaust parameters
- Brake parameters
- Transmission parameters

---

# 49. Live Vehicle Connectivity

A future implementation could connect the ML system to actual vehicle telemetry.

Possible architecture:

    Vehicle Sensors
          |
          v
    CAN / OBD-II
          |
          v
    Edge Device
          |
          v
    Data Processing
          |
          v
    ML Model
          |
          v
    Vehicle Health
          |
          v
    Mobile/Web Dashboard

---

# 50. Deep Learning Extension

The project brief also identifies LSTM and GRU.

These models could be used when sequential sensor data is available.

For example:

    Time t-10
    Time t-9
    ...
    Time t-1
    Time t

can be supplied as a sequence to an LSTM or GRU.

This can allow the model to learn temporal patterns such as gradually increasing temperature or vibration.

---

# 51. Transformer Extension

A Transformer-based model could be used for longer sensor sequences.

Potential input:

    RPM sequence
    Temperature sequence
    Pressure sequence
    Vibration sequence

The Transformer could learn relationships between different points in the sensor timeline.

This would require a substantially larger and properly structured time-series dataset.

---

# 52. Real-Time Dashboard

A future version could provide a dashboard containing:

    Vehicle Health: 72%

    Failure Probability: 28%

    Status: HEALTHY

    Engine RPM: 900
    Oil Pressure: 3.5
    Fuel Pressure: 8.0
    Coolant Pressure: 2.5
    Oil Temperature: 80
    Coolant Temperature: 80

    Maintenance:
    NO IMMEDIATE ACTION

Possible technologies:

- Streamlit
- Flask
- FastAPI
- React
- Android application
- Web dashboard

---

# 53. Mobile Application Extension

The project could be extended into an Android vehicle-health application.

The mobile application could display:

- Vehicle health
- Failure probability
- Sensor values
- Maintenance alerts
- Historical health
- Model explanations
- Maintenance history

A backend API could provide predictions to the mobile application.

---

# 54. Edge AI Extension

For automotive applications, inference could eventually be performed locally on an edge device.

Example:

    Vehicle Sensor
          |
          v
    Edge Computer
          |
          v
    ML Inference
          |
          v
    Local Health Score
          |
          v
    Driver Alert

This can reduce network dependency and latency.

---

# 55. Safety Considerations

Predictive maintenance models should be treated as decision-support systems unless properly validated and certified for the intended automotive application.

A model prediction should not automatically trigger a safety-critical vehicle action without appropriate engineering safeguards.

A production system would require:

- Sensor validation
- Fault tolerance
- Model validation
- Data-quality monitoring
- Calibration
- Threshold validation
- Fail-safe behavior
- Extensive testing
- Real-world validation

---

# 56. Academic / Project Demonstration Scope

This repository is intended to demonstrate the complete machine-learning workflow:

    Data
      ↓
    Preprocessing
      ↓
    Model Training
      ↓
    Evaluation
      ↓
    Prediction
      ↓
    Explainability
      ↓
    Vehicle Health
      ↓
    Maintenance Alert

It demonstrates how raw engine sensor measurements can be transformed into a higher-level vehicle-health output using machine learning.

---

# 57. Key Project Deliverables

The completed project provides:

- Public dataset source
- Google Colab notebook
- Data preprocessing
- Exploratory analysis
- Multiple ML models
- Model comparison
- Classification metrics
- Confusion matrix
- ROC curve
- Random Forest feature importance
- SHAP-based XAI
- Failure probability
- Vehicle health score
- Maintenance status
- Maintenance alert
- Single-vehicle prediction
- Trained model export
- CSV prediction export
- Final ZIP output

---

# 58. Conclusion

This project demonstrates an AI/ML-based predictive maintenance and vehicle health monitoring workflow.

The system accepts engine sensor measurements and applies machine-learning classification models to identify the predicted engine condition.

Random Forest and XGBoost provide nonlinear machine-learning approaches, while Logistic Regression provides a baseline model.

The model output is transformed into a failure probability and a simple vehicle health score.

Explainable AI techniques are included so that the model's behavior can be examined through feature importance and SHAP explanations.

The project therefore covers the complete pipeline from sensor data to predictive analysis and maintenance alert generation.

The implementation is intended as a foundation that can be expanded with richer automotive telemetry, time-series deep learning, live vehicle connectivity, mobile dashboards, edge inference, and real-world validation.

---

# 59. Dataset Reference

Dataset:

Predictive Maintenance Engine Sensor Dataset

Source:

Hugging Face

URL:

https://huggingface.co/datasets/mukherjee78/predictive-maintenance-engine-data

Raw CSV:

https://huggingface.co/datasets/mukherjee78/predictive-maintenance-engine-data/resolve/main/raw_data.csv

---

# 60. Project Brief Reference

This implementation corresponds to:

Module 3 (Connectivity) Project

Project Option 1:

AI/ML-Based Predictive Maintenance and Vehicle Health Monitoring System

The project brief specifies the goal of predicting vehicle/component failure before it occurs and identifies vehicle/engine sensor inputs, ML/DL technologies, failure probability, vehicle health score and maintenance alerts as the intended system components.

---

# 61. Author / Repository Information

Project:

AI/ML-Based Predictive Maintenance and Vehicle Health Monitoring System

Module:

Module 3 – Connectivity

Project Type:

AI / Machine Learning / Automotive

Environment:

Google Colab

Language:

Python

Primary Models:

Logistic Regression
Random Forest
XGBoost

Explainability:

Random Forest Feature Importance
SHAP

Dataset:

Predictive Maintenance Engine Sensor Dataset

---

# END OF README
