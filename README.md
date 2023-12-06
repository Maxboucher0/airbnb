
# Airbnb User Behavior Prediction

## Overview

This repository contains the code and documentation for a machine learning project focused on predicting user behavior on the Airbnb platform. The project aims to understand and forecast whether a user will send a booking request based on various features, including session length and device usage.

## Project Structure

- **`data/`**: This directory contains the dataset used for training and evaluating the model. The dataset, sourced from Kaggle, comprises 7,756 user sessions from 630 unique visitors, with information such as visitor IDs, session IDs, session counts, and activities like messaging, searching, and booking requests.

- **`report/`**: Jupyter notebooks detailing the step-by-step process of data exploration, preprocessing, and model development. 

- **`src/`**: Python scripts containing utility functions, preprocessing steps, and the main machine learning pipeline (`MLpipe_GroupKFold_fbeta.py`). The script `README.py` provides instructions on setting up the project environment.

## Exploratory Data Analysis

The project begins with exploratory data analysis, aiming to understand data distributions, imbalances, and key features influencing user behavior. Feature engineering involves creating new variables, such as "session length," critical for predictive modeling.

## Machine Learning Pipeline

The predictive model, primarily using Logistic Regression, Random Forest, Support Vector Classifier (SVC), and K-Nearest Classifier, undergoes evaluation through GroupKFold cross-validation. Model performance metrics, including mean F-beta scores, are assessed against a baseline.

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/Maxboucher0/airbnb.git
   ```

2. Set up the Python environment:

   ```bash
   pip install -r requirements.txt
   ```

3. Explore the Jupyter notebooks in the `notebooks/` directory for a detailed walkthrough of the project.

4. Execute the machine learning pipeline script:

   ```bash
   python src/MLpipe_GroupKFold_fbeta.py
   ```

## Future Improvements

Possible enhancements include exploring additional data dimensions, incorporating more advanced machine learning models (XGBoost, Naive Bayes Classifier), and refining feature engineering for a more comprehensive predictive model.

Feel free to reach out for any questions or collaborations!
