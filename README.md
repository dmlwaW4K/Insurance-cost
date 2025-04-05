# Predicting Medical Insurance Costs using Multiple Linear Regression

## Project Goal

The objective of this project is to predict individual medical costs billed by health insurance based on a few demographic and personal features. We utilize a Multiple Linear Regression model to understand the relationship between features like age, BMI, and number of children, and the resulting insurance charges.

## Dataset

The dataset used in this project is the **Medical Cost Personal Datasets** available on Kaggle.

*   **Source:** [Kaggle - Insurance Forecast by using Linear Regression](https://www.kaggle.com/datasets/mirichoi0218/insurance)
*   **File Used:** `insurance.csv`

The dataset contains information such as age, sex, BMI (body mass index), number of children, smoker status, region, and individual medical costs billed by the health insurance (charges). For this initial analysis, we focus primarily on the numerical features: `age`, `bmi`, and `children`.

## Dependencies

This project uses the following Python libraries:

*   `pandas`: For data manipulation and loading the CSV file.
*   `numpy`: For numerical operations.
*   `matplotlib`: For creating static, interactive, and animated visualizations.
*   `seaborn`: For high-level data visualization based on matplotlib.
*   `scikit-learn`: For machine learning tasks, including splitting data, training the model, and evaluating performance.
*   `os`: (Optional, used in the script for basic file path handling)

You can install the necessary libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
