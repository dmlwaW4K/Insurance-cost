import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os # To handle file paths potentially

# --- Load the Data ---
# Adjust the path if your file is elsewhere (e.g., 'data/insurance.csv')
file_path = 'insurance.csv'

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    print("Please download the dataset from Kaggle and place it correctly.")
    # Exit or handle appropriately
    exit() # Or raise FileNotFoundError

df = pd.read_csv(file_path)

# --- Initial Data Inspection ---
print("--- Initial Data Inspection ---")
print("First 5 rows:")
print(df.head())
print("\nData Info:")
df.info() # Check data types and non-null counts
print("\nDescriptive Statistics:")
print(df.describe()) # Get summary stats for numerical columns
print("\nChecking for missing values:")
print(df.isnull().sum()) # This dataset is usually clean, but good practice

# ---  Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis ---")

# Select features (independent variables) and target (dependent variable)
# We'll start with numerical features only for basic Multiple Linear Regression
features = ['age', 'bmi', 'children']
target = 'charges'

# Visualize relationships between numerical features and the target
# Pairplot shows scatter plots for relationships and histograms for distributions
print("\nGenerating Pairplot...")
sns.pairplot(df[features + [target]])
plt.suptitle('Pairplot of Numerical Features and Target (Charges)', y=1.02) # Adjust title position
plt.tight_layout()
plt.show()

# Check the distribution of the target variable 'charges'
print("\nGenerating Distribution Plot for Charges...")
plt.figure(figsize=(8, 5))
sns.histplot(df[target], kde=True)
plt.title('Distribution of Insurance Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Observation: Charges distribution is often right-skewed. Linear regression assumes
# normally distributed *residuals*, not necessarily the target itself, but heavy skew
# in the target can sometimes affect model performance or assumptions.

# Correlation Matrix Heatmap
print("\nGenerating Correlation Heatmap...")
plt.figure(figsize=(8, 6))
# Select only the columns we are currently interested in for the correlation matrix
correlation_matrix = df[features + [target]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Numerical Features and Target)')
plt.show()
# Observations: Age and BMI seem to have a positive correlation with charges.
# Children has a weaker correlation.

# ---  Prepare Data for Modeling ---
print("\n--- Preparing Data for Modeling ---")
X = df[features] # Features (independent variables)
y = df[target]   # Target (dependent variable)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# random_state=42 ensures reproducibility of the split

print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

# ---  Train the Multiple Linear Regression Model ---
print("\n--- Training the Model ---")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")

# ---  Examine Model Parameters ---
print("\n--- Model Parameters ---")
intercept = model.intercept_
coefficients = model.coef_

print(f"Intercept (β₀): {intercept:.2f}")
# Pair features with their coefficients for clarity
feature_coeffs = pd.DataFrame({'Feature': features, 'Coefficient (β)': coefficients})
print("Coefficients:")
print(feature_coeffs)

# Interpretation of Coefficients:
# - Intercept: The predicted charge when age, bmi, and children are all zero (often not practically meaningful).
# - Age Coefficient: For a one-year increase in age, holding BMI and children constant, the insurance charge is predicted to increase by approx. $X.
# - BMI Coefficient: For a one-unit increase in BMI, holding age and children constant, the insurance charge is predicted to increase by approx. $Y.
# - Children Coefficient: For one additional child, holding age and BMI constant, the insurance charge is predicted to increase/decrease by approx. $Z.

# ---  Make Predictions on the Test Set ---
print("\n--- Making Predictions ---")
y_pred = model.predict(X_test)
print("Predictions made on the test set.")

# ---  Evaluate the Model ---
print("\n--- Evaluating the Model ---")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred) # R-squared (Coefficient of Determination)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}") # Easier to interpret (in dollars)
print(f"R-squared (R²): {r2:.4f}")

# Interpretation of Evaluation Metrics:
# - RMSE: Indicates the typical prediction error in dollars. A lower RMSE is better.
# - R²: Represents the proportion of the variance in the insurance charges that is predictable
#       from age, bmi, and children using this linear model. E.g., R²=0.12 means ~12%
#       of the variability in charges is explained by these features in this model.
#       (Note: R² is often quite low for this specific subset of features!)

# ---  Visualize Results ---
print("\n--- Visualizing Results ---")

# Scatter plot of Actual vs. Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Charges ($)')
plt.ylabel('Predicted Charges ($)')
plt.title('Actual vs. Predicted Insurance Charges')
# Add a line for perfect predictions (y=x)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', linewidth=2)
plt.grid(True)
plt.show()
# Observation: If the points cluster closely around the red dashed line, the model predicts well.
# Often, for this dataset with only these features, the points will be widely scattered.

# Residual Plot (Predicted vs. Residuals)
# Residuals = Actual - Predicted
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.xlabel('Predicted Charges ($)')
plt.ylabel('Residuals ($)')
plt.title('Residual Plot (Predicted vs. Residuals)')
plt.axhline(y=0, color='red', linestyle='--') # Line at zero residual
plt.grid(True)
plt.show()
# Observation: For a good linear regression fit, residuals should be randomly scattered
# around the horizontal line at zero, with no clear patterns (like a curve or funnel shape).
# A pattern suggests the linear model might not be capturing the relationship well, or
# assumptions (like constant variance of errors) might be violated.

# --- Conclusion ---
print("\n--- Conclusion ---")
print(f"The multiple linear regression model using features {features} to predict insurance charges resulted in an R² of {r2:.4f}.")
print(f"This means approximately {r2*100:.2f}% of the variance in charges can be explained by these features in this linear model.")
print(f"The Root Mean Squared Error (RMSE) was ${rmse:.2f}, indicating the average magnitude of prediction errors.")
print("\nKey relationships found (interpret coefficients from step 7):")
# You can re-iterate the coefficient interpretations here.
print(feature_coeffs)
print("\nLimitations/Next Steps:")
print("- The R² is relatively low, suggesting these features alone don't explain most of the variation in charges.")
print("- Important factors like 'smoker' (categorical) were ignored. Including them (after encoding) would likely improve the model significantly.")
print("- The relationship might not be purely linear (e.g., age effect might accelerate).")
print("- Residual plot might show patterns indicating violations of LR assumptions.")
