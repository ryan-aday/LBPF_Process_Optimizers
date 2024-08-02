import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
data = {
    'gantry_movement': [10, 12, 10, 11, 10, 12, 11, 10, 12, 11],
    'laser_angle': [30, 30, 31, 30, 32, 31, 31, 32, 31, 30],
    'surface_roughness': [0.5, 0.55, 0.52, 0.53, 0.56, 0.54, 0.53, 0.55, 0.57, 0.56]
}

df = pd.DataFrame(data)

# 1. Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe())
print("\n")

# 2. Correlation Analysis
print("Pearson Correlation Matrix:")
pearson_corr = df.corr(method='pearson')
print(pearson_corr)
print("Pearson correlation measures linear relationships between variables.")
print("Range: -1 (perfect negative linear relationship) to 1 (perfect positive linear relationship), 0 indicates no linear relationship.")
print("\n")

print("Spearman Correlation Matrix:")
spearman_corr = df.corr(method='spearman')
print(spearman_corr)
print("Spearman correlation measures monotonic relationships between variables, using ranked data.")
print("Range: -1 (perfect negative monotonic relationship) to 1 (perfect positive monotonic relationship), 0 indicates no monotonic relationship.")
print("\n")

print("Kendall Correlation Matrix:")
kendall_corr = df.corr(method='kendall')
print(kendall_corr)
print("Kendall correlation measures ordinal associations between variables.")
print("Range: -1 (perfect negative ordinal association) to 1 (perfect positive ordinal association), 0 indicates no ordinal association.")
print("\n")

# 3. Regression Analysis
print("Regression Analysis:")
X = df[['gantry_movement', 'laser_angle']]
y = df['surface_roughness']
X = sm.add_constant(X)  # Adds a constant term to the predictor

model = sm.OLS(y, X).fit()
print(model.summary())
print("The statsmodels OLS (Ordinary Least Squares) method fits a linear regression model to the data by minimizing the sum of the squares of the residuals (the differences between observed and predicted values). It provides estimates of the regression coefficients and various statistics to evaluate the model's performance.")
print("\n")

# 4. ANOVA
print("ANOVA Table:")
model_formula = ols('surface_roughness ~ gantry_movement + laser_angle', data=df).fit()
anova_table = sm.stats.anova_lm(model_formula, typ=2)
print(anova_table)
print("In the ANOVA table:")
print("- 'df' (degrees of freedom) represents the number of independent values that can vary in the data.")
print("- 'sum_sq' (sum of squares) measures the variability in the data.")
print("- 'F' (F-statistic) is the ratio of between-group variance to within-group variance.")
print("- 'PR(>F)' is the p-value, and a p-value less than 0.05 typically indicates statistical significance.")
print("\n")

# 5. Try other regression types and provide the best fit

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)

# Ridge Regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

# Lasso Regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Determine the best model based on MSE
mse_dict = {'Linear Regression': mse_lin, 'Ridge Regression': mse_ridge, 'Lasso Regression': mse_lasso}
best_model = min(mse_dict, key=mse_dict.get)

print(f"Mean Squared Error for Linear Regression: {mse_lin}")
print(f"Mean Squared Error for Ridge Regression: {mse_ridge}")
print(f"Mean Squared Error for Lasso Regression: {mse_lasso}")
print(f"The best model based on MSE is: {best_model}")
print("\n")

# Optional: Visualizing Correlations
sns.pairplot(df)
plt.show()
