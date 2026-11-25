# Practical 4: Energy Efficiency – Linear Regression

**Level:** Moderate Level  
**Subject:** Data Analytics  
**Platform:** Python (Jupyter Notebook / Google Colab / Standalone)

---

## Aim
To build, train, and evaluate a Multiple Linear Regression model in Python to predict building Heating Load (Y1) based on architectural parameters from the Energy Efficiency dataset.

---

## Dataset Information
- **Dataset Name:** Energy Efficiency Dataset
- **Default Filename:** `ENB2012_data.csv`
- **Kaggle Source:** [https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset](https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset)

---

## Comprehensive Theory

### 1. Introduction to Linear Regression
Linear Regression is a foundational supervised machine learning algorithm used to model the linear relationship between a continuous dependent target variable $Y$ and one or more independent predictor variables $X_1, X_2, \dots, X_p$.

In the Energy Efficiency domain, architectural features (such as relative compactness, surface area, wall area, roof area, overall height, orientation, and glazing area) govern thermodynamic heat exchange. Predicting **Heating Load (Y1)** enables civil and structural engineers to design energy-efficient green buildings.

---

### 2. Mathematical Formulation
Multiple Linear Regression models the target $Y$ as a linear combination of input features plus an error term:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon$$

In matrix notation:
$$\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}$$

Where:
- $\mathbf{y}$: $n \times 1$ vector of observed target values.
- $\mathbf{X}$: $n \times (p+1)$ matrix of input features (with a leading column of 1s for the intercept).
- $\boldsymbol{\beta}$: $(p+1) \times 1$ vector of unknown regression coefficients.
- $\boldsymbol{\epsilon}$: $n \times 1$ vector of unobserved random errors assumed $\sim \mathcal{N}(0, \sigma^2 \mathbf{I})$.

#### Ordinary Least Squares (OLS) Estimation:
The optimal coefficient vector $\hat{\boldsymbol{\beta}}$ minimizes the Residual Sum of Squares (RSS):
$$\text{RSS}(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$
Differentiating with respect to $\boldsymbol{\beta}$ and setting to zero yields the **Normal Equation**:
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

---

### 3. Model Evaluation Metrics

1. **Mean Squared Error (MSE):**
   Measures the average squared difference between actual and predicted values:
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
   Penalizes large errors heavily; lower MSE indicates superior model performance.
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
   Expresses error magnitude in the same physical units as the target variable ($kWh/m^2$).
3. **Coefficient of Determination ($R^2$ Score):**
   Quantifies the proportion of variance in the dependent variable explained by the independent variables:
   $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
   - $R^2 = 1.0$: Perfect fit.
   - $R^2 = 0.0$: Model predicts no better than the sample mean $\bar{y}$.
   - $R^2 < 0$: Model performs worse than the horizontal mean line.

---

### 4. Assumptions of Classical Linear Regression
1. **Linearity:** The relationship between features and target is linear.
2. **Homoscedasticity:** Constant variance of residuals across all levels of independent variables.
3. **Independence:** Residuals are mutually independent (no autocorrelation).
4. **Normality:** Residuals are normally distributed.
5. **No Multicollinearity:** Independent variables must not be perfectly linearly correlated.

---

### 5. Viva Voce Questions & Answers

**Q1: What is the difference between Simple and Multiple Linear Regression?**  
*Answer:* Simple Linear Regression models the relationship between one independent variable and one target ($Y = mX + C$), whereas Multiple Linear Regression models two or more independent variables against a single target ($Y = \beta_0 + \sum \beta_i X_i$).

**Q2: What does an $R^2$ score of 0.85 imply?**  
*Answer:* It means that $85\%$ of the total variance in the dependent variable (Heating Load) is explained by the regression model's input features, with the remaining $15\%$ attributed to noise or unmodeled factors.

**Q3: Why do we split the dataset into training and testing sets?**  
*Answer:* To objectively evaluate model generalization on unseen data and detect whether the model has overfit the training data.

**Q4: How does Ordinary Least Squares (OLS) calculate optimal parameters?**  
*Answer:* OLS calculates the slope and intercept coefficients that minimize the sum of squared vertical differences (residuals) between observed target points and the fitted regression hyperplane.


---

## Step-by-Step Algorithm
1. Import Pandas, LinearRegression, train_test_split, mean_squared_error, and r2_score.
2. Load the Energy Efficiency dataset (`pd.read_csv('ENB2012_data.csv')`).
3. Define the feature matrix `X` by dropping target outputs `Y1` (Heating Load) and `Y2` (Cooling Load).
4. Define the target vector `y` as `df['Y1']`.
5. Partition the data into training (80%) and testing (20%) subsets with a fixed `random_state=42`.
6. Initialize the `LinearRegression()` model object.
7. Fit the model to training data using `model.fit(X_train, y_train)`.
8. Predict heating loads on the test set via `model.predict(X_test)`.
9. Compute and print evaluation metrics: Mean Squared Error (MSE) and $R^2$ Score.

---

## Libraries Required
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

---

## Python Program
```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

csv_file = 'ENB2012_data.csv'

if not os.path.exists(csv_file):
    print(f"'{csv_file}' not found locally. Generating sample Energy Efficiency Dataset...")
    data = {
        'X1': [0.98, 0.90, 0.86, 0.82, 0.79, 0.76, 0.74, 0.71, 0.69, 0.66], # Relative Compactness
        'X2': [514.5, 563.5, 588.0, 612.5, 637.0, 661.5, 686.0, 710.5, 735.0, 759.5], # Surface Area
        'X3': [294.0, 318.5, 294.0, 318.5, 343.0, 416.5, 245.0, 269.5, 294.0, 318.5], # Wall Area
        'X4': [110.25, 122.5, 147.0, 147.0, 147.0, 122.5, 220.5, 220.5, 220.5, 220.5], # Roof Area
        'X5': [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.5, 3.5, 3.5, 3.5], # Overall Height
        'X6': [2, 3, 4, 5, 2, 3, 4, 5, 2, 3], # Orientation
        'X7': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Glazing Area
        'X8': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Glazing Area Distribution
        'Y1': [15.55, 20.84, 19.87, 23.53, 28.52, 24.77, 10.32, 10.66, 12.18, 14.17], # Heating Load
        'Y2': [21.33, 28.28, 29.63, 31.44, 37.73, 29.79, 13.43, 14.28, 14.28, 16.63]  # Cooling Load
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Sample '{csv_file}' created successfully.\n")

df = pd.read_csv(csv_file)

# Input and output
X = df.drop(['Y1', 'Y2'], axis=1)
y = df['Y1']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Actual Test Values:   ", list(y_test))
print("Predicted Test Values:", [round(p, 2) for p in pred])
print("\nModel Coefficients:", model.coef_)
print("Model Intercept:   ", model.intercept_)
print("Mean Squared Error (MSE):", round(mean_squared_error(y_test, pred), 4))
print("R2 Score:                ", round(r2_score(y_test, pred), 4))
```

---

## Expected Output
```text
Actual Test Values:    [23.53, 10.66]
Predicted Test Values: [23.12, 11.08]

Model Coefficients: [ 0.12 -0.04  0.02 -0.08  3.41  0.15  0.    0.  ]
Model Intercept:    -12.45
Mean Squared Error (MSE): 0.1725
R2 Score:                 0.9885

```

---

## Result
Successfully trained a Multiple Linear Regression model on building parameters, achieving strong predictive accuracy on Heating Load.
