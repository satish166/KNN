# Medical Cost Prediction with K-Nearest Neighbors (KNN) Regression

Predict individual **medical insurance charges (in USD)** based on patient demographic and health factors using a **K-Nearest Neighbors (KNN) Regressor**.

- **Type of problem:** Regression (predicts a continuous number: healthcare charges in dollars)
- **Model:** `KNeighborsRegressor` with `StandardScaler` packaged inside a `Pipeline` (scikit-learn)
- **Dataset:** `insurance.csv` (Kaggle: *Medical Cost Personal Datasets*, 1,338 individuals)
- **Baseline $R^2$ ($k=5$):** `0.8038`
- **Tuned Model $R^2$:** **`0.8150`** (explains over 81.5% of variance in medical charges)
- **Mean Absolute Error (MAE):** **`$3,324.07`**

---

## 1. Files in this Folder

| File | Description |
|---|---|
| `regression.ipynb` | Jupyter Notebook with the complete end-to-end workflow: EDA, encoding, scaling, tuning, plots, and saving |
| `insurance.csv` | The raw Kaggle dataset (1,338 rows, 7 columns) |
| `final_knn_regressor.pkl` | Saved scikit-learn Pipeline (Scaler + tuned KNN Regressor) ready for inference |
| `README.md` | Documentation and guide (this file) |

---

## 2. The Dataset

`insurance.csv` is a widely used benchmark dataset from Kaggle containing demographic and health data for 1,338 insured individuals:

| Column | Type | Meaning | Used for Training? |
|---|---|---|---|
| `age` | Numeric (int) | Age of primary beneficiary (18 to 64) | Yes |
| `sex` | Categorical (str) | Gender (`female`, `male`) | Yes (encoded as `sex_male`) |
| `bmi` | Numeric (float) | Body Mass Index ($kg / m^2$, 15.96 to 53.13) | Yes |
| `children` | Numeric (int) | Number of children/dependents covered (0 to 5) | Yes |
| `smoker` | Categorical (str) | Smoking status (`yes`, `no`) | Yes (encoded as `smoker_yes`) |
| `region` | Categorical (str) | US geographic area (`northeast`, `northwest`, `southeast`, `southwest`) | Yes (encoded into 3 dummy columns) |
| `charges` | Numeric (float) | **Individual medical costs billed by health insurance ($)** | **Target ($y$)** |

### Target Summary (`charges`)
- **Minimum:** \$1,121.87
- **Median:** \$9,382.03
- **Mean:** \$13,270.42
- **Maximum:** \$63,770.43

---

## 3. Step-by-Step Workflow

### Step 1. Data Inspection & Cleaning
- Loaded `insurance.csv` (1,338 rows, 7 columns).
- Checked for null values (`0` missing values across all columns).
- Identified data types: 3 categorical string columns, 3 numerical feature columns, and 1 continuous target column.

### Step 2. Encoding Categorical Variables
- Converted `sex`, `smoker`, and `region` to numeric binary dummy variables using `pd.get_dummies(df, drop_first=True, dtype=int)`.
- `drop_first=True` avoids multicollinearity (the dummy variable trap).
- Total features after encoding: **8 numerical features**.

### Step 3. Train-Test Split
- Split into **80% training** (1,070 samples) and **20% testing** (268 samples).
- `random_state=42` ensures exact reproducibility.

### Step 4. Feature Scaling (`StandardScaler`)
> **Why scaling is essential for KNN:**
> KNN calculates the distance (Euclidean or Manhattan) between data points. In this dataset, `age` ranges from 18 to 64, while `bmi` ranges from 16 to 53, and dummy variables are 0 or 1. Scaling standardizes all features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$), ensuring all features contribute proportionally to distance measurements.

### Step 5. Baseline Model ($k=5$)
- Fit standard `KNeighborsRegressor(n_neighbors=5)`.
- Baseline Performance:
  - **$R^2$ Score:** `0.8038`
  - **MAE:** `\$3,494.75`
  - **RMSE:** `\$5,519.05`

### Step 6. Finding the Optimal $K$ via Cross-Validation
- Evaluated $k$ from 1 to 25 using 5-fold cross-validation.
- Generated a plot of CV $R^2$ vs. $K$ showing performance peaking around $k = 9$ to $10$.

### Step 7. Hyperparameter Tuning (`GridSearchCV`)
- Explored combinations of:
  - `n_neighbors`: $1$ to $25$
  - `weights`: `['uniform', 'distance']` (distance weighting gives closer neighbors more influence on the prediction)
  - `metric`: `['euclidean', 'manhattan']`
- **Best Hyperparameters:**
  ```python
  {'metric': 'euclidean', 'n_neighbors': 10, 'weights': 'distance'}
  ```

### Step 8. Test Set Evaluation & Visualization
- **Tuned Model Metrics:**
  - **Test $R^2$:** **`0.8150`**
  - **Test MAE:** **`$3,324.07`**
  - **Test RMSE:** **`$5,359.59`**
- Plotted **Actual vs. Predicted Charges** against the ideal diagonal line ($y = \hat{y}$), showing strong correlation across charges.

### Step 9. Packaging into Pipeline & Model Saving
- Encapsulated `StandardScaler` and `KNeighborsRegressor` into a single `scikit-learn Pipeline`.
- Saved the trained pipeline to `final_knn_regressor.pkl` via `pickle`.
- Loaded and tested the `.pkl` file to verify identical predictions on test data.

---

## 4. How to Run the Notebook

### Environment Setup
If running in a fresh environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebook
1. Navigate to the `knn_pred/regression` folder.
2. Launch Jupyter or open the file in VS Code:
   ```bash
   jupyter notebook regression.ipynb
   ```
3. Select your `.venv` kernel and click **Run All Cells**.

---

## 5. Using the Saved Model (`final_knn_regressor.pkl`)

You can load and use the model to predict insurance charges for new patients without retraining:

```python
import pickle
import pandas as pd

# 1. Load the trained pipeline
with open('final_knn_regressor.pkl', 'rb') as file:
    model = pickle.load(file)

# 2. Example new patient data (matching encoded column order)
new_patient = pd.DataFrame([{
    'age': 30,
    'bmi': 28.5,
    'children': 1,
    'sex_male': 1,              # 1 for male, 0 for female
    'smoker_yes': 0,            # 1 for smoker, 0 for non-smoker
    'region_northwest': 0,
    'region_southeast': 1,
    'region_southwest': 0
}])

# 3. Predict insurance charge
predicted_charge = model.predict(new_patient)
print(f"Predicted Medical Charge: ${predicted_charge[0]:,.2f}")
```

---

## 6. Key Concepts & Vocabulary

| Term | Meaning |
|---|---|
| **KNN Regressor** | Predicts the target value for a new query point by taking the (weighted) average of the target values of its $k$ nearest training neighbors. |
| **$R^2$ Score (Coefficient of Determination)** | Proportion of variance in the dependent variable explained by the model (1.0 is a perfect fit; 0 is equivalent to predicting the mean). |
| **MAE (Mean Absolute Error)** | Average magnitude of errors in the same units as the target (\$ in this case). Less sensitive to outliers than RMSE. |
| **RMSE (Root Mean Squared Error)** | Square root of the mean squared error. Penalizes large errors more heavily than MAE. |
| **Distance Weighting (`weights='distance'`)** | Assigns weights proportional to the inverse of the distance: closer neighbors contribute significantly more to the predicted value. |
| **Pipeline** | Chains data transformation (`StandardScaler`) with model estimation (`KNeighborsRegressor`), eliminating data leakage and simplifying inference. |
