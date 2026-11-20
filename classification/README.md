# Wine Quality Prediction with K-Nearest Neighbors (KNN) Classification

Predict the **quality rating of wine** based on its physicochemical laboratory tests using a **K-Nearest Neighbors (KNN) Classifier**.

- **Type of problem:** Multi-class classification (predicts wine quality rating: 3, 4, 5, 6, 7, 8)
- **Algorithm:** `KNeighborsClassifier` with `StandardScaler` inside a `Pipeline` (scikit-learn)
- **Dataset:** `WineQT.csv` (1,143 wine samples, 11 chemical features)
- **Baseline Accuracy ($k=5$):** `57.64%`
- **Tuned Model Accuracy:** **`68.12%`** (using distance-weighted voting and optimal $k=24$)

---

## 1. Files in this Folder

| File | Description |
|---|---|
| `classification.ipynb` | Jupyter Notebook containing the end-to-end workflow: EDA, scaling, tuning, evaluation, and saving |
| `WineQT.csv` | The dataset (1,143 rows, 11 features + `quality` + `Id`) |
| `final_knn_model.pkl` | Saved scikit-learn Pipeline (Scaler + tuned KNN model) ready for inference |
| `README.md` | Documentation and usage guide (this file) |

---

## 2. The Dataset

`WineQT.csv` contains 1,143 samples and 13 columns:

| Feature | Description | Used for Training? |
|---|---|---|
| `Id` | Sample identifier | **No (dropped)** |
| `fixed acidity` | Most acids involved with wine (tartaric acid, $g/dm^3$) | Yes |
| `volatile acidity` | Amount of acetic acid in wine ($g/dm^3$) | Yes |
| `citric acid` | Adds 'freshness' and flavor to wines ($g/dm^3$) | Yes |
| `residual sugar` | Amount of sugar remaining after fermentation ($g/dm^3$) | Yes |
| `chlorides` | Amount of salt in the wine ($g/dm^3$) | Yes |
| `free sulfur dioxide` | Free form of $SO_2$ preventing microbial growth ($mg/dm^3$) | Yes |
| `total sulfur dioxide` | Free and bound forms of $SO_2$ ($mg/dm^3$) | Yes |
| `density` | Density of the wine ($g/cm^3$) | Yes |
| `pH` | Acidity/basicity on a scale from 0 (very acidic) to 14 | Yes |
| `sulphates` | Wine additive contributing to $SO_2$ levels ($g/dm^3$) | Yes |
| `alcohol` | Percent alcohol content (% vol.) | Yes |
| `quality` | **Target rating: score between 3 and 8** | **Target ($y$)** |

### Target Distribution

```text
Quality 3:    6 samples  (0.5%)
Quality 4:   33 samples  (2.9%)
Quality 5:  483 samples (42.3%)
Quality 6:  462 samples (40.4%)
Quality 7:  143 samples (12.5%)
Quality 8:   16 samples  (1.4%)
```
> Most wines fall into ratings 5 and 6 (~83% of the dataset), making it an imbalanced multi-class dataset.

---

## 3. Step-by-Step Workflow

### Step 1. Data Exploration & Inspection
- Check shape (`1143, 13`), data types (`float64` features, `int64` target), and missing values (`0` missing values).
- Verify duplicate rows (`0` duplicate rows).

### Step 2. Feature Selection & Target Definition
- Drop the `Id` column since it is an arbitrary index. Retaining it would introduce spurious patterns.
- Target $y$ = `df['quality']`.
- Features $X$ = 11 chemical properties.

### Step 3. Stratified Train-Test Split
- Split into **80% training** (914 samples) and **20% testing** (229 samples).
- `stratify=y` ensures each quality rating class is proportionally represented in both training and test sets.
- `random_state=42` guarantees reproducibility.

### Step 4. Feature Scaling (`StandardScaler`)
> **Why scaling is critical for KNN:**
> KNN calculates the distance (e.g. Euclidean distance) between data points. Features with large numerical ranges (e.g. `total sulfur dioxide` up to 289) would completely overpower features with small ranges (e.g. `chlorides` ~0.08) if left unscaled. `StandardScaler` standardizes each feature to have mean $\mu = 0$ and standard deviation $\sigma = 1$.

### Step 5. Baseline KNN Classifier
- Standard KNN with default $k=5$ and uniform weights.
- **Baseline Accuracy**: `57.64%`

### Step 6. Finding Optimal $K$ via Cross-Validation
- Evaluated $k$ from 1 to 30 using 5-fold cross-validation.
- A visualization plot is generated showing CV accuracy across different values of $k$.

### Step 7. Hyperparameter Tuning (`GridSearchCV`)
We tune three key hyperparameters:
- `n_neighbors`: candidate values $1$ through $30$.
- `weights`:
  - `'uniform'`: all neighbors have equal voting power.
  - `'distance'`: closer neighbors have a higher weight in the vote.
- `metric`:
  - `'euclidean'`: straight-line distance ($L_2$ norm).
  - `'manhattan'`: grid distance ($L_1$ norm).

**Optimal Parameters Found:**
```python
{'metric': 'euclidean', 'n_neighbors': 24, 'weights': 'distance'}
```

### Step 8. Evaluation of Tuned Model
- **Test Accuracy**: **`68.12%`** (an improvement of +10.48% over baseline).
- **Confusion Matrix**: Evaluated via `ConfusionMatrixDisplay` to visualize class-by-class predictions.
- **Classification Report**:
  - Class 5: Precision `0.73`, Recall `0.79`, F1 `0.76`
  - Class 6: Precision `0.64`, Recall `0.71`, F1 `0.67`
  - Class 7: Precision `0.64`, Recall `0.48`, F1 `0.55`

### Step 9. Packaging into Pipeline & Saving
- Created a `Pipeline([('scaler', StandardScaler()), ('knn', best_knn)])`.
- Packaging the scaler with the model prevents data leakage and ensures raw unscaled inputs are automatically normalized.
- Saved to `final_knn_model.pkl` with `pickle`.
- Verified that loading the `.pkl` file produces identical predictions to the trained model.

---

## 4. How to Run the Notebook

### Environment Setup
If setting up a new virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebook
1. Open VS Code or start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `classification.ipynb`.
3. Select the `.venv` kernel at the top right.
4. Click **Run All** to execute all cells.

---

## 5. Using the Saved Model (`final_knn_model.pkl`)

You can load and use the model to predict the quality of new wine samples without retraining:

```python
import pickle
import numpy as np
import pandas as pd

# 1. Load the saved pipeline
with open('final_knn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# 2. Example new wine measurements (11 features in order)
new_wine = pd.DataFrame([{
    'fixed acidity': 7.4,
    'volatile acidity': 0.70,
    'citric acid': 0.00,
    'residual sugar': 1.9,
    'chlorides': 0.076,
    'free sulfur dioxide': 11.0,
    'total sulfur dioxide': 34.0,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4
}])

# 3. Predict quality rating
predicted_quality = model.predict(new_wine)
print(f"Predicted Wine Quality: {predicted_quality[0]}")
```

---

## 6. Key Concepts & Vocabulary

| Term | Meaning |
|---|---|
| **K-Nearest Neighbors (KNN)** | An instance-based learning algorithm that classifies a new sample based on the majority vote of its $k$ closest neighbors in feature space. |
| **K ($k$-value)** | The number of nearest neighbors to consult. Small $k$ is sensitive to noise (overfitting); large $k$ smooths boundaries too much (underfitting). |
| **Euclidean Distance** | Straight-line distance between two points: $\sqrt{\sum (x_i - y_i)^2}$. |
| **Manhattan Distance** | Sum of absolute differences along each axis: $\sum \|x_i - y_i\|$. |
| **Feature Scaling** | Transforming features to a common scale so distances are not dominated by large-magnitude variables. |
| **Distance Weighting** | Giving closer neighbors more voting weight ($w = 1 / d$), reducing the influence of distant points. |
| **Pipeline** | A scikit-learn construct that chains preprocessing steps (like `StandardScaler`) together with an estimator. |
| **Stratified Split** | Train/test split that maintains identical class proportions across both sets. |

---

## 7. Model Insights & Limitations

1. **Extreme Classes:** Ratings 3 and 8 have very few samples (6 and 16 respectively). The model rarely predicts them because their neighborhood is easily dominated by classes 5 and 6.
2. **Binary Classification Alternative:** If mapped to binary classes (`good` wine $\ge 7$ vs `standard` wine $< 7$), accuracy rises to **~75%–82%**.
3. **Inference Speed:** KNN is a lazy learner (no training phase; computation happens at prediction time). For very large datasets, prediction can be slower compared to parametric models like Logistic Regression or Tree-based models.
