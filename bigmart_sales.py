"""
BigMart Sales Prediction  (Analytics Vidhya hackathon)  -  metric: RMSE

Pipeline
  1. Clean data (fat-content labels, missing weight/size, zero visibility)
  2. Feature engineering (outlet age, item category, price per kg, relative visibility ...)
  3. Compare models with 5-fold cross-validation (RMSE)
  4. Blend the best models and write submission.csv

Put this file in the same folder as the 3 CSV files and run:  python bigmart_sales.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

TRAIN = "train_v9rqX0R.csv"
TEST = "test_AbJTz2l.csv"
SAMPLE = "sample_submission_8RXa3c6.csv"
OUT = "submission.csv"
SEED = 42


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ------------------------------------------------------------------ load
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
TARGET = "Item_Outlet_Sales"
n_train = len(train)
# combine so that cleaning/statistics are identical for train and test (no target used)
full = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)


# ------------------------------------------------------------------ cleaning
# 1) Fat content: 'LF', 'low fat', 'reg' are typos of 'Low Fat' / 'Regular'
full["Item_Fat_Content"] = full["Item_Fat_Content"].replace(
    {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"})

# 2) Item category from the ID prefix: FD=Food, DR=Drinks, NC=Non-Consumable
full["Item_Category"] = full["Item_Identifier"].str[:2]
# non-consumables (soap, tools...) cannot be "low fat"
full.loc[full["Item_Category"] == "NC", "Item_Fat_Content"] = "Non-Edible"

# 3) Missing Item_Weight: the same product has the same weight in every store
full["Item_Weight"] = full.groupby("Item_Identifier")["Item_Weight"].transform(
    lambda s: s.fillna(s.mean()))
full["Item_Weight"] = full["Item_Weight"].fillna(
    full.groupby("Item_Type")["Item_Weight"].transform("mean"))

# 4) Missing Outlet_Size: fill with the most common size of the same Outlet_Type
mode_size = full.groupby("Outlet_Type")["Outlet_Size"].agg(
    lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
full["Outlet_Size"] = full["Outlet_Size"].fillna(full["Outlet_Type"].map(mode_size))
full["Outlet_Size"] = full["Outlet_Size"].fillna("Small")

# 5) Visibility of 0 is impossible (an item on the shelf is visible): use item mean
full["Item_Visibility"] = full["Item_Visibility"].replace(0, np.nan)
full["Item_Visibility"] = full["Item_Visibility"].fillna(
    full.groupby("Item_Identifier")["Item_Visibility"].transform("mean"))


# ------------------------------------------------------------------ features
full["Outlet_Years"] = 2013 - full["Outlet_Establishment_Year"]
# visibility relative to the item's average visibility across all stores
full["Visibility_Ratio"] = full["Item_Visibility"] / full.groupby("Item_Identifier")["Item_Visibility"].transform("mean")
full["Price_per_Kg"] = full["Item_MRP"] / full["Item_Weight"]
# average price of the item's type, and how this item compares with it
full["MRP_vs_Type"] = full["Item_MRP"] / full.groupby("Item_Type")["Item_MRP"].transform("mean")
# MRP bands (the data has 4 clear price clusters)
full["MRP_Band"] = pd.cut(full["Item_MRP"], [0, 69, 136, 203, 400], labels=False)
# how many stores sell the item / how many items the store sells
full["Item_Count"] = full.groupby("Item_Identifier")["Item_Identifier"].transform("count")
full["Outlet_Count"] = full.groupby("Outlet_Identifier")["Outlet_Identifier"].transform("count")

cat_cols = ["Item_Fat_Content", "Item_Type", "Item_Category", "Outlet_Identifier",
            "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]
X_all = pd.get_dummies(full.drop(columns=["Item_Identifier", "Outlet_Establishment_Year"]),
                       columns=cat_cols, dtype=float)
X, X_test = X_all.iloc[:n_train].reset_index(drop=True), X_all.iloc[n_train:].reset_index(drop=True)
y = train[TARGET].values
print("Feature matrix:", X.shape, "| test:", X_test.shape)


# ------------------------------------------------------------------ models
# The metric is RMSE on raw sales, so models are trained on raw sales
# (tested: log / sqrt targets gave a worse CV RMSE of ~1110 / ~1088).
# The MRP x Outlet interaction matters most, so tree models are the main learners.
models = {
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
    "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=7, min_samples_leaf=20,
                                          max_features=0.5, n_jobs=-1, random_state=SEED),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                                  min_samples_leaf=30, subsample=0.7,
                                                  random_state=SEED),
}

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {}
print("\n5-fold CV RMSE :")
for name, m in models.items():
    p = cross_val_predict(m, X, y, cv=kf, n_jobs=1)
    oof[name] = p
    print(f"  {name:18s} {rmse(y, p):8.2f}")

# ------------------------------------------------------------------ blend
# non-negative blend weights chosen on the out-of-fold predictions
from scipy.optimize import nnls
names = list(models)
A = np.column_stack([oof[n] for n in names])
w, _ = nnls(A, y)
w = w / w.sum()
print("\nBlend weights:", dict(zip(names, w.round(3))))
print(f"Blend CV RMSE: {rmse(y, A @ w):.2f}")

# ------------------------------------------------------------------ final fit + submission
preds = np.zeros(len(X_test))
for wi, n in zip(w, names):
    if wi > 0:
        models[n].fit(X, y)
        preds += wi * models[n].predict(X_test)
preds = np.clip(preds, 0, None)

sub = pd.read_csv(SAMPLE)
sub["Item_Outlet_Sales"] = preds          # same row order as test.csv / sample file
sub.to_csv(OUT, index=False)
print(f"\nSaved {OUT}: {sub.shape}, NaNs={sub['Item_Outlet_Sales'].isna().sum()}")
print(sub.head())
