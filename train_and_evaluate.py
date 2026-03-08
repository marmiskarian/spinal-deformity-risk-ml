"""
MSD Risk prediction: train and evaluate a classifier for Low/Medium/High risk.
Uses local data: msd_risk_dataset.xlsx (repo root or data/).
Run: python train_and_evaluate.py
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
DATA_PATH = Path("msd_risk_dataset.xlsx")
if not DATA_PATH.exists():
    DATA_PATH = Path("data") / "msd_risk_dataset.xlsx"

TARGET = "MSD_Risk_Level"
# False = all features (demographics, backpack, posture). True = only Age, Height, Weight, Pain_Score, Gender.
USE_SIMPLE_FEATURES = False

SIMPLE_NUM_COLS = ["Age", "Height", "Weight", "Pain_Score"]
SIMPLE_CAT_COLS = ["Gender"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
# Tune hyperparameters (set to False for a single quick run with default params)
TUNE_HYPERPARAMETERS = True


def get_all_numeric_columns(df):
    """All numeric columns except target and categoricals (for non-simple mode)."""
    cat = ["Gender", "Backpack_Position"]
    return [
        c
        for c in df.columns
        if c != TARGET
        and c not in cat
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def main():
    # ---------------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------------
    print("Loading data from", DATA_PATH)
    df = pd.read_excel(DATA_PATH)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    if USE_SIMPLE_FEATURES:
        print("Mode: SIMPLE features only (Age, Height, Weight, Pain_Score, Gender)\n")
        NUM_COLS = SIMPLE_NUM_COLS
        CAT_COLS_USE = SIMPLE_CAT_COLS
    else:
        print("Mode: All features (demographics, backpack, posture)\n")
        NUM_COLS = get_all_numeric_columns(df)
        CAT_COLS_USE = ["Gender", "Backpack_Position"]
    feature_cols = NUM_COLS + CAT_COLS_USE
    X = df[feature_cols]
    y = df[TARGET]

    print("Features used for training:")
    for i, name in enumerate(feature_cols, 1):
        print(f"  {i:2}. {name}")
    print()

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

    # ---------------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_COLS_USE),
        ],
        remainder="drop",
    )
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # ---------------------------------------------------------------------------
    # Train model
    # ---------------------------------------------------------------------------
    base_rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",  # helps Medium class
    )
    if TUNE_HYPERPARAMETERS:
        param_dist = {
            "n_estimators": [150, 200, 300, 400],
            "max_depth": [12, 15, 20, 25, None],
            "min_samples_leaf": [5, 10, 15, 20],
            "max_features": ["sqrt", "log2", None],
        }
        print("Tuning hyperparameters (RandomizedSearchCV, 3-fold CV, 24 combinations)...")
        search = RandomizedSearchCV(
            base_rf,
            param_distributions=param_dist,
            n_iter=24,
            cv=3,
            scoring="accuracy",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train_enc, y_train)
        model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
        print(f"Best CV accuracy: {search.best_score_:.4f}\n")
    else:
        print("Training Random Forest classifier (default params)...")
        base_rf.set_params(n_estimators=200, max_depth=15, min_samples_leaf=10)
        model = base_rf
        model.fit(X_train_enc, y_train)

    # ---------------------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------------------
    y_pred = model.predict(X_test_enc)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print("Classification report (test set):")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (test set):")
    print("Rows = true, Columns = predicted")
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))
    print("Labels:", list(model.classes_))

    # Feature importances (names after one-hot encoding)
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_COLS_USE)
    all_names = NUM_COLS + list(cat_names)
    print("Encoded feature names (as seen by the model):")
    for i, name in enumerate(all_names, 1):
        print(f"  {i:2}. {name}")
    importances = pd.Series(model.feature_importances_, index=all_names).sort_values(
        ascending=False
    )
    print("\nTop 15 feature importances:")
    print(importances.head(15).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
