#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

#%%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

RMV = ['rainfall','id']
FEATURES = [c for c in train.columns if c not in RMV]
X = train[FEATURES]
y = train['rainfall']
X_test = test[FEATURES]

#%%
def objective(trial):
    params = {
        "device": "cuda",
        "verbosity": 0,
        "tree_method": "gpu_hist",
        "eval_metric": "auc",
        "objective": "binary:logistic",
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "alpha": trial.suggest_float("alpha", 0, 10),
        "lambda": trial.suggest_float("lambda", 0, 10),
        "n_estimators": 5000
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for train_idx, val_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        oof[val_idx] = model.predict_proba(X_valid)[:, 1]

    return roc_auc_score(y, oof)

#%%
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best AUC:", study.best_value)
print("Best parameters:", study.best_params)

#%%
best_params = study.best_params
best_params.update({
    "device": "cuda",
    "verbosity": 0,
    "tree_method": "gpu_hist",
    "eval_metric": "auc",
    "objective": "binary:logistic",
    "n_estimators": 10000
})

#%%
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
pred = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Training Fold {fold+1}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=100
    )

    oof[val_idx] = model.predict_proba(X_valid)[:, 1]
    pred += model.predict_proba(X_test)[:, 1] / 5

#%%
auc_score = roc_auc_score(y, oof)
print(f"Final CV AUC: {auc_score:.4f}")

#%%
submission = pd.DataFrame({
    "id": test["id"],
    "rainfall": pred
})
submission.to_csv("submission_optuna_xgb.csv", index=False)
print("Submission saved.")
