# -*- coding: utf-8 -*-
"""
Kaggle F1 进站预测 ✅ 三模型融合完整版
LightGBM + XGBoost + CatBoost | 零报错 | 可直接提交
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. 读取数据 ======================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv")

# ====================== 2. 数据预处理 ======================
X = train.drop(["id", "PitNextLap"], axis=1)
y = train["PitNextLap"]
X_test = test.drop(["id"], axis=1)

# 缺失值填充
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col].fillna(X[col].median(), inplace=True)
        X_test[col].fillna(X[col].median(), inplace=True)
    else:
        X[col].fillna(X[col].mode()[0], inplace=True)
        X_test[col].fillna(X[col].mode()[0], inplace=True)

# 类别特征编码（解决所有标签报错）
cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
for c in cat_cols:
    le = LabelEncoder()
    all_vals = pd.concat([X[c].astype(str), X_test[c].astype(str)])
    le.fit(all_vals)
    X[c] = le.transform(X[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))

# 划分训练集 / 验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====================== 模型 1：LightGBM ======================
print("\n===== 训练 LightGBM =====")
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

model_lgb = lgb.train(
    lgb_params, lgb_train, 1000, valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# ====================== 模型 2：XGBoost（零报错版） ======================
print("\n===== 训练 XGBoost =====")
model_xgb = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    verbosity=0,
    use_label_encoder=False
)
model_xgb.fit(X_train, y_train)

# ====================== 模型 3：CatBoost ======================
print("\n===== 训练 CatBoost =====")
model_cat = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_cols,
    eval_metric="AUC",
    verbose=100,
    early_stopping_rounds=50,
    random_state=42
)
model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

# ====================== 模型预测 ======================
p_lgb_val = model_lgb.predict(X_val)
p_lgb_test = model_lgb.predict(X_test)

p_xgb_val = model_xgb.predict_proba(X_val)[:, 1]
p_xgb_test = model_xgb.predict_proba(X_test)[:, 1]

p_cat_val = model_cat.predict_proba(X_val)[:, 1]
p_cat_test = model_cat.predict_proba(X_test)[:, 1]

# ====================== 三模型融合 ======================
print("\n===== 模型融合 =====")
val_pred = 0.4 * p_lgb_val + 0.3 * p_xgb_val + 0.3 * p_cat_val
test_pred = 0.4 * p_lgb_test + 0.3 * p_xgb_test + 0.3 * p_cat_test

val_auc = roc_auc_score(y_val, val_pred)
print(f"✅ 融合验证集 AUC: {val_auc:.5f}")

# ====================== 生成提交文件 ======================
sub["PitNextLap"] = test_pred
sub.to_csv("submission_3model_final.csv", index=False)
print("\n🎉 全部完成！提交文件：submission_3model_final.csv")