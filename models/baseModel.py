#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from math import sqrt



def train_model(df, CURRENT_WEEK, model_out=None):
    """
    Train XGBoost weekly model
    df      : pandas DataFrame
    nbweek  : nombre de semaines d'entraînement
    return  : (model, {"mae":..,"rmse":..,"r2":..}, train_weeks, test_week)
    """
    nbweek = CURRENT_WEEK -6
    required_cols = [
        "date","airline","ch_code","num_code","dep_time","from",
        "time_taken","stop","arr_time","to","price","Class","dayofweek","week"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in the dataframe: {missing}")

    # Split weeks
    weeks_sorted = sorted(df["week"].dropna().astype(int).unique())
    if len(weeks_sorted) < nbweek :
        raise ValueError(f"Not enough weeks in the data (need: {nbweek+1}).")

    df_train_list = []
    df_test_list = []

    for w in weeks_sorted:
        df_week = df[df["week"] == w].copy()
        df_week = df_week.sample(frac=1, random_state=42)

        split_idx = int(len(df_week) * 0.8)

        df_train_list.append(df_week.iloc[:split_idx])
        df_test_list.append(df_week.iloc[split_idx:])

    df_train = pd.concat(df_train_list, ignore_index=True)
    df_test  = pd.concat(df_test_list, ignore_index=True)

    # Feature / target
    target = "price"
    feature_cols = [
        "airline","ch_code","from","to","Class","dayofweek",
        "num_code","dep_hour","arr_hour","time_taken_minutes","stops_n"
    ]

    X_train, y_train = df_train[feature_cols], df_train[target]
    X_test, y_test   = df_test[feature_cols], df_test[target]

    cat_cols = ["airline", "ch_code", "from", "to", "Class"]
    num_cols = ["dayofweek", "num_code", "dep_hour", "arr_hour", "time_taken_minutes", "stops_n"]
    for col in cat_cols:
        X_train[col] = X_train[col].fillna("MISSING").astype(str)
        X_test[col] = X_test[col].fillna("MISSING").astype(str)

    # Ensure numerical columns are proper numeric types
    for col in num_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float64), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
        tree_method="hist"
    )

    model = Pipeline([
        ("prep", preproc),
        ("reg", xgb),
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict & metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    # Save model if requested
    if model_out:
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_out)
    
    
    print(f"\nTraining weeks : 6 to{CURRENT_WEEK}")
    print(f"MAE   : {metrics['mae']:.2f}")
    print(f"RMSE  : {metrics['rmse']:.2f}")
    print(f"R²    : {metrics['r2']:.4f}")

    return model, metrics


