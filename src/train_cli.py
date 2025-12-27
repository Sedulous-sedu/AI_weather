import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse

def main(path, model="logreg"):
    df = pd.read_csv(path)
    required = ['route','stop_distance_km','day_of_week','time_of_day',
                'traffic_condition','weather','special_event','temperature_celsius','arrival_status']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    X = df[['route','stop_distance_km','day_of_week','time_of_day',
            'traffic_condition','weather','special_event','temperature_celsius']].copy()
    y = df['arrival_status'].astype(str)
    # Basic cleaning
    X['special_event'] = X['special_event'].fillna('None')
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    categorical_features = ['route','day_of_week','traffic_condition','weather','special_event']
    numerical_features = ['stop_distance_km','time_of_day','temperature_celsius']
    pre = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])
    if model == "tree":
        clf = DecisionTreeClassifier(random_state=42, max_depth=10)
    else:
        clf = LogisticRegression(random_state=42, max_iter=1000)
    pipe = Pipeline([('preprocessor', pre), ('classifier', clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Accuracy: {acc:.3f}")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--model", choices=["logreg","tree"], default="logreg")
    args = ap.parse_args()
    main(args.csv_path, args.model)
