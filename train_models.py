import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to Heart.csv")
    args = parser.parse_args()

    # Create models directory
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data)
    
    # Save original data
    df.to_csv(os.path.join(model_dir, "original_data.csv"), index=False)

    # Handle missing values
    df = df.dropna(subset=["Ca", "Thal"])
    
    # Encode categoricals
    le_chest = LabelEncoder()
    le_thal = LabelEncoder()
    
    df["ChestPain"] = le_chest.fit_transform(df["ChestPain"])
    df["Thal"] = le_thal.fit_transform(df["Thal"])
    
    # Save encoders
    joblib.dump(le_chest, os.path.join(model_dir, "le_chest.pkl"))
    joblib.dump(le_thal, os.path.join(model_dir, "le_thal.pkl"))

    # Target and features
    target = "HD"
    feature_names = [c for c in df.columns if c != target]
    
    X = df[feature_names].values
    y = df[target].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler, feature names, and test data
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.pkl"))
    joblib.dump({"X": X_test_scaled, "y": y_test}, os.path.join(model_dir, "test_data.pkl"))

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    metrics = {}
    roc_data = {}
    shap_values_dict = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Save model
        joblib.dump(model, os.path.join(model_dir, f"{name.replace(' ', '_')}.pkl"))
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        cm = confusion_matrix(y_test, y_pred).tolist()
        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": cm
        }
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": roc_auc_score(y_test, y_prob)
        }
        
        # SHAP values
        print(f"Calculating SHAP values for {name}...")
        if name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test_scaled)
            if isinstance(shap_vals, list): 
                shap_vals = shap_vals[1]
            base_val = explainer.expected_value
            if isinstance(base_val, np.ndarray):
                base_val = float(base_val[1])
            else:
                base_val = float(base_val)
        else:
            background = shap.kmeans(X_train_scaled, 10)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            sample_size = min(50, len(X_test_scaled))
            shap_vals = explainer.shap_values(X_test_scaled[:sample_size])
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            base_val = float(explainer.expected_value[1])
            
        shap_values_dict[name] = {
            "values": shap_vals.tolist() if hasattr(shap_vals, "tolist") else shap_vals,
            "base_value": base_val
        }

    # Save metrics, ROC data, and SHAP values
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
        
    with open(os.path.join(model_dir, "roc_data.json"), "w") as f:
        json.dump(roc_data, f, indent=1)
        
    joblib.dump(shap_values_dict, os.path.join(model_dir, "shap_values.pkl"))

    print("\nTraining complete! All artifacts saved to the 'models/' folder.")
    print("You can now run your Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()