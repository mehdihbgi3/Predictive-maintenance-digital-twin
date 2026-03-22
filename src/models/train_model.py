import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


def load_training_data(path="F:/predictive-maintenance-digital-twin/data/features/training_data.parquet"):
    print("Loading training data...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} samples with {len(df.columns)} columns")
    return df


def prepare_features(df, target='will_fail_30d'):
    
    exclude_cols = [
        'date', 'serial_number', 'model', 
        'days_to_failure', 'will_fail_30d', 'will_fail_60d', 'failed'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target].astype(int)
    
    X = X.fillna(0)
    
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {target}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols


class FailurePredictionModel:
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        print(f"Training {self.model_type} model...")
        print(f"Class weights: {weight_dict}")
        
        if self.model_type == 'xgboost':
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='auc'
            )
            
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X_train_scaled, y_train)
                
        elif self.model_type == 'lightgbm':
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.model.fit(X_train_scaled, y_train)
            
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
        
        print("Training complete!")
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(self, X_test, y_test):
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        return self.metrics
    
    def get_feature_importance(self, feature_names):
        if self.model_type in ['xgboost', 'lightgbm']:
            importance = self.model.feature_importances_
        elif self.model_type == 'logistic':
            importance = np.abs(self.model.coef_[0])
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to: {path}")
    
    @classmethod
    def load(cls, path):
        model_data = joblib.load(path)
        
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.metrics = model_data['metrics']
        instance.feature_names = model_data['feature_names']
        
        return instance


def train_and_evaluate():
    
    print("MACHINE LEARNING TRAINING PIPELINE")
    print("-" * 60)
    
    df = load_training_data()
    
    X, y, feature_names = prepare_features(df, target='will_fail_30d')
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(X_train):,} samples")
    print(f"Val: {len(X_val):,} samples")
    print(f"Test: {len(X_test):,} samples")
    
    results = {}
    
    for model_type in ['xgboost', 'lightgbm', 'logistic']:
        print(f"\nTraining {model_type.upper()}")
        
        model = FailurePredictionModel(model_type=model_type)
        model.train(X_train, y_train, X_val, y_val)
        model.feature_names = feature_names
        
        metrics = model.evaluate(X_test, y_test)
        results[model_type] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        model_path = f"F:/predictive-maintenance-digital-twin/models/{model_type}_model.joblib"
        model.save(model_path)
    
    print("\nMODEL COMPARISON SUMMARY")
    summary_df = pd.DataFrame(results).T
    print(summary_df.round(4).to_string())
    
    best_model = max(results, key=lambda x: results[x]['roc_auc'])
    print(f"\nBest Model: {best_model.upper()} (ROC AUC: {results[best_model]['roc_auc']:.4f})")
    
    print(f"\nTOP 15 FEATURES ({best_model.upper()})")
    best = FailurePredictionModel.load(f"F:/predictive-maintenance-digital-twin/models/{best_model}_model.joblib")
    importance = best.get_feature_importance(feature_names)
    print(importance.head(15).to_string(index=False))
    
    results_path = "F:/predictive-maintenance-digital-twin/models/training_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'best_model': best_model,
            'results': results
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = train_and_evaluate()
