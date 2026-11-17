
"""
Model Training, Evaluation, and Selection Script
Student Performance Prediction System

Trains three Machine Learning algorithms:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Calculates genuine performance metrics (MAE, MSE, RMSE, R² Score) on train & test sets,
selects the best model, serializes pipelines using Joblib, and outputs evaluation charts.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from utils.preprocessing import (
    clean_dataset,
    build_preprocessor,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


def ensure_directories(base_dir: str):
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return models_dir, plots_dir


def evaluate_model(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    """
    Computes genuine regression metrics on both training and test splits.
    """
    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Train Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)

    # Test Metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    return {
        'model_name': name,
        'train_metrics': {
            'mae': round(float(train_mae), 3),
            'mse': round(float(train_mse), 3),
            'rmse': round(float(train_rmse), 3),
            'r2': round(float(train_r2), 4)
        },
        'test_metrics': {
            'mae': round(float(test_mae), 3),
            'mse': round(float(test_mse), 3),
            'rmse': round(float(test_rmse), 3),
            'r2': round(float(test_r2), 4)
        }
    }


def plot_model_comparison(results: list, plots_dir: str) -> str:
    """
    Generates a high-resolution comparison chart of R² Score and RMSE across models.
    """
    models = [r['model_name'] for r in results]
    r2_scores = [r['test_metrics']['r2'] for r in results]
    rmse_scores = [r['test_metrics']['rmse'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # 1. R² Score Comparison
    colors_r2 = ['#3B82F6', '#10B981', '#6366F1']
    bars1 = ax1.bar(models, r2_scores, color=colors_r2, width=0.55, edgecolor='black', linewidth=0.5)
    ax1.set_title("Test R² Score Comparison (Higher is Better)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("R² Score", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.4f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. RMSE Comparison
    colors_rmse = ['#F59E0B', '#EF4444', '#EC4899']
    bars2 = ax2.bar(models, rmse_scores, color=colors_rmse, width=0.55, edgecolor='black', linewidth=0.5)
    ax2.set_title("Test RMSE Comparison (Lower is Better)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Root Mean Squared Error (Marks)", fontsize=11)
    ax2.set_ylim(0, max(rmse_scores) * 1.25)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(plots_dir, 'model_comparison.png')
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def plot_feature_importance(best_pipeline: Pipeline, plots_dir: str):
    """
    Extracts and plots feature importances or coefficients from the best model.
    """
    model = best_pipeline.named_steps['regressor']
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    feature_names = [
        'Study Hours', 'Attendance %', 'Previous Marks',
        'Assignments %', 'Sleep Hours', 'Extracurricular'
    ]

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = "Random Forest Feature Importances (Gini Impurity / Variance Reduction)"
        x_label = "Relative Importance"
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        title = "Linear Regression Feature Weights (|Coefficients|)"
        x_label = "Magnitude of Standardized Coefficient"
    else:
        return None

    indices = np.argsort(importances)
    ax.barh(range(len(indices)), importances[indices], color='#3B82F6', align='center', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    feat_path = os.path.join(plots_dir, 'feature_importance.png')
    fig.savefig(feat_path, dpi=150)
    plt.close(fig)
    return feat_path


def train_and_select_best_model(csv_path: str = None) -> dict:
    """
    Main training workflow:
    1. Loads dataset
    2. Cleans & validates
    3. Train-test split (80/20)
    4. Trains 3 candidate models
    5. Evaluates metrics
    6. Selects & saves the best model
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if csv_path is None:
        csv_path = os.path.join(base_dir, 'student_data.csv')

    models_dir, plots_dir = ensure_directories(base_dir)

    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Data
    print(f"[1/6] Loading data from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    print(f"      Total raw records: {len(df_raw)}")

    # 2. Clean & Preprocess
    print("[2/6] Cleaning dataset and validating features...")
    df_clean = clean_dataset(df_raw)
    print(f"      Valid records for training: {len(df_clean)}")

    X = df_clean[FEATURE_COLUMNS]
    y = df_clean[TARGET_COLUMN]

    # 3. Train-Test Split (80% Train, 20% Test)
    print("[3/6] Performing Train-Test split (80% Train, 20% Test, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"      Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 4. Define Candidate Models
    candidate_algorithms = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_leaf=3, random_state=42)
    }

    results = []
    trained_pipelines = {}

    print("[4/6] Training and evaluating candidate algorithms...")
    for name, regressor in candidate_algorithms.items():
        print(f"      Training {name}...")
        # Build unified pipeline: Preprocessing -> Regressor
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('regressor', regressor)
        ])

        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Save individual model
        filename = f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(pipeline, os.path.join(models_dir, filename))

        # Evaluate
        eval_result = evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)
        results.append(eval_result)

        test_m = eval_result['test_metrics']
        print(f"      -> {name} Results: R² = {test_m['r2']:.4f} | RMSE = {test_m['rmse']:.3f} | MAE = {test_m['mae']:.3f}")

    # 5. Select Best Model
    # Primary selection criteria: Highest Test R² score, tie-breaker: Lowest RMSE
    print("[5/6] Comparing candidates and selecting best-performing model...")
    results.sort(key=lambda x: (x['test_metrics']['r2'], -x['test_metrics']['rmse']), reverse=True)
    best_result = results[0]
    best_model_name = best_result['model_name']
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\n" + "*" * 60)
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(f"Test R² Score: {best_result['test_metrics']['r2']:.4f}")
    print(f"Test RMSE:     {best_result['test_metrics']['rmse']:.3f}")
    print(f"Test MAE:      {best_result['test_metrics']['mae']:.3f}")
    print("*" * 60 + "\n")

    # Save best model to models/best_model.pkl
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(best_pipeline, best_model_path)
    print(f"Saved primary production model to: {best_model_path}")

    # Save metrics metadata to JSON
    metadata = {
        'best_model': best_model_name,
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'comparison_results': results,
        'test_size': 0.20,
        'random_state': 42,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    metrics_file = os.path.join(models_dir, 'model_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved evaluation metrics to: {metrics_file}")

    # 6. Generate comparison plots
    print("[6/6] Generating model evaluation comparison charts...")
    plot_model_comparison(results, plots_dir)
    plot_feature_importance(best_pipeline, plots_dir)
    print("Model training and artifacts generation successfully finished!")

    return metadata


if __name__ == '__main__':
    train_and_select_best_model()
Displaying test_system.py.
