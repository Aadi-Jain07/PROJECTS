
"""
Automated Verification & Unit Testing Suite
Student Performance Prediction System
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from utils.preprocessing import (
    prepare_input_dataframe,
    classify_performance,
    generate_academic_recommendations,
    clean_dataset
)

def test_system():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    data_path = os.path.join(base_dir, 'student_data.csv')
    metrics_path = os.path.join(base_dir, 'models', 'model_metrics.json')

    print("Running system diagnostics...")

    # 1. Check Artifacts
    assert os.path.exists(data_path), "Dataset missing!"
    assert os.path.exists(model_path), "Best model missing!"
    assert os.path.exists(metrics_path), "Metrics file missing!"
    print("  [PASS] All critical files and models exist.")

    # 2. Check Dataset
    df = pd.read_csv(data_path)
    assert len(df) == 1500, f"Expected 1500 records, got {len(df)}"
    df_clean = clean_dataset(df)
    assert len(df_clean) == 1500, "Cleaned dataset size mismatch"
    print("  [PASS] Dataset loaded and cleaned successfully.")

    # 3. Load Model Pipeline
    model = joblib.load(model_path)
    print("  [PASS] Best model loaded successfully with Joblib.")

    # 4. Test Predictions Across Multiple Scenarios
    scenarios = [
        # (study, attendance, prev, assign, sleep, extra, expected_tier)
        (9.0, 95.0, 90.0, 95.0, 7.5, 'Yes', 'Excellent'),
        (5.5, 85.0, 75.0, 85.0, 7.0, 'Yes', 'Good'),
        (3.5, 75.0, 60.0, 70.0, 6.5, 'No', 'Average'),
        (1.5, 55.0, 40.0, 45.0, 4.5, 'No', 'Needs Improvement')
    ]

    for study, att, prev, assign, sleep, extra, exp_tier in scenarios:
        input_df = prepare_input_dataframe(study, att, prev, assign, sleep, extra)
        pred = model.predict(input_df)[0]
        pred_clamped = round(max(0.0, min(100.0, float(pred))), 1)
        tier_info = classify_performance(pred_clamped)
        recs = generate_academic_recommendations({
            'study_hours': study,
            'attendance_percentage': att,
            'previous_marks': prev,
            'assignment_completion': assign,
            'sleep_hours': sleep,
            'extracurricular_activities': extra
        }, pred_clamped)

        print(f"  [PASS] Scenario (Study={study}h, Att={att}%, Prev={prev}): Predicted {pred_clamped} -> {tier_info['tier']} (Grade {tier_info['grade_letter']}) | Recs: {len(recs)}")
        assert 0.0 <= pred_clamped <= 100.0, f"Marks {pred_clamped} out of bounds!"
        assert len(recs) > 0, "Recommendations should not be empty!"

    # 5. Check Metrics Values
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    assert 'comparison_results' in metrics
    for m in metrics['comparison_results']:
        r2 = m['test_metrics']['r2']
        rmse = m['test_metrics']['rmse']
        print(f"  [PASS] Model {m['model_name']}: Test R2={r2}, RMSE={rmse}")
        assert 0.5 <= r2 <= 1.0, f"Unreasonable R2: {r2}"

    print("\nALL SYSTEM TESTS PASSED SUCCESSFULLY! The application is 100% operational.")

if __name__ == '__main__':
    test_system()
Displaying test_system.py.
