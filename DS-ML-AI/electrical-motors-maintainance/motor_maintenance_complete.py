#!/usr/bin/env python3
"""
Industrial Electrical Motor Predictive Maintenance
End-to-End Machine Learning Pipeline

This script includes:
1. Synthetic data generation based on real-world motor parameters
2. Data preprocessing and feature engineering
3. Multiple ML model training and evaluation
4. Model deployment and prediction functions
5. Complete documentation

Author: Generated for Motor Predictive Maintenance
Date: 2025-10-28
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            roc_auc_score, f1_score, precision_score, recall_score)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("INDUSTRIAL ELECTRICAL MOTOR PREDICTIVE MAINTENANCE")
print("End-to-End Machine Learning Pipeline")
print("=" * 80)
print()


# ============================================================================
# FUNCTION 1: SYNTHETIC DATA GENERATION
# ============================================================================

def generate_motor_maintenance_data(n_samples=10000, failure_rate=0.15):
    """
    Generate realistic synthetic data for industrial motor predictive maintenance.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    failure_rate : float
        Proportion of failure cases (0.15 = 15% failures)

    Returns:
    --------
    pd.DataFrame : Synthetic motor data with maintenance labels

    Features based on real-world motor monitoring:
    - Temperature (K): Normal 318-348K (45-75°C), Critical >363K (>90°C)
    - Vibration (mm/s RMS): Normal <2.8, Warning 2.8-7.1, Critical >7.1
    - Current (A): Based on motor load
    - Voltage (V): Industrial standard 380-440V
    - Rotational Speed (RPM): Varies by motor type
    - Power Factor: Efficiency metric (0.7-0.95 for healthy motors)
    """

    data = []
    motor_types = ['Type_L', 'Type_M', 'Type_H']  # Low, Medium, High power

    for i in range(n_samples):
        is_failure = np.random.random() < failure_rate
        motor_type = np.random.choice(motor_types)
        operating_hours = np.random.uniform(0, 50000)
        tool_wear = operating_hours / 1000 + np.random.normal(0, 5)
        tool_wear = max(0, tool_wear)

        if is_failure:
            # Generate parameters indicating potential failure
            air_temp = np.random.uniform(25, 40)
            process_temp = np.random.uniform(85, 110)
            vibration = np.random.uniform(5.0, 15.0)
            current = np.random.uniform(85, 150)
            voltage = np.random.uniform(350, 390) if np.random.random() < 0.5 else np.random.uniform(450, 480)

            if motor_type == 'Type_L':
                speed = np.random.uniform(1200, 1600)
            elif motor_type == 'Type_M':
                speed = np.random.uniform(1300, 1700)
            else:
                speed = np.random.uniform(1400, 1800)

            torque = np.random.uniform(45, 80)
            power_factor = np.random.uniform(0.6, 0.75)
            load_percent = np.random.uniform(85, 110)
            target = 1

        else:
            # Generate parameters for healthy motor operation
            air_temp = np.random.uniform(18, 30)
            process_temp = np.random.uniform(45, 80)
            vibration = np.random.uniform(0.5, 4.5)
            current = np.random.uniform(40, 85)
            voltage = np.random.uniform(395, 430)

            if motor_type == 'Type_L':
                speed = np.random.uniform(1420, 1480)
            elif motor_type == 'Type_M':
                speed = np.random.uniform(1430, 1490)
            else:
                speed = np.random.uniform(1440, 1500)

            torque = np.random.uniform(20, 50)
            power_factor = np.random.uniform(0.80, 0.95)
            load_percent = np.random.uniform(40, 85)
            target = 0

        # Add realistic noise
        process_temp += np.random.normal(0, 2)
        vibration += np.random.normal(0, 0.3)
        current += np.random.normal(0, 3)

        record = {
            'Motor_ID': f'MOTOR_{i+1:05d}',
            'Motor_Type': motor_type,
            'Air_Temperature_K': air_temp + 273.15,
            'Process_Temperature_K': process_temp + 273.15,
            'Rotational_Speed_RPM': max(0, speed),
            'Torque_Nm': max(0, torque),
            'Tool_Wear_min': max(0, tool_wear),
            'Vibration_mm_s': max(0, vibration),
            'Current_A': max(0, current),
            'Voltage_V': voltage,
            'Power_Factor': np.clip(power_factor, 0.5, 1.0),
            'Load_Percent': np.clip(load_percent, 0, 120),
            'Operating_Hours': operating_hours,
            'Maintenance_Required': target
        }
        data.append(record)

    return pd.DataFrame(data)


# ============================================================================
# FUNCTION 2: FEATURE ENGINEERING
# ============================================================================

def engineer_features(X):
    """
    Create engineered features from raw motor parameters.

    Parameters:
    -----------
    X : pd.DataFrame
        Raw feature DataFrame

    Returns:
    --------
    pd.DataFrame : DataFrame with engineered features
    """
    X = X.copy()

    # Temperature difference (thermal stress indicator)
    X['Temp_Difference'] = X['Process_Temperature_K'] - X['Air_Temperature_K']

    # Power calculations
    X['Apparent_Power'] = X['Current_A'] * X['Voltage_V'] / 1000  # kW
    X['Active_Power'] = X['Apparent_Power'] * X['Power_Factor']

    # Mechanical health indicators
    X['Speed_Torque_Ratio'] = X['Rotational_Speed_RPM'] / (X['Torque_Nm'] + 1)
    X['Wear_Rate'] = X['Tool_Wear_min'] / (X['Operating_Hours'] + 1)

    return X


# ============================================================================
# FUNCTION 3: PREDICTION FOR NEW DATA
# ============================================================================

def predict_motor_maintenance(motor_data_dict, model, scaler, label_encoder, feature_names):
    """
    Predict maintenance requirement for a new motor.

    Parameters:
    -----------
    motor_data_dict : dict
        Dictionary containing motor parameters
    model : trained model
        The trained classification model
    scaler : StandardScaler
        Fitted scaler for feature normalization
    label_encoder : LabelEncoder
        Fitted encoder for motor type
    feature_names : list
        List of feature names in correct order

    Returns:
    --------
    dict : Prediction results with probability and risk level
    """

    # Create DataFrame from input
    motor_df = pd.DataFrame([motor_data_dict])

    # Encode motor type
    motor_df['Motor_Type_Encoded'] = label_encoder.transform(motor_df['Motor_Type'])
    motor_df = motor_df.drop('Motor_Type', axis=1)

    # Engineer features
    motor_df = engineer_features(motor_df)

    # Ensure column order matches training data
    motor_df = motor_df[feature_names]

    # Scale features
    motor_scaled = scaler.transform(motor_df)

    # Make prediction
    prediction = model.predict(motor_scaled)[0]
    probability = model.predict_proba(motor_scaled)[0]

    return {
        'maintenance_required': bool(prediction),
        'probability_no_maintenance': probability[0],
        'probability_maintenance': probability[1],
        'risk_level': 'HIGH' if probability[1] > 0.7 else 'MEDIUM' if probability[1] > 0.3 else 'LOW'
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function for the complete pipeline."""

    # Step 1: Generate synthetic data
    print("STEP 1: Generating Synthetic Motor Data")
    print("-" * 80)
    df = generate_motor_maintenance_data(n_samples=10000, failure_rate=0.15)
    print(f"✓ Generated {len(df)} records")
    print(f"✓ Failure rate: {df['Maintenance_Required'].mean()*100:.1f}%")
    df.to_csv('motor_maintenance_data_full.csv', index=False)
    print("✓ Saved to 'motor_maintenance_data_full.csv'\n")

    # Step 2: Data preprocessing
    print("STEP 2: Data Preprocessing")
    print("-" * 80)

    X = df.drop(['Motor_ID', 'Maintenance_Required'], axis=1)
    y = df['Maintenance_Required']

    # Encode categorical variable
    le = LabelEncoder()
    X['Motor_Type_Encoded'] = le.fit_transform(X['Motor_Type'])
    X = X.drop('Motor_Type', axis=1)

    # Engineer features
    X = engineer_features(X)
    print(f"✓ Created {X.shape[1]} features\n")

    # Step 3: Train-test split
    print("STEP 3: Splitting Data")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training: {len(X_train)} samples")
    print(f"✓ Testing: {len(X_test)} samples\n")

    # Save split data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv('motor_maintenance_train.csv', index=False)
    test_data.to_csv('motor_maintenance_test.csv', index=False)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Train models
    print("STEP 4: Training Models")
    print("-" * 80)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        print(f"✓ (F1: {results[name]['f1_score']:.4f})")

    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']
    print(f"\n✓ Best Model: {best_model_name}\n")

    # Step 5: Save model
    print("STEP 5: Saving Model")
    print("-" * 80)
    model_package = {
        'model': best_model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_names': X_train.columns.tolist(),
        'model_type': best_model_name
    }

    with open('motor_maintenance_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    print("✓ Model saved to 'motor_maintenance_model.pkl'\n")

    # Step 6: Demonstration
    print("STEP 6: Sample Predictions")
    print("-" * 80)

    # Example: Healthy motor
    healthy_motor = {
        'Motor_Type': 'Type_M',
        'Air_Temperature_K': 298.15,
        'Process_Temperature_K': 328.15,
        'Rotational_Speed_RPM': 1450,
        'Torque_Nm': 35,
        'Tool_Wear_min': 15,
        'Vibration_mm_s': 2.0,
        'Current_A': 65,
        'Voltage_V': 415,
        'Power_Factor': 0.88,
        'Load_Percent': 60,
        'Operating_Hours': 15000
    }

    result = predict_motor_maintenance(
        healthy_motor, best_model, scaler, le, X_train.columns.tolist()
    )

    print("Healthy Motor Example:")
    print(f"  Maintenance Required: {'YES' if result['maintenance_required'] else 'NO'}")
    print(f"  Failure Probability: {result['probability_maintenance']*100:.2f}%")
    print(f"  Risk Level: {result['risk_level']}\n")

    print("=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
