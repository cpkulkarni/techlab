
# ============================================================================
# PART 7: MODEL EVALUATION AND DETAILED ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Detailed Model Evaluation (Using Best Model: Random Forest)")
print("-" * 80)

# Use Random Forest as the best model for detailed analysis
best_model = results['Random Forest']['model']
y_pred_best = results['Random Forest']['predictions']
y_pred_proba_best = results['Random Forest']['probabilities']

# Detailed Classification Report
print("\n📊 Detailed Classification Report:")
print("-" * 80)
print(classification_report(y_test, y_pred_best, 
                          target_names=['No Maintenance', 'Maintenance Required'],
                          digits=4))

# Confusion Matrix
print("\n📊 Confusion Matrix:")
print("-" * 80)
cm = confusion_matrix(y_test, y_pred_best)
print(f"\n                    Predicted")
print(f"                No Maint.  Maint. Req.")
print(f"Actual No Maint.    {cm[0,0]:>6}      {cm[0,1]:>6}")
print(f"       Maint. Req.  {cm[1,0]:>6}      {cm[1,1]:>6}")

print(f"\nTrue Negatives (Correct no-maintenance):  {cm[0,0]}")
print(f"False Positives (False alarms):           {cm[0,1]}")
print(f"False Negatives (Missed failures):        {cm[1,0]}")
print(f"True Positives (Correct predictions):     {cm[1,1]}")

# Feature Importance for Random Forest
print("\n\n📊 Feature Importance Analysis (Random Forest):")
print("-" * 80)

feature_names = X_train.columns.tolist()
feature_importance_rf = pd.DataFrame({
    'Feature': feature_names,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance_rf.head(15).to_string(index=False))

# Save detailed results
feature_importance_rf.to_csv('feature_importance_random_forest.csv', index=False)
print("\n✓ Feature importance saved as 'feature_importance_random_forest.csv'")

# ============================================================================
# PART 8: PREDICTION FUNCTION FOR NEW DATA
# ============================================================================

print("\n\n" + "=" * 80)
print("STEP 8: Creating Prediction Function for New Motor Data")
print("-" * 80)

def predict_motor_maintenance(motor_data_dict, model, scaler, label_encoder):
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
    
    Returns:
    --------
    dict : Prediction results with probability
    """
    
    # Create DataFrame from input
    motor_df = pd.DataFrame([motor_data_dict])
    
    # Encode motor type
    motor_df['Motor_Type_Encoded'] = label_encoder.transform(motor_df['Motor_Type'])
    motor_df = motor_df.drop('Motor_Type', axis=1)
    
    # Create engineered features
    motor_df['Temp_Difference'] = motor_df['Process_Temperature_K'] - motor_df['Air_Temperature_K']
    motor_df['Apparent_Power'] = motor_df['Current_A'] * motor_df['Voltage_V'] / 1000
    motor_df['Active_Power'] = motor_df['Apparent_Power'] * motor_df['Power_Factor']
    motor_df['Speed_Torque_Ratio'] = motor_df['Rotational_Speed_RPM'] / (motor_df['Torque_Nm'] + 1)
    motor_df['Wear_Rate'] = motor_df['Tool_Wear_min'] / (motor_df['Operating_Hours'] + 1)
    
    # Ensure column order matches training data
    motor_df = motor_df[X_train.columns]
    
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

print("\n✓ Prediction function created successfully")
print()
