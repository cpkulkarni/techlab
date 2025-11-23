
# ============================================================================
# PART 9: DEMONSTRATION WITH SAMPLE PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Sample Predictions on New Motor Data")
print("-" * 80)

# Example 1: Healthy motor
print("\n🔍 Example 1: Healthy Motor Operating Conditions")
print("-" * 80)
healthy_motor = {
    'Motor_Type': 'Type_M',
    'Air_Temperature_K': 298.15,  # 25°C
    'Process_Temperature_K': 328.15,  # 55°C
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

print("\nInput Parameters:")
for key, value in healthy_motor.items():
    print(f"  {key}: {value}")

result = predict_motor_maintenance(healthy_motor, best_model, scaler, le)

print("\n🎯 Prediction Results:")
print(f"  Maintenance Required: {'YES ⚠️' if result['maintenance_required'] else 'NO ✓'}")
print(f"  Probability of Failure: {result['probability_maintenance']*100:.2f}%")
print(f"  Risk Level: {result['risk_level']}")

# Example 2: Motor with warning signs
print("\n\n🔍 Example 2: Motor Showing Warning Signs")
print("-" * 80)
warning_motor = {
    'Motor_Type': 'Type_H',
    'Air_Temperature_K': 303.15,  # 30°C
    'Process_Temperature_K': 363.15,  # 90°C (High!)
    'Rotational_Speed_RPM': 1550,
    'Torque_Nm': 55,
    'Tool_Wear_min': 40,
    'Vibration_mm_s': 7.5,  # High vibration!
    'Current_A': 105,  # High current
    'Voltage_V': 375,  # Low voltage
    'Power_Factor': 0.70,  # Poor power factor
    'Load_Percent': 95,  # High load
    'Operating_Hours': 35000
}

print("\nInput Parameters:")
for key, value in warning_motor.items():
    print(f"  {key}: {value}")

result = predict_motor_maintenance(warning_motor, best_model, scaler, le)

print("\n🎯 Prediction Results:")
print(f"  Maintenance Required: {'YES ⚠️' if result['maintenance_required'] else 'NO ✓'}")
print(f"  Probability of Failure: {result['probability_maintenance']*100:.2f}%")
print(f"  Risk Level: {result['risk_level']}")

# Example 3: Critical motor condition
print("\n\n🔍 Example 3: Motor in Critical Condition")
print("-" * 80)
critical_motor = {
    'Motor_Type': 'Type_L',
    'Air_Temperature_K': 308.15,  # 35°C
    'Process_Temperature_K': 378.15,  # 105°C (Critical!)
    'Rotational_Speed_RPM': 1350,
    'Torque_Nm': 68,
    'Tool_Wear_min': 50,
    'Vibration_mm_s': 12.0,  # Very high vibration!
    'Current_A': 130,  # Very high current
    'Voltage_V': 365,  # Very low voltage
    'Power_Factor': 0.65,  # Very poor power factor
    'Load_Percent': 105,  # Overload
    'Operating_Hours': 42000
}

print("\nInput Parameters:")
for key, value in critical_motor.items():
    print(f"  {key}: {value}")

result = predict_motor_maintenance(critical_motor, best_model, scaler, le)

print("\n🎯 Prediction Results:")
print(f"  Maintenance Required: {'YES ⚠️' if result['maintenance_required'] else 'NO ✓'}")
print(f"  Probability of Failure: {result['probability_maintenance']*100:.2f}%")
print(f"  Risk Level: {result['risk_level']}")

print("\n\n" + "=" * 80)
print("✅ PREDICTIVE MAINTENANCE MODEL PIPELINE COMPLETE")
print("=" * 80)
print()
