
# Complete End-to-End Python Code for Industrial Electrical Motor Predictive Maintenance
# This code includes: data generation, preprocessing, model training, evaluation, and prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("INDUSTRIAL ELECTRICAL MOTOR PREDICTIVE MAINTENANCE")
print("End-to-End Machine Learning Pipeline")
print("=" * 80)
print()

# ============================================================================
# PART 1: SYNTHETIC DATA GENERATION BASED ON REAL-WORLD MOTOR PARAMETERS
# ============================================================================

print("STEP 1: Generating Synthetic Motor Data Based on Real-World Scenarios")
print("-" * 80)

def generate_motor_maintenance_data(n_samples=10000, failure_rate=0.15):
    """
    Generate realistic synthetic data for industrial motor predictive maintenance.
    
    Parameters based on real-world motor monitoring:
    - Temperature (°C): Normal 40-75°C, Warning 75-90°C, Critical >90°C
    - Vibration (mm/s RMS): Normal <2.8, Warning 2.8-7.1, Critical >7.1
    - Current (Amperes): Based on motor load
    - Voltage (Volts): Typically around 380-440V for industrial motors
    - Rotational Speed (RPM): Varies by motor type
    - Power Factor: Efficiency metric (0.7-0.95 for healthy motors)
    - Operating Hours: Cumulative runtime
    - Load (%): Percentage of rated load
    """
    
    data = []
    
    # Define motor types
    motor_types = ['Type_L', 'Type_M', 'Type_H']  # Low, Medium, High power
    
    for i in range(n_samples):
        # Randomly determine if this will be a failure case
        is_failure = np.random.random() < failure_rate
        
        # Motor type
        motor_type = np.random.choice(motor_types)
        
        # Operating hours (0 to 50000 hours)
        operating_hours = np.random.uniform(0, 50000)
        
        # Tool wear increases with operating hours
        tool_wear = operating_hours / 1000 + np.random.normal(0, 5)
        tool_wear = max(0, tool_wear)
        
        if is_failure:
            # Generate parameters indicating potential failure
            
            # Elevated temperature (overheating)
            air_temp = np.random.uniform(25, 40)  # Ambient temperature
            process_temp = np.random.uniform(85, 110)  # Process/winding temperature
            
            # High vibration (bearing issues, misalignment, imbalance)
            vibration = np.random.uniform(5.0, 15.0)
            
            # Current imbalance or overload
            current = np.random.uniform(85, 150)
            
            # Voltage issues
            voltage = np.random.uniform(350, 390) if np.random.random() < 0.5 else np.random.uniform(450, 480)
            
            # Speed variation
            if motor_type == 'Type_L':
                speed = np.random.uniform(1200, 1600)
            elif motor_type == 'Type_M':
                speed = np.random.uniform(1300, 1700)
            else:
                speed = np.random.uniform(1400, 1800)
            
            # Torque variation
            torque = np.random.uniform(45, 80)
            
            # Poor power factor
            power_factor = np.random.uniform(0.6, 0.75)
            
            # High load
            load_percent = np.random.uniform(85, 110)
            
            # Maintenance needed
            target = 1
            
        else:
            # Generate parameters for healthy motor operation
            
            # Normal temperature
            air_temp = np.random.uniform(18, 30)
            process_temp = np.random.uniform(45, 80)
            
            # Low vibration
            vibration = np.random.uniform(0.5, 4.5)
            
            # Normal current
            current = np.random.uniform(40, 85)
            
            # Stable voltage
            voltage = np.random.uniform(395, 430)
            
            # Normal speed
            if motor_type == 'Type_L':
                speed = np.random.uniform(1420, 1480)
            elif motor_type == 'Type_M':
                speed = np.random.uniform(1430, 1490)
            else:
                speed = np.random.uniform(1440, 1500)
            
            # Normal torque
            torque = np.random.uniform(20, 50)
            
            # Good power factor
            power_factor = np.random.uniform(0.80, 0.95)
            
            # Normal load
            load_percent = np.random.uniform(40, 85)
            
            # No maintenance needed
            target = 0
        
        # Add some noise to make data more realistic
        process_temp += np.random.normal(0, 2)
        vibration += np.random.normal(0, 0.3)
        current += np.random.normal(0, 3)
        
        # Create record
        record = {
            'Motor_ID': f'MOTOR_{i+1:05d}',
            'Motor_Type': motor_type,
            'Air_Temperature_K': air_temp + 273.15,  # Convert to Kelvin
            'Process_Temperature_K': process_temp + 273.15,  # Convert to Kelvin
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

# Generate dataset
print("Generating 10,000 motor operation records...")
df = generate_motor_maintenance_data(n_samples=10000, failure_rate=0.15)

print(f"✓ Dataset generated with {len(df)} records")
print(f"✓ Failure cases: {df['Maintenance_Required'].sum()} ({df['Maintenance_Required'].mean()*100:.1f}%)")
print(f"✓ Healthy cases: {(1-df['Maintenance_Required']).sum()} ({(1-df['Maintenance_Required'].mean())*100:.1f}%)")
print()

# Display sample data
print("Sample of generated data (first 5 records):")
print(df.head())
print()

# Save the full dataset
df.to_csv('motor_maintenance_data_full.csv', index=False)
print("✓ Full dataset saved as 'motor_maintenance_data_full.csv'")
print()
