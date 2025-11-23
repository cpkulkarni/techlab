
# ============================================================================
# PART 2: DATA EXPLORATION AND VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Data Exploration and Statistical Analysis")
print("-" * 80)

# Basic statistics
print("\nDataset Statistics:")
print(df.describe())
print()

# Check for missing values
print("\nMissing Values Check:")
print(df.isnull().sum())
print("✓ No missing values found" if df.isnull().sum().sum() == 0 else "⚠ Missing values detected")
print()

# Distribution by motor type
print("\nDistribution by Motor Type:")
print(df.groupby('Motor_Type')['Maintenance_Required'].agg(['count', 'sum', 'mean']))
print()

# Feature correlation with target
print("\nFeature Correlation with Maintenance Required:")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('Maintenance_Required')

correlations = df[numerical_cols + ['Maintenance_Required']].corr()['Maintenance_Required'].sort_values(ascending=False)
print(correlations)
print()

# ============================================================================
# PART 3: DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Data Preprocessing and Feature Engineering")
print("-" * 80)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

# Separate features and target
X = df.drop(['Motor_ID', 'Maintenance_Required'], axis=1)
y = df['Maintenance_Required']

print(f"\nOriginal dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Encode categorical variable (Motor_Type)
le = LabelEncoder()
X['Motor_Type_Encoded'] = le.fit_transform(X['Motor_Type'])
X = X.drop('Motor_Type', axis=1)

print(f"\n✓ Encoded Motor_Type categorical variable")
print(f"  Motor types: {list(le.classes_)}")

# Create additional features (feature engineering)
print("\n✓ Creating engineered features...")

# Temperature difference (indicates thermal stress)
X['Temp_Difference'] = X['Process_Temperature_K'] - X['Air_Temperature_K']

# Power (approximation from current and voltage)
X['Apparent_Power'] = X['Current_A'] * X['Voltage_V'] / 1000  # in kW

# Actual power considering power factor
X['Active_Power'] = X['Apparent_Power'] * X['Power_Factor']

# Speed-Torque ratio (mechanical health indicator)
X['Speed_Torque_Ratio'] = X['Rotational_Speed_RPM'] / (X['Torque_Nm'] + 1)

# Wear rate (wear per operating hour)
X['Wear_Rate'] = X['Tool_Wear_min'] / (X['Operating_Hours'] + 1)

print(f"  Added 5 engineered features")
print(f"  Final feature count: {X.shape[1]}")

# ============================================================================
# PART 4: SPLIT DATA INTO TRAINING AND TEST SETS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Splitting Data into Training and Test Sets")
print("-" * 80)

# Split with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

# Save training and test data
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('motor_maintenance_train.csv', index=False)
test_data.to_csv('motor_maintenance_test.csv', index=False)

print("\n✓ Training data saved as 'motor_maintenance_train.csv'")
print("✓ Test data saved as 'motor_maintenance_test.csv'")

# Feature scaling
print("\n✓ Applying feature scaling (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("  Features normalized to zero mean and unit variance")
print()
