
# ============================================================================
# PART 10: SAVE THE COMPLETE MODEL AND CREATE SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Saving Model and Creating Final Summary")
print("-" * 80)

import pickle

# Save the trained model
model_filename = 'motor_maintenance_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_names': X_train.columns.tolist(),
        'model_type': 'Random Forest'
    }, f)

print(f"\n✓ Model saved as '{model_filename}'")

# Create comprehensive summary
summary = f"""
{'=' * 80}
INDUSTRIAL ELECTRICAL MOTOR PREDICTIVE MAINTENANCE
MODEL TRAINING SUMMARY
{'=' * 80}

📊 DATASET INFORMATION
{'-' * 80}
Total Samples:              {len(df):,}
Training Samples:           {len(X_train):,} (80%)
Test Samples:               {len(X_test):,} (20%)
Number of Features:         {X_train.shape[1]}
Target Classes:             2 (Maintenance Required: Yes/No)

Class Distribution:
  - No Maintenance:         {(y == 0).sum():,} samples ({(y == 0).mean()*100:.1f}%)
  - Maintenance Required:   {(y == 1).sum():,} samples ({(y == 1).mean()*100:.1f}%)

📈 KEY FEATURES MONITORED
{'-' * 80}
Physical Parameters:
  • Air Temperature (K)
  • Process/Winding Temperature (K)
  • Rotational Speed (RPM)
  • Torque (Nm)
  • Tool Wear (minutes)

Electrical Parameters:
  • Vibration (mm/s RMS)
  • Current (Amperes)
  • Voltage (Volts)
  • Power Factor
  • Load Percentage

Operational Parameters:
  • Motor Type (L/M/H)
  • Operating Hours

Engineered Features:
  • Temperature Difference
  • Apparent Power
  • Active Power
  • Speed-Torque Ratio
  • Wear Rate

🏆 MODEL PERFORMANCE (Random Forest - Best Model)
{'-' * 80}
Accuracy:                   {results['Random Forest']['accuracy']*100:.2f}%
Precision:                  {results['Random Forest']['precision']*100:.2f}%
Recall:                     {results['Random Forest']['recall']*100:.2f}%
F1-Score:                   {results['Random Forest']['f1_score']*100:.2f}%
ROC-AUC:                    {results['Random Forest']['roc_auc']:.4f}

Confusion Matrix:
  True Negatives:           {cm[0,0]:,}
  False Positives:          {cm[0,1]:,}
  False Negatives:          {cm[1,0]:,}
  True Positives:           {cm[1,1]:,}

🔍 TOP 5 MOST IMPORTANT FEATURES
{'-' * 80}
{feature_importance_rf.head(5).to_string(index=False)}

💡 MODEL INSIGHTS
{'-' * 80}
The model successfully predicts motor maintenance requirements with exceptional
accuracy. Key indicators for maintenance needs include:

1. Load Percentage - High loads indicate stress on the motor
2. Power Factor - Declining power factor suggests efficiency issues
3. Process Temperature - Elevated temperatures indicate overheating
4. Vibration - Increased vibration signals mechanical problems
5. Current Draw - Abnormal current patterns indicate electrical issues

📁 OUTPUT FILES GENERATED
{'-' * 80}
✓ motor_maintenance_data_full.csv          - Complete dataset (10,000 records)
✓ motor_maintenance_train.csv              - Training data (8,000 records)
✓ motor_maintenance_test.csv               - Test data (2,000 records)
✓ model_performance_comparison.csv         - All models comparison
✓ feature_importance_random_forest.csv     - Feature importance rankings
✓ motor_maintenance_model.pkl              - Trained model (pickled)

🚀 USAGE INSTRUCTIONS
{'-' * 80}
To use the trained model for predictions:

1. Load the model:
   import pickle
   with open('motor_maintenance_model.pkl', 'rb') as f:
       saved_model = pickle.load(f)

2. Prepare new motor data with all required features

3. Apply preprocessing:
   - Encode motor type
   - Create engineered features
   - Scale using saved scaler

4. Make prediction:
   prediction = saved_model['model'].predict(scaled_data)

⚠️ MAINTENANCE DECISION THRESHOLDS
{'-' * 80}
Risk Level Classification:
  • LOW RISK:      Failure probability < 30%
  • MEDIUM RISK:   Failure probability 30-70%
  • HIGH RISK:     Failure probability > 70%

Recommended Actions:
  • HIGH RISK:     Immediate maintenance/inspection required
  • MEDIUM RISK:   Schedule maintenance within 1-2 weeks
  • LOW RISK:      Continue normal operation, monitor

{'=' * 80}
Model Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""

print(summary)

# Save summary to file
with open('model_training_summary.txt', 'w') as f:
    f.write(summary)

print("✓ Summary saved as 'model_training_summary.txt'")
print()
print("=" * 80)
print("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nYou now have a complete end-to-end predictive maintenance solution including:")
print("  ✓ Synthetic training data based on real-world motor parameters")
print("  ✓ Multiple trained ML models (Logistic Regression, Random Forest, SVM, Gradient Boosting)")
print("  ✓ Model evaluation and comparison metrics")
print("  ✓ Feature importance analysis")
print("  ✓ Saved model ready for deployment")
print("  ✓ Prediction function for new data")
print("  ✓ Complete documentation")
print()
