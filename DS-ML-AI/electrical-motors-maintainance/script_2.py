
# ============================================================================
# PART 5: MODEL TRAINING - MULTIPLE ALGORITHMS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Training Multiple Machine Learning Models")
print("-" * 80)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42)
}

# Train and evaluate each model
results = {}

print("\nTraining models...")
print("-" * 80)

for name, model in models.items():
    print(f"\n{name}:")
    print("  Training...", end=" ")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    print("✓")
    
    # Make predictions
    print("  Predicting...", end=" ")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    print("✓")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = None
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC: {roc_auc:.4f}")

# Create results comparison DataFrame
print("\n" + "=" * 80)
print("Model Performance Comparison")
print("-" * 80)

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] if results[m]['roc_auc'] else 0 for m in results.keys()]
})

results_df = results_df.sort_values('F1-Score', ascending=False)
print("\n", results_df.to_string(index=False))

# Save results
results_df.to_csv('model_performance_comparison.csv', index=False)
print("\n✓ Model comparison saved as 'model_performance_comparison.csv'")
print()
