
# ============================================================================
# PART 6: ADVANCED MODEL WITH XGBOOST (Best for Predictive Maintenance)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Training XGBoost Model (Industry Standard for PdM)")
print("-" * 80)

# Install and import XGBoost
try:
    import xgboost as xgb
    print("✓ XGBoost library loaded")
except ImportError:
    print("⚠ XGBoost not available, skipping this section")
    xgb = None

if xgb:
    print("\nTraining XGBoost Classifier...")
    
    # Create XGBoost model with optimized parameters for imbalanced data
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()  # Handle imbalance
    )
    
    # Train the model
    xgb_model.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    print("✓ XGBoost model trained successfully")
    
    # Make predictions
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    print("\nXGBoost Model Performance:")
    print("-" * 40)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred_xgb):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 40)
    print(classification_report(y_test, y_pred_xgb, 
                              target_names=['No Maintenance', 'Maintenance Required']))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print("-" * 40)
    cm = confusion_matrix(y_test, y_pred_xgb)
    print(f"True Negatives:  {cm[0,0]:>5} | False Positives: {cm[0,1]:>5}")
    print(f"False Negatives: {cm[1,0]:>5} | True Positives:  {cm[1,1]:>5}")
    
    # Feature Importance
    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    feature_names = X_train.columns.tolist()
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv('feature_importance_xgboost.csv', index=False)
    print("\n✓ Feature importance saved as 'feature_importance_xgboost.csv'")
    
print()
