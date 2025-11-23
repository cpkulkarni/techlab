import plotly.graph_objects as go
import json

# Data from the provided JSON
data = {
  "models": ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"],
  "accuracy": [100.00, 100.00, 100.00, 99.95],
  "precision": [100.00, 100.00, 100.00, 100.00],
  "recall": [100.00, 100.00, 100.00, 99.66],
  "f1_score": [100.00, 100.00, 100.00, 99.83]
}

# Abbreviate model names to fit 15 character limit
abbreviated_models = ["Log Regression", "Random Forest", "SVM", "Grad Boosting"]

# Create the grouped bar chart
fig = go.Figure()

# Add bars for each metric using brand colors
fig.add_trace(go.Bar(
    name='Accuracy',
    x=abbreviated_models,
    y=data['accuracy'],
    marker_color='#1FB8CD',  # Strong cyan
    cliponaxis=False
))

fig.add_trace(go.Bar(
    name='Precision',
    x=abbreviated_models,
    y=data['precision'],
    marker_color='#DB4545',  # Bright red
    cliponaxis=False
))

fig.add_trace(go.Bar(
    name='Recall',
    x=abbreviated_models,
    y=data['recall'],
    marker_color='#2E8B57',  # Sea green
    cliponaxis=False
))

fig.add_trace(go.Bar(
    name='F1-Score',
    x=abbreviated_models,
    y=data['f1_score'],
    marker_color='#5D878F',  # Cyan
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='Model Performance - Motor Maintenance',
    xaxis_title='Models',
    yaxis_title='Performance %',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Set y-axis range from 99% to 100% to show differences clearly
fig.update_yaxes(range=[99, 100])

# Save as both PNG and SVG
fig.write_image('model_performance.png')
fig.write_image('model_performance.svg', format='svg')