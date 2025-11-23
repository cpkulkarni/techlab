import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Data from the provided JSON
features = ["Load_Percent", "Power_Factor", "Process_Temperature_K", "Vibration_mm_s", "Current_A", "Apparent_Power", "Torque_Nm", "Speed_Torque_Ratio", "Temp_Difference", "Voltage_V"]
importance = [0.237053, 0.221168, 0.176276, 0.115282, 0.101747, 0.060076, 0.036590, 0.015637, 0.013547, 0.007805]

# Create DataFrame and sort by importance (descending)
df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})
df = df.sort_values('Importance', ascending=True)  # ascending=True for horizontal bar chart (highest at top)

# Shorten feature names to meet 15 character limit
df['Feature_Short'] = df['Feature'].replace({
    'Process_Temperature_K': 'Process_Temp_K',
    'Speed_Torque_Ratio': 'Speed_Torque'
})

# Create horizontal bar chart
fig = go.Figure(go.Bar(
    x=df['Importance'],
    y=df['Feature_Short'],
    orientation='h',
    marker_color='#1FB8CD',  # Using primary brand color
    text=[f'{val:.3f}' for val in df['Importance']],  # Value labels
    textposition='outside',
    textfont=dict(size=12)
))

# Update layout
fig.update_layout(
    title='Top 10 Features for Motor Maintenance',
    xaxis_title='Importance',
    yaxis_title='Features',
    xaxis=dict(range=[0, 0.25])
)

# Update traces for better appearance
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('feature_importance_chart.png')
fig.write_image('feature_importance_chart.svg', format='svg')

fig.show()