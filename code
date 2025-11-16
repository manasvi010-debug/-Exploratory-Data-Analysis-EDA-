# -Exploratory-Data-Analysis-EDA-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset Overview:")
print(df.head(10))
print("\nDataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
# Task 1: Generate Summary Statistics
print("="*80)
print("TASK 1: SUMMARY STATISTICS")
print("="*80)

numeric_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
summary_stats = df[numeric_cols].describe()

print("\nDescriptive Statistics (describe method):")
print(summary_stats)

print("\n\nAdditional Statistics:")
print(f"Mean:\n{df[numeric_cols].mean()}\n")
print(f"Median:\n{df[numeric_cols].median()}\n")
print(f"Standard Deviation:\n{df[numeric_cols].std()}\n")
print(f"Skewness:\n{df[numeric_cols].skew()}\n")
print(f"Kurtosis:\n{df[numeric_cols].kurtosis()}\n")

# Save summary stats to CSV
summary_stats.to_csv('summary_statistics.csv')
print("Summary statistics saved to 'summary_statistics.csv'")
import plotly.graph_objects as go

# Data for all four features
sepal_length = [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5.0, 5.5, 4.9, 4.4, 5.1, 5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0, 7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5, 5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.0, 6.5, 5.8, 6.5, 6.4, 6.7, 6.8, 6.7, 6.3, 6.5, 6.2, 5.9]

sepal_width = [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 3.5, 3.2, 3.1, 3.4, 3.4, 3.1, 3.0, 3.3, 3.2, 3.5, 3.1, 3.4, 3.2, 3.2, 3.2, 3.2, 2.8, 2.8, 3.2, 3.0, 2.8, 3.0, 2.8, 3.4, 3.1, 3.0, 3.1, 3.1, 3.1, 2.7, 3.2, 3.3, 3.0, 2.6, 3.0, 2.6, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8, 2.9, 3.7, 3.0, 2.9, 2.7, 2.7, 3.0, 3.4, 3.1, 2.3, 3.0, 2.5, 2.6, 3.0, 2.6, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8, 3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 3.3, 3.2, 2.3, 2.6, 3.0, 2.6, 3.3, 3.4, 2.9, 3.1, 2.3, 2.8, 2.8, 3.2, 2.8, 3.0, 2.8, 3.8, 2.8, 2.8, 2.6, 3.0, 3.4, 3.1, 3.0, 3.1, 3.1, 3.1, 2.7, 3.2, 3.3, 3.0, 2.5, 3.0, 3.4, 3.0]

petal_length = [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.4, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4.0, 4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4.0, 4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1, 6.0, 5.1, 5.9, 5.6, 5.8, 5.6, 5.9, 6.1, 6.3, 6.5, 5.8, 6.7, 6.4, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5, 5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.0, 6.5, 5.8, 6.5, 6.4, 6.7, 6.8, 6.7, 6.3, 6.5, 6.2, 5.9]

petal_width = [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0, 1.1, 1.0, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.0, 2.0, 1.8, 2.1, 1.8, 1.8, 1.8, 2.0, 2.0, 1.5, 1.0, 2.0, 1.9, 1.8, 1.8, 2.0, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6, 1.9, 2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0, 2.3, 1.8]

# Create figure with grouped histograms using different bin positions
fig = go.Figure()

# Use offset positions to show histograms side by side
fig.add_trace(go.Histogram(
    x=sepal_length,
    name='Sepal Length',
    marker_color='#1FB8CD',
    xbins=dict(start=0, end=8, size=0.5),
    offsetgroup=0
))

fig.add_trace(go.Histogram(
    x=sepal_width,
    name='Sepal Width',
    marker_color='#DB4545',
    xbins=dict(start=0, end=8, size=0.5),
    offsetgroup=1
))

fig.add_trace(go.Histogram(
    x=petal_length,
    name='Petal Length',
    marker_color='#2E8B57',
    xbins=dict(start=0, end=8, size=0.5),
    offsetgroup=2
))

fig.add_trace(go.Histogram(
    x=petal_width,
    name='Petal Width',
    marker_color='#5D878F',
    xbins=dict(start=0, end=8, size=0.5),
    offsetgroup=3
))

fig.update_layout(
    title='Iris Features Distribution',
    xaxis_title='Value (cm)',
    yaxis_title='Frequency',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('iris_histogram.png')
fig.write_image('iris_histogram.svg', format='svg')
