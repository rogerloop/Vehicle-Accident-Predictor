import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix

# --- CONFIG ---
DATA_FILE = 'data/AP7_Final_Training_Set.csv'
MODEL_FILE = 'models/accident_xgboost.pkl'

# Load model and data
print("Loading model and data...")
model = joblib.load(MODEL_FILE)
df = pd.read_csv(DATA_FILE)

# Prepare data (Same split as training for consistency)
X = df.drop(columns=['Y_ACCIDENT', 'timestamp_hora', 'station_id'])
y = df['Y_ACCIDENT']
timestamp = pd.to_datetime(df['timestamp_hora'])

# Use only the Test Set (last 20%)
split_index = int(len(df) * 0.8)
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]
time_test = timestamp.iloc[split_index:]

print(f"Validating with {len(X_test)} samples ({time_test.min()} to {time_test.max()})")

# --- PREDICTIONS ---
print("Generating predictions...")
probs = model.predict_proba(X_test)[:, 1]

# Optimal threshold (from previous output)
THRESHOLD = 0.231195

# Create results DataFrame
results = pd.DataFrame({
    'timestamp': time_test,
    'probabilidad': probs,
    'real_accident': y_test
})

# Add Risk Level
results['nivel_riesgo'] = pd.cut(
    results['probabilidad'], 
    bins=[-1, 0.10, THRESHOLD, 1.0], 
    labels=['Green', 'Yellow', 'Red']
)

# Does the model discriminate well? 
print("\n--- Discrimination Analysis ---")
avg_prob_accident = results[results['real_accident'] == 1]['probabilidad'].mean()
avg_prob_normal = results[results['real_accident'] == 0]['probabilidad'].mean()

print(f"Average probability when accident occurs: {avg_prob_accident:.4f}")
print(f"Average probability when no accident occurs: {avg_prob_normal:.4f}")
factor = avg_prob_accident / avg_prob_normal
print(f"The model assigns {factor:.1f} times more risk to accident situations.")

# Density Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=results[results['real_accident'] == 0], x='probabilidad', label='No Accident', fill=True, alpha=0.3)
sns.kdeplot(data=results[results['real_accident'] == 1], x='probabilidad', label='Real Accident', color='red', fill=True, alpha=0.3)
plt.axvline(THRESHOLD, color='black', linestyle='--', label=f'Threshold ({THRESHOLD:.2f})')
plt.title('Probability Distribution: Accidents vs Normality')
plt.xlabel('Predicted Risk Probability')
plt.legend()
plt.savefig('validation_density.png')
print("Saved: validation_density.png")

# Weekly Heat Map (Risk Heatmap)
# Aggregate risk by Day of Week and Hour
results['hour'] = results['timestamp'].dt.hour
results['dow'] = results['timestamp'].dt.dayofweek

heatmap_data = results.pivot_table(
    index='dow', 
    columns='hour', 
    values='probabilidad', 
    aggfunc='mean'
)

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='YlOrRd', yticklabels=days)
plt.title('Heat Map of Predicted Accident Risk by Day of Week and Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.savefig('validation_heatmap.png')
print("Saved: validation_heatmap.png")

# Segment Validation (Where does it see more danger?) 
# Recover segmento_pk from original dataframe
X_test_with_pk = X_test.copy()
X_test_with_pk['segmento_pk'] = df.iloc[split_index:]['segmento_pk']
X_test_with_pk['probabilidad'] = probs

risk_by_segment = X_test_with_pk.groupby('segmento_pk')['probabilidad'].mean().sort_values(ascending=False)

print("\n--- Top 5 Most Dangerous Segments (according to model) ---")
print(risk_by_segment.head(5))

plt.figure(figsize=(12, 4))
risk_by_segment.sort_index().plot(kind='bar', color='lightblue')
plt.title('Risk by Road Segment (Average Predicted Probability)')
plt.ylabel('Average Probability')
plt.xlabel('Road Segment Kilometer (segmento_pk)')
plt.savefig('validation_segments.png')
print("Saved: validation_segments.png")

print("\nValidation completed. Check the generated images.")