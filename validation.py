import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score

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
df_test_full = df.iloc[split_index:].copy()

print(f"Validating with {len(X_test)} samples ({time_test.min()} to {time_test.max()})")

# --- PREDICTIONS ---
print("Generating predictions...")
probs = model.predict_proba(X_test)[:, 1] # Probabilidad de accidente

THRESHOLD = 0.5
y_pred = (probs > THRESHOLD).astype(int)
# Guardar probabilidades en el DataFrame para análisis
df_test_full['probabilidad'] = probs


# -- VISUALIZACIONES --
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 12))
fig.suptitle(f'Model Diagnosis (Threshold: {THRESHOLD})', fontsize=16)

# --- CURVA ROC + TABLA MÉTRICAS ---
ax1 = plt.subplot(2, 3, 1)
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

# Calcular métricas
acc = accuracy_score(y_test, y_pred)
# Recall Positivo (Sensibilidad): Cuántos accidentes reales detectamos
recall_pos = recall_score(y_test, y_pred, pos_label=1)
# Recall Negativo (Especificidad): Cuántos NO accidentes clasificamos bien
recall_neg = recall_score(y_test, y_pred, pos_label=0)

ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve & Performance')
ax1.legend(loc="lower right")

# Tabla de Métricas dentro del gráfico
metrics_text = (
    f"Accuracy: {acc:.1%}\n"
    f"AUC:      {roc_auc:.3f}\n"
    f"Recall (+): {recall_pos:.1%} (Accidentes)\n"
    f"Recall (-): {recall_neg:.1%} (Normalidad)"
)
ax1.text(0.5, 0.3, metrics_text, transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

# --- CONFUSION MATRIX ---
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
ax2.set_title('Matriz de Confusión')
ax2.set_xlabel('Predicción')
ax2.set_ylabel('Realidad')
ax2.set_xticklabels(['Normal', 'Accidente'])
ax2.set_yticklabels(['Normal', 'Accidente'])

# --- MAPA DE CALOR (HORA vs DÍA) ---
ax4 = plt.subplot(2, 3, 4)
df_test_full['hour'] = time_test.dt.hour
df_test_full['dow'] = time_test.dt.dayofweek
heatmap_data = df_test_full.pivot_table(index='dow', columns='hour', values='probabilidad', aggfunc='mean')
days = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
sns.heatmap(heatmap_data, cmap='YlOrRd', yticklabels=days, ax=ax4)
ax4.set_title('Mapa de Calor: Riesgo por Hora y Día')

# --- RIESGO POR TRAMO ---
ax5 = plt.subplot(2, 3, (5, 6)) # Ocupa 2 espacios
risk_by_segment = df_test_full.groupby('segmento_pk')['probabilidad'].mean()
risk_by_segment.plot(kind='bar', ax=ax5, color='skyblue', edgecolor='grey')
ax5.set_title('Riesgo Medio por Kilómetro (Segmento)')
ax5.set_ylabel('Probabilidad')

plt.tight_layout()
plt.savefig('validation_dashboard_complete2.png', dpi=300)
print("\n Diagnóstico completado. Imagen guardada: validation_dashboard_complete.png")

print("\nValidation completed. Check the generated images.")