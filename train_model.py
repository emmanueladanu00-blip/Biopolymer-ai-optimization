import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load Data
df = pd.read_csv('data/dataset.csv')

features = [
    'chitosan_percent', 'cellulose_percent', 'lignin_percent',
    'glycerol_percent', 'glutaraldehyde_percent'
]
targets = [
    'tensile_strength_mpa', 'elongation_percent',
    'water_absorption_percent', 'biodegradation_days'
]

X = df[features].values
y = df[targets].values

# 2. Train-Test Split (80% train, 20% test for academic validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale Features and Targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Save scalers for live predictions in the Web App
os.makedirs('models', exist_ok=True)
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')

# 4. Build the ANN Architecture (4 Hidden Layers - Objective iii)
model = Sequential([
    Input(shape=(len(features),)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    
    Dense(len(targets), activation='linear') # 4 continuous property outputs
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss='mse', metrics=['mae'])
model.summary()

# 5. Train Model with Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

print("\n🚀 Training Deep Neural Network...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# 6. Save Trained Model
model.save('models/ann_model.keras')
print("\n✅ Trained model saved to 'models/ann_model.keras'")

# 7. Evaluate on Test Set & Calculate R² Metrics (Required for Chapter 4)
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

print("\n" + "="*60)
print("🎯 THESIS PERFORMANCE METRICS (Objective iii)")
print("="*60)

for i, col in enumerate(targets):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    print(f"Target: {col:<26} | R² Score: {r2:.4f} | RMSE: {rmse:.4f}")

# 8. Generate Thesis Loss Curve Figure
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('ANN Training & Validation Convergence Curve')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data/loss_curve.png', dpi=300)
plt.close()

# 9. Generate Parity Plot (Actual vs Predicted for Tensile Strength)
plt.figure(figsize=(6, 6))
plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.7, color='teal')
plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
plt.title('Validation: Actual vs Predicted Tensile Strength (MPa)')
plt.xlabel('Measured Tensile Strength (MPa)')
plt.ylabel('AI Predicted Tensile Strength (MPa)')
plt.grid(True)
plt.tight_layout()
plt.savefig('data/parity_plot.png', dpi=300)
plt.close()

print("\n📊 High-resolution thesis figures saved in 'data/loss_curve.png' and 'data/parity_plot.png'")