import numpy as np
import pandas as pd
import os

np.random.seed(42)
n_samples = 550

data = []

for _ in range(n_samples):
    r1, r2, r3 = np.random.dirichlet(np.ones(3))
    cs_percent = np.round(r1 * 100, 2)
    ce_percent = np.round(r2 * 100, 2)
    lg_percent = np.round(100.0 - (cs_percent + ce_percent), 2)
    
    glycerol_percent = np.round(np.random.uniform(1.0, 5.0), 2)
    glutaraldehyde_percent = np.round(np.random.uniform(0.1, 1.5), 2)

    ts_base = 15.0
    ts_ce = (ce_percent / 100.0) * 25.0
    ts_crosslink = glutaraldehyde_percent * 6.0
    ts_plasticizer = glycerol_percent * 2.0
    tensile_strength = ts_base + ts_ce + ts_crosslink - ts_plasticizer + np.random.normal(0, 1.5)
    tensile_strength = np.clip(tensile_strength, 10.0, 50.0)

    elongation = (glycerol_percent * 8.0) + (cs_percent * 0.15) - (glutaraldehyde_percent * 5.0) + np.random.normal(0, 2.0)
    elongation = np.clip(elongation, 5.0, 60.0)

    wa_base = 500.0 * (cs_percent / 100.0)
    wa_lignin_reduction = (lg_percent / 100.0) * 380.0
    wa_crosslink_reduction = glutaraldehyde_percent * 40.0
    water_absorption = wa_base - wa_lignin_reduction - wa_crosslink_reduction + np.random.normal(0, 15.0)
    water_absorption = np.clip(water_absorption, 50.0, 550.0)

    degrad_days = 16.0 + (lg_percent * 0.1) - (ce_percent * 0.05) + np.random.normal(0, 1.0)
    degrad_days = np.clip(degrad_days, 10.0, 30.0)

    cost_per_kg = (
        (cs_percent / 100.0 * 25.0) +
        (ce_percent / 100.0 * 2.0) +
        (lg_percent / 100.0 * 1.5) +
        ((glycerol_percent + glutaraldehyde_percent) / 100.0 * 3.0)
    )

    data.append([
        cs_percent, ce_percent, lg_percent,
        glycerol_percent, glutaraldehyde_percent,
        np.round(tensile_strength, 2),
        np.round(elongation, 2),
        np.round(water_absorption, 2),
        np.round(degrad_days, 1),
        np.round(cost_per_kg, 2)
    ])

columns = [
    'chitosan_percent', 'cellulose_percent', 'lignin_percent',
    'glycerol_percent', 'glutaraldehyde_percent',
    'tensile_strength_mpa', 'elongation_percent',
    'water_absorption_percent', 'biodegradation_days',
    'cost_per_kg'
]

df = pd.DataFrame(data, columns=columns)
os.makedirs('data', exist_ok=True)
df.to_csv('data/dataset.csv', index=False)
print(f" SUCCESS: Generated {len(df)} samples and saved to 'data/dataset.csv'!")
print(df.head())