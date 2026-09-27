import os
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from deap import base, creator, tools

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("🔄 Loading Trained Neural Network and Scalers...")
model = tf.keras.models.load_model('models/ann_model.keras')
scaler_X = joblib.load('models/scaler_X.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')

# 1. Setup DEAP Multi-Objective Fitness
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def generate_individual():
    r = np.random.dirichlet(np.ones(3))
    cs = float(r[0] * 100.0)
    ce = float(r[1] * 100.0)
    lg = float(100.0 - (cs + ce))
    gly = float(random.uniform(1.0, 5.0))
    glut = float(random.uniform(0.1, 1.5))
    return creator.Individual([cs, ce, lg, gly, glut])

toolbox.register("individual", generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fast Batch Evaluation: Evaluates entire batch at once
def evaluate_population(pop):
    X_raw = []
    costs = []
    
    for ind in pop:
        poly_sum = ind[0] + ind[1] + ind[2]
        if poly_sum <= 0:
            poly_sum = 1.0
        cs = (ind[0] / poly_sum) * 100.0
        ce = (ind[1] / poly_sum) * 100.0
        lg = 100.0 - (cs + ce)
        gly = float(np.clip(ind[3], 1.0, 5.0))
        glut = float(np.clip(ind[4], 0.1, 1.5))
        
        ind[0], ind[1], ind[2], ind[3], ind[4] = cs, ce, lg, gly, glut
        X_raw.append([cs, ce, lg, gly, glut])
        
        cost = (
            (cs / 100.0 * 25.0) +
            (ce / 100.0 * 2.0) +
            (lg / 100.0 * 1.5) +
            ((gly + glut) / 100.0 * 3.0)
        )
        costs.append(cost)

    X_arr = np.array(X_raw)
    X_scaled = scaler_X.transform(X_arr)
    preds_scaled = model(X_scaled, training=False).numpy()
    preds = scaler_y.inverse_transform(preds_scaled)

    for i, ind in enumerate(pop):
        tensile = float(preds[i, 0])
        elongation = float(preds[i, 1])
        water_abs = float(preds[i, 2])
        degrad = float(preds[i, 3])
        cost = float(costs[i])

        if tensile < 25.0:
            tensile -= 40.0 # Penalty for failing cement bag strength
            
        ind.fitness.values = (tensile, elongation, water_abs, degrad, cost)

toolbox.register("mate", tools.cxBlend, alpha=0.3)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2.0, indpb=0.3)
toolbox.register("select_parents", tools.selTournament, tournsize=2)
toolbox.register("select_survivors", tools.selNSGA2)

def run_optimization():
    POP_SIZE = 100
    NGEN = 200
    CXPB = 0.8
    MUTPB = 0.25

    print(f"\n🧬 Starting Fast NSGA-II Genetic Algorithm ({POP_SIZE} Pop, {NGEN} Generations)...")
    pop = toolbox.population(n=POP_SIZE)
    evaluate_population(pop)

    for gen in range(1, NGEN + 1):
        # 1. Select Parents
        offspring = toolbox.select_parents(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # 2. Crossover & Mutation
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        for ind in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(ind)
                del ind.fitness.values

        # 3. Clamp physical bounds
        for ind in offspring:
            ind[0] = max(5.0, ind[0])
            ind[1] = max(5.0, ind[1])
            ind[2] = max(0.0, ind[2])
            ind[3] = float(np.clip(ind[3], 1.0, 5.0))
            ind[4] = float(np.clip(ind[4], 0.1, 1.5))

        # 4. Evaluate in batch
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            evaluate_population(invalid_ind)

        # 5. NSGA-II Multi-Objective Environmental Selection
        pop = toolbox.select_survivors(pop + offspring, k=POP_SIZE)

        if gen % 50 == 0 or gen == NGEN:
            print(f" -> Generation {gen}/{NGEN} Complete")

    # Extract Pareto Optimal Front
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    results = []
    for ind in pareto_front:
        tensile, elongation, water_abs, degrad, cost = ind.fitness.values
        if tensile >= 25.0:
            results.append({
                'Chitosan_%': round(ind[0], 2),
                'Cellulose_%': round(ind[1], 2),
                'Lignin_%': round(ind[2], 2),
                'Glycerol_%': round(ind[3], 2),
                'Glutaraldehyde_%': round(ind[4], 2),
                'Tensile_Strength_MPa': round(tensile, 2),
                'Elongation_%': round(elongation, 2),
                'Water_Absorption_%': round(water_abs, 2),
                'Degradation_Days': round(degrad, 1),
                'Cost_per_kg_USD': round(cost, 2)
            })

    df_pareto = pd.DataFrame(results).drop_duplicates().sort_values(by='Tensile_Strength_MPa', ascending=False)
    os.makedirs('data', exist_ok=True)
    df_pareto.to_csv('data/optimal_formulations.csv', index=False)
    
    print("\n" + "="*85)
    print("🏆 TOP 3 OPTIMAL CANDIDATE FORMULATIONS FOUND BY AI (Objectives iii, iv, v)")
    print("="*85)
    print(df_pareto.head(3).to_string(index=False))

    # Plot Pareto Front Figure
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        df_pareto['Cost_per_kg_USD'],
        df_pareto['Tensile_Strength_MPa'],
        c=df_pareto['Water_Absorption_%'],
        cmap='viridis_r',
        s=90, edgecolors='black', alpha=0.85
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Water Absorption % (Lower is Better Barrier)', rotation=270, labelpad=15)
    plt.axhline(25.0, color='r', linestyle='--', label='Min Cement Standard (25 MPa)')
    plt.title('Multi-Objective Pareto Front: Tensile Strength vs Cost vs Moisture Barrier')
    plt.xlabel('Production Cost ($/kg)')
    plt.ylabel('Tensile Strength (MPa)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/pareto_front.png', dpi=300)
    plt.close()
    print("\n📈 Thesis Pareto Front Figure saved in 'data/pareto_front.png'")

if __name__ == "__main__":
    run_optimization()