import pandas as pd
import numpy as np
import joblib
import config
from tqdm import tqdm


def load_resources():
    # Load the trained model and the encoder
    model_path = config.DATA_PROCESSED / 'epm_model_optimized_turbo.pkl'
    encoder_path = config.DATA_PROCESSED / 'aircraft_encoder.pkl'
    data_path = config.DATA_PROCESSED / 'dataset_train_ready.csv'

    if not model_path.exists():
        print("Error: Model not found. Run train_model.py first.")
        return None, None, None

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    df = pd.read_csv(data_path)

    return model, encoder, df


def calculate_min_safe_velocity(mass_kg, altitude, wing_area, temperature):
    """
    Calculate the minimum speed to avoid stalling (falling).
    We take a 30% safety margin.
    """
    # 1. Get air density
    pressure = 101325 * (1 - 2.25577e-5 * altitude) ** 5.25588
    r_air = 287.05
    rho = pressure / (r_air * temperature)

    # 2. Lift coefficient (approximate)
    cl_max = 1.4

    # 3. Stall speed formula
    g = 9.81
    try:
        v_stall = np.sqrt((2 * mass_kg * g) / (rho * wing_area * cl_max))
    except ValueError:
        return 999  # Return high value if math error

    # Safety margin 1.3
    return v_stall * 1.3


def predict_optimal_conditions(model, row):
    """
    Test many speeds and altitudes to find the lowest EPM.
    """
    # Create the grid of values to test
    velocities = np.linspace(140, 260, 50)  # Test speeds from 140 to 260
    altitudes = np.linspace(8000, 13000, 50)  # Test altitudes from 8km to 13km
    V_grid, H_grid = np.meshgrid(velocities, altitudes)

    sim_df = pd.DataFrame({
        'velocity': V_grid.ravel(),
        'altitude': H_grid.ravel()
    })

    # Keep flight conditions constant (Mass, Wind, Plane type)
    cols_const = ['mass_kg', 'aircraft_type_encoded', 'wing_area', 'cd0', 'cd2', 'u_wind', 'v_wind']
    for c in cols_const:
        sim_df[c] = row[c]

    # Adjust temperature based on altitude (it gets colder higher up)
    delta_h = sim_df['altitude'] - row['altitude']
    sim_df['temperature'] = row['temperature'] - (0.0065 * delta_h)

    # --- SAFETY CHECK ---
    # Estimate density
    rho_est = 1.225 * (1 - 2.25577e-5 * sim_df['altitude']) ** 4.25

    # Check stall speed for every point
    v_stall = np.sqrt((2 * row['mass_kg'] * 9.81) / (rho_est * row['wing_area'] * 1.4))
    v_safe = v_stall * 1.3

    # Mark dangerous speeds
    sim_df['is_safe'] = sim_df['velocity'] > v_safe

    # Predict EPM using our AI
    features_order = [
        'velocity', 'altitude', 'mass_kg',
        'aircraft_type_encoded', 'wing_area', 'cd0', 'cd2',
        'temperature', 'u_wind', 'v_wind'
    ]
    sim_df['predicted_epm'] = model.predict(sim_df[features_order])

    # --- FILTER RESULTS ---
    # Only look at safe speeds
    safe_solutions = sim_df[sim_df['is_safe']]

    if safe_solutions.empty:
        # If nothing is safe, take the best available (should not happen often)
        best_idx = sim_df['predicted_epm'].idxmin()
    else:
        best_idx = safe_solutions['predicted_epm'].idxmin()

    return sim_df.loc[best_idx]


def main():
    model, encoder, df = load_resources()
    if model is None: return

    print("Starting optimization with safety checks...")

    # Pick one point per flight to avoid duplicates
    unique_flights = df.groupby('segment_id').first().reset_index()

    # Test on 100 random flights
    n_samples = min(100, len(unique_flights))
    sample_flights = unique_flights.sample(n=n_samples, random_state=42)

    results = []

    for _, row in tqdm(sample_flights.iterrows(), total=n_samples):
        opt = predict_optimal_conditions(model, row)

        real_epm = row['epm']
        pred_epm_opt = opt['predicted_epm']

        if real_epm > 0:
            saving_pct = (real_epm - pred_epm_opt) / real_epm * 100

            # Ignore extreme values
            if saving_pct < 60:
                results.append({
                    'aircraft': encoder.inverse_transform([int(row['aircraft_type_encoded'])])[0],
                    'mass_t': row['mass_kg'] / 1000,
                    'v_real': row['velocity'],
                    'v_opt': opt['velocity'],
                    'alt_real': row['altitude'],
                    'alt_opt': opt['altitude'],
                    'saving_pct': saving_pct
                })

    res_df = pd.DataFrame(results)



if __name__ == "__main__":
    main()