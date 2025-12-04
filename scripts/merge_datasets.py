import pandas as pd
import numpy as np
import json
import config
from pathlib import Path


def load_aircraft_specs():
    # Load the json file with aircraft details
    path = config.DATA_RAW / 'aircraft_openap.json'

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'r') as f:
        data = json.load(f)

    # Convert the dictionary to a list for pandas
    rows = []
    for icao, specs in data.items():
        rows.append({
            'aircraft_type': icao,
            'mtow': specs['mtow'],  # Max weight
            'wing_area': specs['wing_area'],
            'cd0': specs['cd0'],  # Drag coeff 1
            'cd2': specs['cd2'],  # Drag coeff 2
            'engine_type': specs.get('engine_type', 'unknown')
        })

    return pd.DataFrame(rows)


def load_weather():
    # Load the big weather csv file
    path = config.DATA_RAW / 'weather_full.csv'

    if not path.exists():
        print(f"Warning: Weather file is missing: {path}")
        return None

    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Round coordinates to match the weather grid (every 0.25 degrees)
    df['lat_grid'] = df['lat'].round(2)
    df['lon_grid'] = df['lon'].round(2)
    df['hour_grid'] = df['datetime'].dt.floor('H')  # Round time to the hour

    # Remove duplicates just in case
    df = df.drop_duplicates(subset=['lat_grid', 'lon_grid', 'hour_grid'])

    return df


def calculate_physics_and_epm(df):
    # Calculate the Energy Per Meter (EPM) using physics
    # We use numpy because it's faster for large datasets

    # --- A. Mass ---
    # We don't know the real mass, so we estimate it
    # We assume it's between 60% and 90% of the max weight (MTOW)
    np.random.seed(42)
    flight_ids = df['segment_id'].unique()

    # Assign a random ratio for each flight
    mass_ratios = {fid: np.random.uniform(0.6, 0.9) for fid in flight_ids}

    df['mass_ratio'] = df['segment_id'].map(mass_ratios)
    df['mass_kg'] = df['mtow'] * df['mass_ratio']

    # --- B. Air Density (Rho) ---
    # Calculate pressure based on altitude
    pressure = 101325 * (1 - 2.25577e-5 * df['altitude']) ** 5.25588

    # Use weather temp if we have it, else use standard temp
    temp_k = df['temperature'] if 'temperature' in df.columns else 288.15

    # Formula for density
    r_air = config.R_AIR
    df['rho'] = pressure / (r_air * temp_k)

    # --- C. Aerodynamics ---
    # In cruise, Lift equals Weight
    weight_n = df['mass_kg'] * config.G
    lift = weight_n

    # Dynamic pressure
    q = 0.5 * df['rho'] * (df['velocity'] ** 2)

    # Lift Coefficient
    cl = lift / (q * df['wing_area'])

    # Drag Coefficient (Polar curve)
    cd = df['cd0'] + df['cd2'] * (cl ** 2)

    # Total Drag
    drag = q * df['wing_area'] * cd

    # --- D. EPM ---
    # Energy per meter is basically the Drag divided by efficiency
    df['epm'] = drag / config.ETA

    return df


def main():

    # 1. Load Flights
    fpath_flights = config.DATA_RAW / 'flights_adsb_full.csv'
    if not fpath_flights.exists():
        print(f"Error: {fpath_flights} is missing.")
        return

    df_flights = pd.read_csv(fpath_flights)
    df_flights['timestamp'] = pd.to_datetime(df_flights['timestamp'])

    # 2. Merge with Plane Specs
    df_specs = load_aircraft_specs()

    # Keep only flights where we know the plane type
    df_merged = pd.merge(df_flights, df_specs, on='aircraft_type', how='inner')

    # 3. Merge with Weather
    df_weather = load_weather()

    # Prepare flight coordinates for merging
    df_merged['lat_grid'] = (df_merged['latitude'] * 4).round() / 4
    df_merged['lon_grid'] = (df_merged['longitude'] * 4).round() / 4
    df_merged['hour_grid'] = df_merged['timestamp'].dt.round('H')

    # Merge weather data
    df_merged = pd.merge(
        df_merged,
        df_weather,
        on=['lat_grid', 'lon_grid', 'hour_grid'],
        how='left',
        suffixes=('', '_meteo')
    )

    # Remove rows where we didn't find weather data
    before_drop = len(df_merged)
    df_merged = df_merged.dropna(subset=['u_wind', 'temperature'])


    # 4. Do the physics math
    df_final = calculate_physics_and_epm(df_merged)

    # 5. Keep only useful columns for the AI model
    cols_to_keep = [
        'timestamp', 'latitude', 'longitude', 'altitude',
        'velocity', 'mass_kg', 'segment_id',
        'aircraft_type', 'wing_area', 'cd0', 'cd2',
        'temperature', 'u_wind', 'v_wind', 'wind_speed', 'humidity',
        'epm'  # This is what we want to predict
    ]

    final_cols = [c for c in cols_to_keep if c in df_final.columns]
    df_final = df_final[final_cols]

    # Save to csv
    output_path = config.DATA_PROCESSED / 'dataset_final.csv'
    df_final.to_csv(output_path, index=False)



if __name__ == "__main__":
    main()