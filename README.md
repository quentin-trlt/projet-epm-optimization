# Aircraft Energy Optimization through Machine Learning

## Overview

This project develops a machine learning pipeline to optimize aircraft energy consumption during flight. The goal is to predict optimal flight parameters (velocity and altitude) that minimize Energy Per Meter (EPM), potentially reducing fuel consumption by 5-15% compared to standard operations.

## Pipeline Architecture

The system follows a modular ETL architecture with seven main components:

### 1. Data Collection
- **ADS-B Flight Data** (`collect_adsb.py`): Retrieves historical flight trajectories from ADSB.lol, collecting lat/lon coordinates, altitudes, velocities, and aircraft identifiers
- **Meteorological Data** (`collect_era5.py`): Fetches weather conditions from ERA5 Copernicus including temperature, wind components (u/v), humidity, and pressure at flight positions
- **Aircraft Performance** (`collect_openap.py`): Gathers aircraft specifications from OpenAP database including wing area, drag coefficients (cd0, cd2), and maximum takeoff weight

### 2. Data Fusion
The `merge_datasets.py` module combines all data sources and performs physical calculations:
- Spatiotemporal alignment of weather data with flight trajectories (0.25° grid, hourly interpolation)
- Air density computation using barometric formula and ideal gas law
- Lift and drag force calculations based on cruise flight assumptions
- EPM computation incorporating aerodynamic efficiency

EPM formula integrates five components: induced power, aerodynamic drag, parasite losses, rotor profile power, and avionics consumption.

### 3. Preprocessing
`preprocessing.py` handles data cleaning, feature engineering, and aircraft type encoding for model input.

## Surrogate Model

Since real flight data doesn't contain optimal parameters, we train a surrogate model to predict EPM for any combination of flight conditions:

**Input Features:**
- Flight parameters: velocity, altitude, aircraft mass
- Aircraft characteristics: type, wing area, drag coefficients
- Environmental conditions: temperature, wind components

**Target:**
- EPM (Energy Per Meter)

**Model Architecture:**
Random Forest Regressor with 100 estimators, max depth 15. Training uses GroupShuffleSplit to prevent data leakage across flight segments.

The surrogate acts as a fast approximation of the complex physics calculations, enabling rapid evaluation of thousands of parameter combinations during optimization.

## Grid Search Optimization

For each set of flight conditions (origin, destination, aircraft type, weather), we perform a systematic grid search:

1. **Define Parameter Grid:**
   - Velocity range: typically 200-280 m/s (cruise speeds)
   - Altitude range: 8,000-12,000 m (commercial flight levels)
   - Grid resolution: ~10-20 samples per dimension

2. **Evaluation Process:**
   - For each (velocity, altitude) pair in the grid
   - Use surrogate model to predict EPM
   - Account for wind effects on ground speed
   - Calculate total energy for complete trajectory

3. **Selection:**
   - Identify parameter combination with minimum predicted EPM
   - This represents the optimal flight profile for given conditions

The grid search generates synthetic optimization scenarios since real flights are not operated at optimal efficiency. This approach creates the training data needed to learn optimal parameters across diverse conditions.

## Results

The trained model achieves reasonable predictive performance on test data, with R² scores indicating good generalization. The optimization notebook (`optimize_flight.py`) demonstrates the application of the trained model to find optimal parameters for specific flight scenarios.

## Technical Stack

- **Data Processing:** pandas, numpy
- **Aviation Libraries:** OpenAP, traffic
- **Machine Learning:** scikit-learn, joblib
- **Weather Data:** cdsapi, pygrib
- **Visualization:** matplotlib, seaborn

## References

1. OpenAP: Open Aircraft Performance Model
2. ERA5: Fifth generation ECMWF atmospheric reanalysis
3. ADSB.lol: Historical ADS-B flight tracking data
4. OpenSky Network research papers on trajectory-based fuel estimation
