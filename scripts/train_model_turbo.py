import pandas as pd
import joblib
import config
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train():
    data_path = config.DATA_PROCESSED / 'dataset_train_ready.csv'
    if not data_path.exists():
        return

    df = pd.read_csv(data_path)

    # Limit dataset size to speed up training
    if len(df) > 500000:
        df = df.sample(n=500000, random_state=42)

    # Define input features and target
    features = [
        'velocity', 'altitude', 'mass_kg',
        'aircraft_type_encoded', 'wing_area', 'cd0', 'cd2',
        'temperature', 'u_wind', 'v_wind'
    ]
    target = 'epm'

    X = df[features]
    y = df[target]
    groups = df['segment_id']

    # Split data by flight segment to avoid leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Create Random Forest model with fixed parameters for speed
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,  # Limit depth to prevent overfitting
        n_jobs=-1,     # Use all CPU cores
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Save the trained model
    model_path = config.DATA_PROCESSED / 'epm_model_optimized_turbo.pkl'
    joblib.dump(model, model_path)

if __name__ == "__main__":
    train()