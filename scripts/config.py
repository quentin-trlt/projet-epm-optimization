from pathlib import Path

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

# Create folders if they do not exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# --- Flight Filtering Settings ---
# Only Europe
OPENSKY_BBOX = (40, -10, 55, 15)

# Altitude filters to keep only cruise phase
OPENSKY_ALT_MIN = 8000
OPENSKY_ALT_MAX = 13000

# Minimum speed to filter out errors (m/s)
VELOCITY_MIN = 50

# --- Physics Constants ---
G = 9.807       # Gravity (m/s^2)
R_AIR = 287.05  # Specific gas constant for dry air
ETA = 0.7       # Global propulsive efficiency factor