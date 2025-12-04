import pygrib
import csv
import math
import config

grib_file = config.DATA_RAW / "ERA5-DATA.grib"
grbs = pygrib.open(str(grib_file))

TARGETS = {
    "t": "temperature",
    "u": "u_wind",
    "v": "v_wind",
    "r": "humidity",
    "z": "geopotential"
}

data_store = {}

for grb in grbs:
    var_code = grb.shortName
    if var_code not in TARGETS:
        continue

    var_name = TARGETS[var_code]
    timestamp = grb.validDate.isoformat()
    values = grb.values
    lats, lons = grb.latlons()
    ny, nx = values.shape

    if timestamp not in data_store:
        data_store[timestamp] = {}

    for i in range(ny):
        for j in range(nx):
            key = (float(lats[i, j]), float(lons[i, j]))
            if key not in data_store[timestamp]:
                data_store[timestamp][key] = {}
            data_store[timestamp][key][var_name] = float(values[i, j])

output_file = config.DATA_RAW / "weather_full.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "datetime", "lat", "lon",
        "wind_speed", "wind_dir",
        "u_wind", "v_wind",
        "temperature",
        "humidity",
        "geopotential"
    ])

    for timestamp, grid in data_store.items():
        for (lat, lon), vars in grid.items():

            u = vars.get("u_wind")
            v = vars.get("v_wind")
            if u is not None and v is not None:
                wind_speed = math.sqrt(u*u + v*v)
                wind_dir = (math.degrees(math.atan2(u, v)) + 360) % 360
            else:
                wind_speed = None
                wind_dir = None

            writer.writerow([
                timestamp,
                lat, lon,
                wind_speed,
                wind_dir,
                vars.get("u_wind"),
                vars.get("v_wind"),
                vars.get("temperature"),
                vars.get("humidity"),
                vars.get("geopotential")
            ])