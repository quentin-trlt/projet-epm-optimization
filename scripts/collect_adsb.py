import pandas as pd
import gzip
import json
import tarfile
import requests
from pathlib import Path
import config
from tqdm import tqdm


class ADSBLolCollector:
    def __init__(self):
        self.github_repo = "adsblol/globe_history_2025"
        self.base_url = f"https://github.com/{self.github_repo}/releases/download"
        config.DATA_RAW.mkdir(parents=True, exist_ok=True)

        # Dates to collect data from
        self.dates = [
            "2025-03-24",
            "2025-06-24",
            "2025-11-24"
        ]

        # URLs for the split archive files
        self.download_urls = {
            "2025-03-24": [
                "https://github.com/adsblol/globe_history_2025/releases/download/v2025.03.24-planes-readsb-prod-0/v2025.03.24-planes-readsb-prod-0.tar.aa",
                "https://github.com/adsblol/globe_history_2025/releases/download/v2025.03.24-planes-readsb-prod-0/v2025.03.24-planes-readsb-prod-0.tar.ab"
            ],
            "2025-06-24": [
                "https://github.com/adsblol/globe_history_2025/releases/download/v2025.06.24-planes-readsb-prod-0/v2025.06.24-planes-readsb-prod-0.tar.aa",
                "https://github.com/adsblol/globe_history_2025/releases/download/v2025.06.24-planes-readsb-prod-0/v2025.06.24-planes-readsb-prod-0.tar.ab"
            ],
            "2025-11-24": [
                "https://github.com/adsblol/globe_history_2025/releases/download/v2025.11.24-planes-readsb-prod-0/v2025.11.24-planes-readsb-prod-0.tar.aa",
                "https://github.com/adsblol/globe_history_2025/releases/download/v2025.11.24-planes-readsb-prod-0/v2025.11.24-planes-readsb-prod-0.tar.ab"
            ]
        }

    def download_file(self, url, output_path):
        # Download a file and show progress
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

    def download_and_extract(self, date):
        # Handle downloading split parts and extracting them
        date_dir = config.DATA_RAW / f"adsb_{date}"
        date_dir.mkdir(parents=True, exist_ok=True)

        extract_dir = config.DATA_RAW / f"adsb_{date}_extracted"
        traces_dir = extract_dir / "traces"

        # Check if we already have the data
        if traces_dir.exists() and list(traces_dir.rglob("*.json")):
            return traces_dir

        urls = self.download_urls.get(date, [])
        if not urls:
            return None

        # Download each part of the archive
        tar_parts = []
        for url in urls:
            filename = url.split("/")[-1]
            output_path = date_dir / filename

            if not output_path.exists():
                self.download_file(url, output_path)

            tar_parts.append(output_path)

        # Merge parts into one big tar file
        combined_tar = date_dir / f"v{date.replace('-', '.')}-planes-readsb-prod-0.tar"
        if not combined_tar.exists():
            with open(combined_tar, 'wb') as outfile:
                for part in sorted(tar_parts):
                    with open(part, 'rb') as infile:
                        outfile.write(infile.read())

        # Extract the json files
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(combined_tar, 'r') as tar:
            tar.extractall(extract_dir)

        return traces_dir

    def parse_trace_file(self, json_file):
        # Read a compressed json file and extract points
        traces = []

        try:
            with gzip.open(json_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)

            # Basic flight info
            icao = data.get('icao', '').lower()
            callsign = data.get('r', '').strip()
            aircraft_type = data.get('t', '')
            base_timestamp = data.get('timestamp', 0)

            trace_array = data.get('trace', [])

            # Loop through each point in the trace
            for point in trace_array:
                if len(point) < 3:
                    continue

                timestamp_offset = point[0]
                lat = point[1]
                lon = point[2]

                if lat is None or lon is None:
                    continue

                # Get data fields if they exist (format specific to adsb.lol)
                altitude = point[3] if len(point) > 3 else None
                ground_speed = point[4] if len(point) > 4 else None
                track = point[5] if len(point) > 5 else None
                vertical_rate = point[7] if len(point) > 7 else None
                alt_geom = point[10] if len(point) > 10 else None

                # Prefer geometric altitude if available
                final_altitude = alt_geom if alt_geom is not None else altitude
                absolute_timestamp = base_timestamp + timestamp_offset

                traces.append({
                    'icao24': icao,
                    'callsign': callsign,
                    'aircraft_type': aircraft_type,
                    'timestamp': absolute_timestamp,
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': final_altitude,
                    'ground_speed': ground_speed,
                    'track': track,
                    'vertical_rate': vertical_rate,
                })

        except Exception:
            pass

        return traces

    def collect_day(self, date, max_files=10000, max_aircraft=500):
        # Process one full day of data
        traces_dir = self.download_and_extract(date)

        if traces_dir is None or not traces_dir.exists():
            return None

        json_files = list(traces_dir.rglob("*.json"))

        # Limit files to save time
        if len(json_files) > max_files:
            json_files = json_files[:max_files]

        all_traces = []

        for json_file in tqdm(json_files, desc="Parsing"):
            traces = self.parse_trace_file(json_file)
            all_traces.extend(traces)

        if not all_traces:
            return None

        df = pd.DataFrame(all_traces)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Filter invalid or useless data
        df = self._filter_data(df)

        if df.empty:
            return None

        # Rebuild flight paths
        df = self._reconstruct_trajectories(df)
        df = self._segment_trajectories(df)

        # Limit aircraft count to save memory
        if df['aircraft_id'].nunique() > max_aircraft:
            selected_aircraft = df['aircraft_id'].unique()[:max_aircraft]
            df = df[df['aircraft_id'].isin(selected_aircraft)]

        df['date'] = date
        return df

    def collect_all_days(self, max_files=10000, max_aircraft=500):
        # Run collection for all configured dates
        all_dfs = []

        for date in self.dates:
            df = self.collect_day(date, max_files=max_files, max_aircraft=max_aircraft)
            if df is not None:
                all_dfs.append(df)

        if not all_dfs:
            return None

        df_full = pd.concat(all_dfs, ignore_index=True)

        output_file = config.DATA_RAW / 'flights_adsb_full.csv'
        df_full.to_csv(output_file, index=False)

        return df_full

    def _filter_data(self, df):
        # Keep only data within our geographic box and altitude limits
        lat_min, lon_min, lat_max, lon_max = config.OPENSKY_BBOX

        df = df.copy()

        # Geographic bounds
        df = df[
            (df['latitude'] >= lat_min) &
            (df['latitude'] <= lat_max) &
            (df['longitude'] >= lon_min) &
            (df['longitude'] <= lon_max)
            ]

        # Altitude bounds (convert feet to meters first)
        df = df[df['altitude'].notna()].copy()
        df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')
        df = df[df['altitude'].notna()].copy()
        df['altitude'] = df['altitude'] * 0.3048

        df = df[
            (df['altitude'] >= config.OPENSKY_ALT_MIN) &
            (df['altitude'] <= config.OPENSKY_ALT_MAX)
            ]

        # Speed bounds (convert knots to m/s)
        if 'ground_speed' in df.columns:
            df = df[df['ground_speed'].notna()].copy()
            df['ground_speed'] = pd.to_numeric(df['ground_speed'], errors='coerce')
            df['ground_speed'] = df['ground_speed'] * 0.514444
            df = df[df['ground_speed'] > config.VELOCITY_MIN]

        # Keep only stable flight (remove climb/descent)
        if 'vertical_rate' in df.columns:
            df['vertical_rate'] = pd.to_numeric(df['vertical_rate'], errors='coerce')
            # Convert fpm to m/s
            df.loc[df['vertical_rate'].notna(), 'vertical_rate'] = \
                df.loc[df['vertical_rate'].notna(), 'vertical_rate'] * 0.00508

            df = df[(df['vertical_rate'].isna()) | (df['vertical_rate'].abs() < 1.5)]

        # Rename columns for consistency
        df = df.rename(columns={
            'icao24': 'aircraft_id',
            'ground_speed': 'velocity'
        })

        return df

    def _reconstruct_trajectories(self, df):
        # Sort points by time and remove duplicates
        df = df.sort_values(['aircraft_id', 'timestamp']).reset_index(drop=True)
        df = df.drop_duplicates(subset=['aircraft_id', 'timestamp'], keep='first')

        # Give a unique ID to each trajectory
        df['trajectory_id'] = df.groupby('aircraft_id').ngroup()

        return df

    def _segment_trajectories(self, df, segment_duration=60):
        # Cut flights into small segments (e.g. 1 minute)
        segments = []

        for aircraft_id, group in tqdm(df.groupby('aircraft_id'), desc="Segmenting"):
            group = group.sort_values('timestamp').reset_index(drop=True)

            if len(group) < 2:
                group['segment_id'] = f"{aircraft_id}_0"
                segments.append(group)
                continue

            segment_id = 0
            segment_start_idx = 0

            for idx in range(1, len(group)):
                time_diff = (group.iloc[idx]['timestamp'] -
                             group.iloc[segment_start_idx]['timestamp']).total_seconds()

                # If time gap or max duration reached, start new segment
                if time_diff >= segment_duration or idx == len(group) - 1:
                    end_idx = idx + 1 if idx == len(group) - 1 else idx

                    segment_df = group.iloc[segment_start_idx:end_idx].copy()
                    segment_df['segment_id'] = f"{aircraft_id}_{segment_id}"
                    segments.append(segment_df)

                    segment_id += 1
                    segment_start_idx = idx

        if segments:
            return pd.concat(segments, ignore_index=True)

        return df


if __name__ == "__main__":
    collector = ADSBLolCollector()
    df = collector.collect_all_days(max_files=10000, max_aircraft=500)