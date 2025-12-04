"""
OpenAP Performance Data Collector
"""
import openap
from openap import Drag
import json
from pathlib import Path
import config

# Mapping common ICAO codes to OpenAP supported types
ICAO_TO_OPENAP = {
    'B738': 'b738', 'B737': 'b738', 'B734': 'b734',
    'B789': 'b789', 'B788': 'b788', 'B78X': 'b788',
    'B77W': 'b77w', 'B772': 'b772',
    'B748': 'b748', 'B744': 'b744',
    'B763': 'b763', 'B752': 'b752',
    'B38M': 'b738',
    'A20N': 'a20n', 'A320': 'a320', 'A321': 'a321', 'A319': 'a319',
    'A318': 'a318', 'A21N': 'a321',
    'A359': 'a359', 'A35K': 'a359',
    'A333': 'a333', 'A332': 'a332', 'A337': 'a333',
    'A343': 'a343',
    'A388': 'a388',
    'E195': 'e195', 'E190': 'e190',
    'E35L': 'e190',
    'E295': 'e195',
}

def collect_aircraft_data():
    """Fetches OpenAP data for all defined aircraft types."""
    aircraft_data = {}
    unknown_types = []

    for icao_code, openap_code in ICAO_TO_OPENAP.items():
        try:
            # Load basic aircraft data
            ac = openap.prop.aircraft(openap_code, use_synonym=True)

            # Extract technical specifications
            aircraft_data[icao_code] = {
                'openap_type': openap_code,
                'mtow': ac['limits']['MTOW'],      # Max Take-Off Weight
                'oew': ac['limits']['OEW'],        # Operating Empty Weight
                'mfc': ac['mfc'],                  # Max Fuel Capacity
                'wing_area': ac['wing']['area'],
                # Drag coefficients
                'cd0': ac['drag']['cd0'],          # Parasitic drag
                'k': ac['drag']['k'],              # Induced drag factor
                'oswald_efficiency': ac['drag']['e'],
                'gear_drag': ac['drag']['gears'],
                # Operational limits
                'vmo': ac['vmo'],                  # Max operating speed
                'mmo': ac['mmo'],                  # Max operating Mach
                'ceiling': ac['ceiling'],
                'cruise_alt': ac['cruise']['height'],
                'cruise_mach': ac['cruise']['mach'],
                'engine_type': ac['engine']['default'],
            }

            # Calculate CD2 (induced drag coefficient)
            # Formula derivation from BADA: k = cd2 * (pi * aspect_ratio)
            # Therefore: cd2 = k / (pi * AR)
            span = ac['wing']['span']
            area = ac['wing']['area']
            aspect_ratio = span**2 / area

            aircraft_data[icao_code]['cd2'] = ac['drag']['k'] / (3.14159 * aspect_ratio)

            print(f"[OK] {icao_code} -> {openap_code}")

        except Exception as e:
            unknown_types.append(icao_code)
            print(f"[ERROR] {icao_code}: {str(e)}")

    # Save to JSON
    output_file = config.DATA_RAW / 'aircraft_openap.json'
    with open(output_file, 'w') as f:
        json.dump(aircraft_data, f, indent=2)


    if unknown_types:
        print(f"\nUnknown types ({len(unknown_types)}):")
        print(", ".join(unknown_types))

    return aircraft_data

if __name__ == "__main__":
    collect_aircraft_data()