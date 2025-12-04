import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import config


def process_data():
    input_path = config.DATA_PROCESSED / 'dataset_final.csv'
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)

    # Remove empty lines
    df = df.dropna()

    # Change aircraft names to numbers (Label Encoding)
    # The model needs numbers, not text
    le = LabelEncoder()
    df['aircraft_type_encoded'] = le.fit_transform(df['aircraft_type'])

    # Save the encoder so we can get the names back later
    encoder_path = config.DATA_PROCESSED / 'aircraft_encoder.pkl'
    joblib.dump(le, encoder_path)

    # Save the clean file for training
    output_path = config.DATA_PROCESSED / 'dataset_train_ready.csv'
    df.to_csv(output_path, index=False)



if __name__ == "__main__":
    process_data()