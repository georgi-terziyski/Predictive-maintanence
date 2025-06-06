import pandas as pd
import os
import argparse

def standardize_and_validate_csv(file_path):
    """
    Standardizes a given CSV file to match the required format for the prediction model.
    - Renames columns to a standard set of names.
    - Removes duplicate columns.
    - Reorders columns to a standard order.
    - Validates that all required columns are present.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Define the standard column order and names
        required_columns = [
            'Timestamp', 'Machine_ID', 'AFR', 'Current', 'Pressure', 'RPM', 'Temperature', 'Vibration'
        ]

        # Define a mapping from possible legacy/incorrect names to the standard names
        column_mapping = {
            'timestamp': 'Timestamp',
            'machine_id': 'Machine_ID',
            'afr': 'AFR',
            'current': 'Current',
            'pressure': 'Pressure',
            'rpm': 'RPM',
            'temperature': 'Temperature',
            'vibration': 'Vibration'
        }

        # Rename columns based on the mapping
        df.rename(columns=column_mapping, inplace=True, errors='ignore')

        # Drop duplicate columns, keeping the first occurrence
        df = df.loc[:, ~df.columns.duplicated()]

        # Ensure all required columns are present, adding them with NA if they are missing
        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA
                print(f"Warning: Column '{col}' was missing. It has been added with NA values.")

        # Reorder the dataframe to match the standard format
        df = df[required_columns]

        # Overwrite the original file with the standardized data
        df.to_csv(file_path, index=False)
        print(f"Successfully standardized and validated {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standardize and validate CSV files for the predictive maintenance model.')
    parser.add_argument('file_path', type=str, help='The path to the CSV file to be standardized.')
    args = parser.parse_args()

    if os.path.exists(args.file_path):
        standardize_and_validate_csv(args.file_path)
    else:
        print(f"Error: The file '{args.file_path}' does not exist.")
