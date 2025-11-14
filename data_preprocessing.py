import pandas as pd
import numpy as np

#! Constants
INPUT_FILE= 'data/Accidents_de_tr_nsit_amb_morts_o_ferits_greus_a_Catalunya.csv'
OUTPUT_FILE= 'data/Accidents_AP7.csv'

COLUMNS_TO_KEEP = [
    'Any',                # Year   
    'data',               # Complete date
    'hora',               # Hour
    'grupHora',           # Time range
    'tipusDia',           # Type of day (e.g. 'Feiner')
    'grupDiaLab',         # Day group (e.g. 'Laborable')
    'via',                # Road (e.g. 'AP-7')
    'pk',                 # Kilometer point
    'nomMunicipi',        # Municipality name
    'nomComarca',         # County name
    'nomProvincia',       # Demarcation name
    'zona',
    'D_SUBZONA',
    'D_CLIMATOLOGIA',     # E.g. 'Bon temps'
    'D_BOIRA',            # E.g. 'No'
    'D_VENT',             # E.g. 'Vent fluix'
    'D_LLUMINOSITAT',     # E.g. 'Llum de dia'
    'D_SUPERFICIE',       # E.g. 'Seca'
    'D_CARACT_ENTORN',    # E.g. 'Recta'
    'D_CIRCULACIO_MESURES_ESP', # E.g. 'Sense mesures especials'
    'C_VELOCITAT_VIA',    # Speed limit (numeric)
    'D_LIMIT_VELOCITAT',  # Speed limit (text)
    'D_TIPUS_VIA',        # E.g. 'Autopista'
    'D_SUBTIPUS_TRAM',    # E.g. 'Tram general'
    'D_TRACAT_ALTIMETRIC', # E.g. 'Pla'
    'D_SENTITS_VIA',      # E.g. 'Dos sentits'
    'D_INTER_SECCIO',     # E.g. 'No'
    'D_REGULACIO_PRIORITAT', # E.g. 'Sense regulació'
    'D_TITULARITAT_VIA',  # E.g. 'Estat'
    'D_CARRIL_ESPECIAL',
    'D_FUNC_ESP_VIA',
    'D_GRAVETAT'       # E.g. 'Ferit greu', 'Mort', 'Ferit lleu'
]

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the DataFrame by handling missing values and duplicates."""
    df = df.drop_duplicates()
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def filter_data_by_column_value(df, column_name, value):
    """Filter the DataFrame by a specific column value."""
    return df[df[column_name] == value]

def select_columns(df, columns_to_keep):
    """Select specific columns from the DataFrame."""
    
    # Check that the columns exist in the DataFrame 
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    missing_columns = [col for col in columns_to_keep if col not in df.columns]

    if missing_columns:
        print(f"Warning: The following columns are missing and will be ignored: {missing_columns}")
    
    return df[existing_columns]

def save_data(df, output_path):
    """Save the DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    data = load_data(INPUT_FILE)
    cleaned_data = clean_data(data)
    filtered_data = filter_data_by_column_value(cleaned_data, 'via', 'AP-7')
    preprocessed_data = select_columns(filtered_data, COLUMNS_TO_KEEP)
    
    # Save the preprocess data to a new CSV file
    save_data(preprocessed_data, OUTPUT_FILE)
    
    print(f"Total accidents (original): {len(data)}")
    print(f"Filtered accidents (AP-7): {len(filtered_data)}")
    print(f"Original columns: {len(data.columns)}")
    print(f"Preprocessed columns: {len(preprocessed_data.columns)}")
    