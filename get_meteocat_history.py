import requests
import pandas as pd
import time
import os
from datetime import datetime

API_KEY = "0UKnjucrjp3ePOpc7fd0O9p3kgbfdL482wTN9ARG"

#! XEMA stations near AP-7 (ADD AS NECESSARY)
#! e.g.: X4 (El Papiol), X8 (Vilobí d'Onyar), XD (Tarragona)
ESTACIONS = ['X4', 'X8', 'XD', 'XG']

# Get the variable codes from the API documentation
# 32 = Temperature (ºC)
# 4  = Precipitation (mm)
# 35 = Average wind speed (m/s)
# 47 = Maximum wind gust (m/s)
# 50 = Relative humidity (%)
CODIGOS_VARIABLES = [32, 4, 35, 47, 50]

DATA_INICI = "2010-01-01"
DATA_FI = "2023-12-31"

OUTPUT_DIR = "data/meteo_history"

# URL of the Meteocat XEMA API
API_URL = "https://api.meteo.cat/xema/v1/observacions/horaris"

def generate_date_chunks(start_date, end_date, chunk_days=30):
    """
    Generates tuples of (start, end) in 30-day ranges,
    because the API does not allow more than 31 days per query.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    date_chunks = []
    current_start = start
    
    while current_start <= end:
        current_end = current_start + pd.Timedelta(days=chunk_days)
        if current_end > end:
            current_end = end
        
        # Date format for the API (YYYY-MM-DDZ)
        date_chunks.append((
            current_start.strftime('%Y-%m-%dZ'),
            current_end.strftime('%Y-%m-%dZ')
        ))
        current_start = current_end + pd.Timedelta(days=1)
        
    return date_chunks

def fetch_data_for_station(codi_estacio):
    """
    Downloads ALL data for ONE station, month by month.
    """
    print(f"\n--- Starting download for Station: {codi_estacio} ---")
    
    headers = {"X-Api-Key": API_KEY}
    date_chunks = generate_date_chunks(DATA_INICI, DATA_FI)
    
    all_data_df = None
    
    # Loop through each variable to download
    for codi_variable in CODIGOS_VARIABLES:
        print(f"  Downloading variable: {codi_variable}...")
        
        variable_data = [] # List to store data for this variable
        
        # Loop through all 30-day ranges
        for i, (data_ini, data_fi) in enumerate(date_chunks):
            
            params = {
                "codiEstacio": codi_estacio,
                "codiVariable": codi_variable,
                "dataIni": data_ini,
                "dataFi": data_fi
            }
            
            try:
                response = requests.get(API_URL, headers=headers, params=params, timeout=20)
                response.raise_for_status() # Raise error if API fails
                
                data = response.json()
                
                # 'observacions' is the list of data
                if 'observacions' in data and data['observacions']:
                    variable_data.extend(data['observacions'])
                
                print(f" Chunk {i+1}/{len(date_chunks)} ({data_ini}) completed.")
                
                # Pause to avoid overloading the API
                time.sleep(0.5) # 0.5 second pause
                
            except requests.exceptions.HTTPError as e:
                print(f"    HTTP Error! {e.response.status_code} in chunk {i+1}. Skipping...")
            except requests.exceptions.RequestException as e:
                print(f"    Network error! {e}. Skipping chunk...")
        
        if not variable_data:
            print(f"  No data found for variable {codi_variable}.")
            continue
            
        # Convert the list of JSONs to a DataFrame
        df_var = pd.DataFrame(variable_data)
        
        # Rename 'valor' to 'variable_XX' to be able to merge them
        df_var = df_var.rename(columns={'valor': f'var_{codi_variable}'})
        
        # Keep only date, hour and the value
        df_var = df_var[['data', 'hora', f'var_{codi_variable}']]
        
        # Merge with the main DataFrame of the station
        if all_data_df is None:
            all_data_df = df_var
        else:
            all_data_df = pd.merge(all_data_df, df_var, on=['data', 'hora'], how='outer')

    return all_data_df

def create_timestamp(df):
    """
    Creates a 'timestamp_hora' compatible with the grid
    The API returns 'data' (e.g. 2010-01-01T00:00:00Z) and 'hora' (e.g. 13:00)
    """
    # The API returns 'data' with time 00:00:00. We extract it
    df['fecha_dia'] = pd.to_datetime(df['data']).dt.date
    
    # Combine the day date with the 'hora' (e.g. 13:00) and force it to 'datetime'
    df['timestamp_hora'] = pd.to_datetime(df['fecha_dia'].astype(str) + ' ' + df['hora'])
    
    # Select and reorder columns
    cols_to_keep = ['timestamp_hora'] + [col for col in df.columns if col.startswith('var_')]
    df_final = df[cols_to_keep]
    
    return df_final.drop_duplicates()


if __name__ == "__main__":
    
    if API_KEY == "posa_la_teva_api_key_de_meteocat_aqui":
        print("Error: You must edit the script and set your API_KEY in the 'API_KEY' variable.")
        sys.exit()

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Main loop: Download data for each station
    for estacio in ESTACIONS:
        df_estacio = fetch_data_for_station(estacio)
        
        if df_estacio is not None and not df_estacio.empty:
            # Process the final DataFrame for the station
            df_estacio_final = create_timestamp(df_estacio)
            
            output_path = os.path.join(OUTPUT_DIR, f"meteo_{estacio}_{DATA_INICI}_a_{DATA_FI}.csv")
            df_estacio_final.to_csv(output_path, index=False)
            print(f"--- SUCCESS! Data from station {estacio} saved to {output_path} ---")
        else:
            print(f"--- Failed: Could not download data for station {estacio} ---")

    print("\nWeather data download completed.")