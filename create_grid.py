import pandas as pd
import numpy as np
import sys

INPUT_FILE = 'data/Accidents_AP7.csv'
OUTPUT_FILE = 'data/AP7_Grid_Temporal_Y.csv'

# Define the segment of the road we want to analyze (10km) 
SEGMENT_SIZE_KM = 10

def load_and_prep_accidents(filepath):
    """
    Load the accidents and prepare them for merging:
    1. Load the preprocessed CSV
    2. Convert 'data' and 'hora' into a 'timestamp_hora' (e.g. 19/06/2010 05:00:00)
    3. Assign each accident to a 'segmento_pk' (e.g. 138 -> 140)
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit() 

    # Replace the commas in 'pk' with dots, bc if not the conversion to numeric fails
    if pd.api.types.is_object_dtype(df['pk']):
        df['pk'] = df['pk'].astype(str).str.replace(',', '.', regex=False)
    
    # Convert 'data' (e.g. 19/06/2010) to datetime format (e.g. 2010-06-19 00:00:00)
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    
    # Convert 'hora' to the hour format suitable for processing (e.g. 5.3 -> 5:00:00)
    df['hora_timedelta'] = pd.to_timedelta(df['hora'].round().astype(int), unit='h')
    df['timestamp_hora'] = df['data'] + df['hora_timedelta'] # Combine date + hour
    
    # Convert 'pk' to numeric. Coerce errors: converts errors to NaN
    df['pk'] = pd.to_numeric(df['pk'], errors='coerce')
    
    # Assign each accident to its corresponding 'segmento_pk'
    #* Segments go from 0-9.9km -> segment 130.0-139.9km 
    df['segmento_pk'] = (df['pk'] / SEGMENT_SIZE_KM).apply(np.floor) * SEGMENT_SIZE_KM
    
    df = df.dropna(subset=['timestamp_hora', 'segmento_pk'])
    
    df['segmento_pk'] = df['segmento_pk'].astype(int)
    
    return df

def create_full_grid(min_date, max_date, pk_segments):
    """
    Creates the dataset grid with all combinations of hours and PK segments.
    """
    print("Generating Grid...")
    
    # Generate all hourly timestamps between min_date and max_date
    all_hours = pd.date_range(start=min_date, end=max_date, freq='h')
    
    print(f" - Time range: {min_date} to {max_date} ({len(all_hours)} hours)")
    print(f" - PK range: {pk_segments.min()} to {pk_segments.max()} ({len(pk_segments)} segments)")
    
    # Create the MultiIndex with all combinations (cartesian product)
    grid_index = pd.MultiIndex.from_product(
        [all_hours, pk_segments],
        names=['timestamp_hora', 'segmento_pk']
    )
    
    # Convert the index into a DataFrame
    grid_df = pd.DataFrame(index=grid_index).reset_index()
    print(f"Grid created. Total rows (hours x segments): {len(grid_df):,}")
    
    return grid_df

def map_accidents_to_grid(grid_df, df_accidents):
    """
    Maps accidents (cases '1') on the grid.
    All other rows will be '0'.
    """
    print("Mapping accidents (cases '1') to the grid...")
    
    # We only need the keys and mark that an accident occurred
    # We use drop_duplicates to mark 1 (accident) or 0 (no),
    # regardless of whether there was 1 or 2 accidents in that hour/segment
    df_accidents_slim = df_accidents[['timestamp_hora', 'segmento_pk']].drop_duplicates()
    df_accidents_slim['Y_ACCIDENT'] = 1
    
    # Keep EVERY row from the grid and add 'Y_ACCIDENT' where it matches
    grid_with_y = grid_df.merge(
        df_accidents_slim,
        on=['timestamp_hora', 'segmento_pk'],
        how='left'
    )
    
    # The grid rows that didn't have an accident (the vast majority)
    # will have a 'NaN' after the merge. We convert them to '0'
    grid_with_y['Y_ACCIDENT'] = grid_with_y['Y_ACCIDENT'].fillna(0).astype(int)
    
    return grid_with_y


if __name__ == "__main__":
    df_accidents = load_and_prep_accidents(INPUT_FILE)
    
    # Time: from the first day to the last day
    min_date = df_accidents['timestamp_hora'].min().floor('D')
    max_date = df_accidents['timestamp_hora'].max().ceil('D')
    
    # Space: from the first PK segment to the last
    min_pk_segment = np.floor(df_accidents['pk'].min() / SEGMENT_SIZE_KM) * SEGMENT_SIZE_KM
    max_pk_segment = np.ceil(df_accidents['pk'].max() / SEGMENT_SIZE_KM) * SEGMENT_SIZE_KM

    pk_segments = np.arange(
        min_pk_segment,
        max_pk_segment + SEGMENT_SIZE_KM, # +1 to include the last one
        SEGMENT_SIZE_KM
    ).astype(int)

    # Create the full Grid (all '0's and '1's)
    grid_df = create_full_grid(min_date, max_date, pk_segments)
    
    # Map the accidents to the Grid
    final_grid = map_accidents_to_grid(grid_df, df_accidents)

    print(f"\nSaving final grid to {OUTPUT_FILE}...")
    final_grid.to_csv(OUTPUT_FILE, index=False)
    
    print("\n--- Operation Summary ---")
    print(f"Total rows in the Grid: {len(final_grid):,}")
    print(f"Total 'hour-segment' with accidents (Y=1): {final_grid['Y_ACCIDENT'].sum():,}")
    print(f"Percentage of hours with accidents: {final_grid['Y_ACCIDENT'].mean() * 100:.4f}%")
    