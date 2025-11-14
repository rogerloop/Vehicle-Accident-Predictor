import numpy as np
import pandas as pd
import sys

GRID_FILE = 'data/AP7_Grid_Temporal_Y.csv'
ACCIDENTS_FILE = 'data/Accidents_AP7.csv' 
OUTPUT_FILE = 'data/AP7_Grid_with_Features.csv'
SEGMENT_SIZE_KM = 10

# Static columns: DO not change over time
STATIC_FEATURES = ['D_TIPUS_VIA', 'D_TRACAT_ALTIMETRIC', 'D_SENTITS_VIA',
    'D_LIMIT_VELOCITAT', 'C_VELOCITAT_VIA', 'D_TITULARITAT_VIA']

# Dynamic columns: Change over time 
#! THIS WILL BE FILLED WITH METEOCAT DATA
DYNAMIC_FEATURES = ['D_CLIMATOLOGIA', 'D_BOIRA', 'D_VENT', 'D_LLUMINOSITAT', 'D_SUPERFICIE']

def add_temporal_features(grid_df):
    """
    Adds temporal features to the grid DataFrame.
    """
    print("Adding Temporal Features...")
    

    grid_df['hour_of_day'] = grid_df['timestamp_hora'].dt.hour
    grid_df['month'] = grid_df['timestamp_hora'].dt.month
    grid_df['day_of_week'] = grid_df['timestamp_hora'].dt.dayofweek # Monday=0, Sunday=6
    
    #* Cyclical encoding (sin and cos transformations, because LSTM does not understand cyclic nature of time)
    # Hour
    grid_df['hora_sin'] = np.sin(2 * np.pi * grid_df['hour'] / 24)
    grid_df['hora_cos'] = np.cos(2 * np.pi * grid_df['hour'] / 24)

    # Month
    grid_df['mes_sin'] = np.sin(2 * np.pi * grid_df['month'] / 12)
    grid_df['mes_cos'] = np.cos(2 * np.pi * grid_df['month'] / 12)

    # Day of week
    grid_df['dia_semana_sin'] = np.sin(2 * np.pi * grid_df['day_of_week'] / 7)
    grid_df['dia_semana_cos'] = np.cos(2 * np.pi * grid_df['day_of_week'] / 7)

    return grid_df