import pandas as pd
import numpy as np

# Load the 5,000-row empirical seed
df = pd.read_csv('C:/Users/abina/Downloads/AI_weather/AI_weather 2/AI_weather/assets/aurak_shuttle_data_full.csv')

def expand_dataset(original_df, target_rows=250000):
    # 1. Bootstrap sampling (sampling with replacement to create the volume)
    synthetic_df = original_df.sample(n=target_rows, replace=True).reset_index(drop=True)
    
    # 2. Add Gaussian Jitter to continuous variables to prevent identical duplicates
    # This simulates 'sensor noise' and local variance in real-world RAKTA data
    noise_map = {
        'stop_distance_km': 0.05,      # 50-meter variance
        'time_of_day': 0.08,           # ~5-minute variance
        'temperature_celsius': 0.2,    # 0.2 degree variance
        'delay_minutes': 0.4           # ~24-second variance
    }
    
    for col, noise_std in noise_map.items():
        noise = np.random.normal(0, noise_std, size=target_rows)
        synthetic_df[col] = synthetic_df[col] + noise
        
        # 3. Apply physical constraints (Clamping)
        # e.g., Shuttles can't travel negative distances or exist at 2:00 AM
        if col == 'stop_distance_km':
            synthetic_df[col] = synthetic_df[col].clip(lower=0.5, upper=15.0)
        elif col == 'time_of_day':
            synthetic_df[col] = synthetic_df[col].clip(lower=7.0, upper=22.0)
        elif col == 'temperature_celsius':
            synthetic_df[col] = synthetic_df[col].clip(lower=24.0, upper=42.0)
    
    # 4. Consistency Check: Re-label status based on delay
    # Based on the original data, a delay > 5.0 minutes is 'Late'
    synthetic_df['arrival_status'] = np.where(synthetic_df['delay_minutes'] > 5.0, 'Late', 'On-Time')
    
    return synthetic_df

# Generate 105,000 rows
expanded_data = expand_dataset(df)
expanded_data.to_csv('aurak_shuttle_data_250k.csv', index=False)