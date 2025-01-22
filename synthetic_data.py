# generate_data.py
import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate synthetic features
    machine_ids = np.random.randint(1, 11, n_samples)
    
    # Temperature data with more realistic patterns
    # Normal operating temperature: 60-80°C
    # High risk above 85°C
    temperatures = np.concatenate([
        np.random.normal(70, 5, int(n_samples * 0.7)),  # 70% normal operation
        np.random.normal(90, 5, int(n_samples * 0.3))   # 30% high temperature
    ])
    np.random.shuffle(temperatures)
    
    # Run time with more realistic patterns
    # Normal runtime: 80-120 hours
    # Risky runtime: > 140 hours
    run_times = np.concatenate([
        np.random.normal(100, 10, int(n_samples * 0.6)),  # 60% normal runtime
        np.random.normal(150, 15, int(n_samples * 0.4))   # 40% extended runtime
    ])
    np.random.shuffle(run_times)
    
    # Generate downtime based on more complex conditions
    downtime = []
    for temp, time in zip(temperatures, run_times):
        risk_score = 0
        
        # Temperature-based risk
        if temp < 60:
            risk_score += 0.3  # Too cold
        elif temp > 85:
            risk_score += 0.4  # Too hot
        elif temp > 90:
            risk_score += 0.6  # Critically hot
            
        # Runtime-based risk
        if time > 140:
            risk_score += 0.3  # Extended runtime
        if time > 160:
            risk_score += 0.3  # Critical runtime
            
        # Combined effects
        if temp > 85 and time > 140:
            risk_score += 0.2  # Compound risk
            
        # Add some randomness
        risk_score += np.random.normal(0, 0.1)
        
        # Determine downtime
        downtime.append(1 if risk_score > 0.7 else 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Machine_ID': machine_ids,
        'Temperature': temperatures,
        'Run_Time': run_times,
        'Downtime_Flag': downtime
    })
    
    # Add some noise reduction and cleanup
    df = df[df['Temperature'] > 0]  # Remove implausible temperatures
    df = df[df['Run_Time'] > 0]     # Remove implausible run times
    
    # Save to CSV
    df.to_csv('data/manufacturing_data.csv', index=False)
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Downtime events: {df['Downtime_Flag'].sum()} ({df['Downtime_Flag'].mean()*100:.1f}%)")
    print("\nTemperature Statistics:")
    print(df['Temperature'].describe())
    print("\nRun Time Statistics:")
    print(df['Run_Time'].describe())
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()