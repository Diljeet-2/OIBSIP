import pandas as pd
import numpy as np
from datetime import datetime

# Create realistic India unemployment dataset matching Kaggle structure
regions = [
    "Haryana", "Tripura", "Jharkhand", "Odisha", "Himachal Pradesh",
    "Uttarakhand", "Assam", "Punjab", "Meghalaya", "West Bengal",
    "Bihar", "Gujarat", "Telangana", "Rajasthan", "Andhra Pradesh",
    "Karnataka", "Tamil Nadu", "Kerala", "Maharashtra", "Delhi",
    "Uttar Pradesh", "Madhya Pradesh", "Goa", "Manipur"
]

# Create date range
start_date = datetime(2019, 1, 1)
end_date = datetime(2021, 12, 31)
dates = pd.date_range(start_date, end_date, freq='MS')

data = []

# Generate realistic data
np.random.seed(42)
for region in regions:
    base_unemployment = 3.5 + np.random.uniform(-1, 2)
    base_employed = 52 + np.random.uniform(-5, 5)
    base_labour = 58 + np.random.uniform(-3, 3)
    
    for date in dates:
        month = date.month
        year = date.year
        
        # Seasonal variation
        seasonal = 0.5 * np.sin((month / 12) * 2 * np.pi)
        
        # COVID impact (sharp increase March-Aug 2020, then gradual recovery)
        covid_impact = 0
        if year == 2020 and month >= 3:
            if month <= 8:
                covid_impact = 3.0 + np.random.uniform(0, 2)
            else:
                covid_impact = 2.5 - (month - 8) * 0.3 + np.random.uniform(-0.5, 0.5)
        elif year == 2021:
            covid_impact = max(0.2, 1.5 - (month / 12) * 0.8) + np.random.uniform(-0.3, 0.3)
        
        unemployment_rate = max(0.5, base_unemployment + seasonal + covid_impact + np.random.normal(0, 0.2))
        employed = max(30, base_employed - 0.4 * seasonal - 0.6 * covid_impact + np.random.normal(0, 0.5))
        labour_participation = max(40, base_labour - 0.2 * covid_impact + np.random.normal(0, 0.3))
        
        data.append({
            'Region': region,
            'Date': date,
            'Frequency': 'Monthly',
            'Estimated Unemployment Rate (%)': round(unemployment_rate, 2),
            'Estimated Employed': round(employed * 1e6, 0),
            'Estimated Labour Participation Rate (%)': round(labour_participation, 2)
        })

df = pd.DataFrame(data)
df = df.sort_values(['Region', 'Date']).reset_index(drop=True)

# Save to CSV
df.to_csv('Unemployment in India.csv', index=False)

print("✓ Dataset created successfully!")
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nData types:")
print(df.dtypes)
print(f"\n✓ File saved: Unemployment in India.csv")
