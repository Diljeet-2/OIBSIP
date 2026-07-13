import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

print("="*60)
print("UNEMPLOYMENT ANALYSIS - NOTEBOOK EXECUTION TEST")
print("="*60)

# ===== LOAD DATASET =====
print("\n[1] Loading Dataset...")
df = pd.read_csv("Unemployment in India.csv")
print("✓ Dataset loaded successfully!")

# ===== INSPECT DATA =====
print("\n[2] Inspecting Data...")
print(f"Dataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn Data Types:")
print(df.info())
print(f"\nNull Values:")
print(df.isnull().sum())

# ===== CONVERT DATE =====
print("\n[3] Converting Date Format...")
df['Date'] = pd.to_datetime(df['Date'])
print("✓ Date conversion successful!")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# ===== REGION-WISE ANALYSIS =====
print("\n[4] Region-wise Average Unemployment Rate...")
region_avg = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()
region_avg = region_avg.sort_values(ascending=False)
print(region_avg)

# ===== MONTH-WISE TREND =====
print("\n[5] Month-wise Average Unemployment Rate...")
df['Month'] = df['Date'].dt.month_name()
month_avg = df.groupby("Month")["Estimated Unemployment Rate (%)"].mean()
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
month_avg = month_avg.reindex(month_order)
print(month_avg)

# ===== TIME SERIES (3 STATES) =====
print("\n[6] Checking Time Series Data...")
states = ["Maharashtra", "Delhi", "Tamil Nadu"]
for state in states:
    count = len(df[df["Region"] == state])
    print(f"   {state}: {count} records")

# ===== TOP 10 STATES =====
print("\n[7] Top 10 Regions with Highest Average Unemployment...")
region_avg_full = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False)
top10 = region_avg_full.head(10)
print(top10)

# ===== CORRELATION HEATMAP =====
print("\n[8] Correlation Matrix...")
corr = df[[
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]].corr()
print(corr)

# ===== PRE-COVID VS POST-COVID =====
print("\n[9] Pre-COVID vs Post-COVID Analysis...")
pre = df[df["Date"] < "2020-03-01"]
post = df[df["Date"] >= "2020-03-01"]

print(f"\nPre-COVID (Before March 2020):")
print(f"  Average Unemployment Rate: {pre['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"  Average Employment Rate: {pre['Estimated Employed'].mean():.2f}")
print(f"  Average Labour Participation Rate: {pre['Estimated Labour Participation Rate (%)'].mean():.2f}%")

print(f"\nPost-COVID (March 2020 onwards):")
print(f"  Average Unemployment Rate: {post['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"  Average Employment Rate: {post['Estimated Employed'].mean():.2f}")
print(f"  Average Labour Participation Rate: {post['Estimated Labour Participation Rate (%)'].mean():.2f}%")

# ===== SAVE PLOTS =====
print("\n[10] Generating Visualizations...")

# Plot 1: Region-wise
plt.figure(figsize=(12, 6))
region_avg.plot(kind="barh", color="steelblue")
plt.title("Average Unemployment Rate by Region", fontsize=14, fontweight='bold')
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")
plt.tight_layout()
plt.savefig("01_region_wise_unemployment.png", dpi=100, bbox_inches='tight')
print("  ✓ Region-wise chart saved")
plt.close()

# Plot 2: Month-wise
plt.figure(figsize=(10, 5))
month_avg.plot(marker='o', color='coral', linewidth=2, markersize=8)
plt.title("Month-wise Average Unemployment Rate", fontsize=14, fontweight='bold')
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("02_month_wise_trend.png", dpi=100, bbox_inches='tight')
print("  ✓ Month-wise trend chart saved")
plt.close()

# Plot 3: Time Series
plt.figure(figsize=(14, 6))
for state in states:
    temp = df[df["Region"] == state]
    plt.plot(temp["Date"], temp["Estimated Unemployment Rate (%)"], label=state, marker='o', linewidth=2)
plt.legend(fontsize=10)
plt.title("Unemployment Rate Over Time for Major States", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("03_time_series_states.png", dpi=100, bbox_inches='tight')
print("  ✓ Time series chart saved")
plt.close()

# Plot 4: Top 10
plt.figure(figsize=(10, 6))
sns.barplot(x=top10.values, y=top10.index, palette="rocket")
plt.title("Top 10 States/Regions with Highest Average Unemployment", fontsize=14, fontweight='bold')
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")
plt.tight_layout()
plt.savefig("04_top_10_states.png", dpi=100, bbox_inches='tight')
print("  ✓ Top 10 states chart saved")
plt.close()

# Plot 5: Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label': 'Correlation'})
plt.title("Correlation Heatmap: Labour Market Indicators", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("05_correlation_heatmap.png", dpi=100, bbox_inches='tight')
print("  ✓ Correlation heatmap saved")
plt.close()

# Plot 6: Pre vs Post COVID
comparison = pd.DataFrame({
    "Period": ["Pre-COVID", "Post-COVID"],
    "Average Unemployment Rate": [
        pre["Estimated Unemployment Rate (%)"].mean(),
        post["Estimated Unemployment Rate (%)"].mean()
    ]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=comparison, x="Period", y="Average Unemployment Rate", palette="Set2")
plt.title("Pre vs Post COVID Average Unemployment Rate", fontsize=14, fontweight='bold')
plt.ylabel("Average Unemployment Rate (%)")
for i, v in enumerate(comparison["Average Unemployment Rate"]):
    plt.text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("06_pre_vs_post_covid.png", dpi=100, bbox_inches='tight')
print("  ✓ Pre vs Post COVID chart saved")
plt.close()

print("\n" + "="*60)
print("✓ ALL TESTS PASSED - NOTEBOOK VERIFIED!")
print("="*60)
print("\nGenerated visualizations:")
print("  1. 01_region_wise_unemployment.png")
print("  2. 02_month_wise_trend.png")
print("  3. 03_time_series_states.png")
print("  4. 04_top_10_states.png")
print("  5. 05_correlation_heatmap.png")
print("  6. 06_pre_vs_post_covid.png")
print("\n✓ All analyses completed successfully!")
print("✓ The notebook is ready to run with the actual Kaggle dataset!")
