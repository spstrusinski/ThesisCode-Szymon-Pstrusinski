import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the Excel file
file_path = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\E8 - szkoły (aktualizacja 07.2025).xlsx"
df = pd.read_excel(file_path)

# Filter data for OKE 3 and exclude 2019
df_filtered = df[(df['ID OKE'] == 3) & (df['Year'] != 2019)].copy()  # Add .copy() here

# Convert Math-Average to numeric - FIXED version
df_filtered.loc[:, 'Math-Average'] = pd.to_numeric(df_filtered['Math-Average'], errors='coerce')

# Calculate percentiles for each year
results = []
years = sorted(df_filtered['Year'].unique())

for year in years:
    year_data = df_filtered[df_filtered['Year'] == year].dropna(subset=['Math-Average'])

    if len(year_data) > 0:
        # Calculate cutoffs
        top_cutoff = year_data['Math-Average'].quantile(0.9)
        bottom_cutoff = year_data['Math-Average'].quantile(0.1)

        # Calculate averages
        top_avg = year_data[year_data['Math-Average'] >= top_cutoff]['Math-Average'].mean()
        bottom_avg = year_data[year_data['Math-Average'] <= bottom_cutoff]['Math-Average'].mean()

        results.append({
            'Year': year,
            'Top_10_Avg': top_avg,
            'Bottom_10_Avg': bottom_avg
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(results_df['Year'], results_df['Top_10_Avg'],
         marker='o', linestyle='-', color='green', label='Top 10% Schools')
plt.plot(results_df['Year'], results_df['Bottom_10_Avg'],
         marker='s', linestyle='-', color='red', label='Bottom 10% Schools')

plt.title('Math Scores: Top 10% vs Bottom 10% Schools (OKE 3, Excluding 2019)')
plt.xlabel('Year')
plt.ylabel('Average Math Score')
plt.xticks(years)  # Ensure all years are shown
plt.legend()
plt.grid(True, alpha=0.3)

# Save and show plot
plt.savefig('math_scores_comparison.png', dpi=300, bbox_inches='tight')
plt.show()