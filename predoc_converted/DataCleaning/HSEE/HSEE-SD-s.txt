import pandas as pd
import numpy as np

# Load dataset
file_path = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\Working.xlsx'
df = pd.read_excel(file_path)

# Create all SD columns initialized with NaN
sd_columns = [
    'SD_Math_Overall', 'SD_Polish_Overall', 'SD_English_Overall',
    'SD_Math_Rural', 'SD_Polish_Rural', 'SD_English_Rural',
    'SD_Math_Urban', 'SD_Polish_Urban', 'SD_English_Urban'
]

for col in sd_columns:
    df[col] = np.nan

# Filter for ID OKE == 3
mask = df['ID OKE'] == 3
df_filtered = df.loc[mask].copy()


# Improved numeric conversion function
def convert_to_float(x):
    if isinstance(x, str):
        # Clean string values
        x = x.replace(',', '.').replace(' ', '').strip()
        # Handle cases like '-' or '.'
        if x in ['-', '.', '']:
            return np.nan
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan
    elif isinstance(x, (int, float)):
        return float(x)
    return np.nan


# Convert subject columns to numeric
subjects = {
    'Math': 'Math-Average',
    'Polish': 'Polish-Average',
    'English': 'English-Average'
}

for subj_col in subjects.values():
    df_filtered[subj_col] = df_filtered[subj_col].apply(convert_to_float)

# Get unique YearTypes
year_types = df_filtered['YearType'].unique()

# Compute and assign SD values for each subject and YearType
for subject, col_name in subjects.items():
    print(f"Processing {subject}...")

    # 1. Overall standard deviations
    overall_sd = df_filtered.groupby('YearType')[col_name].std()
    for year_type in year_types:
        if year_type in overall_sd.index:
            sd_value = overall_sd.loc[year_type]
            # Update main DataFrame
            df.loc[(df['ID OKE'] == 3) & (df['YearType'] == year_type), f'SD_{subject}_Overall'] = sd_value

    # 2. Rural standard deviations
    rural_mask = (df_filtered['gmina_class'] == 'rural')
    rural_sd = df_filtered[rural_mask].groupby('YearType')[col_name].std()
    for year_type in year_types:
        if year_type in rural_sd.index:
            sd_value = rural_sd.loc[year_type]
            df.loc[(df['ID OKE'] == 3) & (df['YearType'] == year_type), f'SD_{subject}_Rural'] = sd_value

    # 3. Urban standard deviations
    urban_mask = (df_filtered['gmina_class'] == 'urban')
    urban_sd = df_filtered[urban_mask].groupby('YearType')[col_name].std()
    for year_type in year_types:
        if year_type in urban_sd.index:
            sd_value = urban_sd.loc[year_type]
            df.loc[(df['ID OKE'] == 3) & (df['YearType'] == year_type), f'SD_{subject}_Urban'] = sd_value

# Save results
output_path = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\Working_with_SD.xlsx'
df.to_excel(output_path, index=False)
print(f"File saved successfully to: {output_path}")