import pandas as pd
import numpy as np

# Full path to your Excel file
file_path = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\2020minus-copcop.xlsx'

# Load the Excel file
df = pd.read_excel(file_path, sheet_name='szkoly_wyniki')

# Create new column with normalized gmina names
df['gmina_normalized'] = df['Gmina'].str.lower().str.replace(' ', '')

# Convert gmina_normalized to string and handle missing values
df['gmina_normalized'] = df['gmina_normalized'].fillna('').astype(str)

# Create the urban/rural indicator column
df['urban_rural'] = np.where(
    df['gmina_normalized'].str.startswith('m.'),
    'urban',
    'rural'
)


# Ensure gmina_normalized is string type (handle missing values)
df['gmina_normalized'] = df['gmina_normalized'].fillna('').astype(str)

# Create the new gmina_edited column
df['gmina_edited'] = df['gmina_normalized'].apply(
    lambda x: x[2:] if x.startswith('m.') else x
)


df['powiat_normalized'] = df['Powiat'].str.lower().str.replace(' ', '')

# Ensure gmina_normalized is string type (handle missing values)
df['powiat_normalized'] = df['powiat_normalized'].fillna('').astype(str)

# Create the new gmina_edited column
df['powiat_edited'] = df['powiat_normalized'].apply(
    lambda x: x[2:] if x.startswith('m.') else x
)


# Save back to the same Excel file
df.to_excel(file_path, index=False)

















