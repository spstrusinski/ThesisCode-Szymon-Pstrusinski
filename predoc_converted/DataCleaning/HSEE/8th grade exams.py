import pandas as pd
import numpy


import pandas as pd
import numpy as np


# Read the Excel file (update the path with your actual filename)
file_path = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\E8 - szkoły (aktualizacja 07.2025).xlsx"
df = pd.read_excel(file_path)

# Create the new modified column
df['gmina_nazwa_modified'] = df['Gmina - nazwa'].str.replace(' ', '').str.lower()





# Define classification function
def classify_gmina(typ):
    if typ in ['Gmina wiejska', 'Obszar wiejski']:
        return 'rural'
    elif typ in ['Gmina miejska', 'Miasto']:
        return 'urban'
    else:
        return np.nan  # Handle unexpected values

# Create new column
df['gmina_class'] = df['Typ gminy'].apply(classify_gmina)


#  Create powiat modified column
df['powiat_nazwa_modified'] = df['powiat - nazwa'].str.replace(' ', '').str.lower()

# Create special combination columns
df['powiat_special'] = df['powiat_nazwa_modified'] + '-' + df['gmina_class']
df['gmina_special'] = df['gmina_nazwa_modified'] + '-' + df['gmina_class']


df.to_excel(file_path, index=False)