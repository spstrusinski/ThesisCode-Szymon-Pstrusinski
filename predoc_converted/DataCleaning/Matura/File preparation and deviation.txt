import os
import pandas as pd
from unidecode import unidecode

# 1) Exact paths
input_path  = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc dla Szymona.xlsx'
output_path = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_flattened.xlsx'

# 2) Make sure the output folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 3) Read the first two rows as a MultiIndex header
df = pd.read_excel(input_path, header=[0, 1], engine='openpyxl')

# 4) Function to clean & unify column names
def clean_name(text: str) -> str:
    s = unidecode(str(text))                    # remove accents
    s = s.strip().lstrip('*').strip().lower()    # drop asterisks & lowercase
    # replace symbols/spacing
    s = (s.replace('%', 'percent')
          .replace('(', '').replace(')', '')
          .replace('-', ' ')
          .replace('/', '_'))
    s = '_'.join(s.split())
    while '__' in s:
        s = s.replace('__', '_')
    return s

# 5) Flatten the two header rows into one snake_case name
new_cols = []
for lvl0, lvl1 in df.columns:
    combo = f"{lvl0} {lvl1}" if pd.notna(lvl0) and str(lvl0).strip() else lvl1
    new_cols.append(clean_name(combo))
df.columns = new_cols

# 6) Save to new Excel file
df.to_excel(output_path, index=False)
print(f"✅ Done — flattened file saved to:\n{output_path}")








import os
import pandas as pd
import numpy as np

# 1) Paths — update these to your actual files
input_path  = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_metrics.xlsx'
output_path = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_grouped_metrics.xlsx'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 2) Load your existing metrics file
df = pd.read_excel(input_path, engine='openpyxl')

# 3) Fix up the school‐type column name
type_cols = [c for c in df.columns if 'typ_placowki' in c]
if len(type_cols) == 1:
    df.rename(columns={type_cols[0]: 'typ_placowki'}, inplace=True)
else:
    raise KeyError(f"Expected one column with 'typ_placowki', found: {type_cols}")

# 3.1) Merge "liceum ogólnokształcące" into "liceum"
df['typ_placowki'] = df['typ_placowki'].replace(
    'liceum ogólnokształcące',
    'liceum'
)

# 4) Weighted‐std helper
def weighted_std(x, w):
    mask = (~x.isna()) & (~w.isna()) & (w > 0)
    xm, wm = x[mask], w[mask]
    if len(xm) == 0 or wm.sum() == 0:
        return np.nan
    m = np.average(xm, weights=wm)
    v = np.average((xm - m)**2, weights=wm)
    return np.sqrt(v)

# 5) Identify all subject‐mean columns and the two averages
subject_scores = [c for c in df.columns if '_sredni_score_' in c]
average_cols   = ['avg_basic_score', 'avg_advanced_score']
all_scores     = subject_scores + average_cols

# 6) Build group‐level stats
group_stats = []
for (yr, stype), grp in df.groupby(['year', 'typ_placowki']):
    stats = {'year': yr, 'typ_placowki': stype}
    for col in all_scores:
        # unweighted (population) std dev
        stats[f'std_between_{col}'] = grp[col].dropna().std(ddof=0)
        # weighted std dev only for individual subjects
        if col in subject_scores:
            base = col.split('_sredni_score')[0]
            # find matching students column
            candidates = [c for c in grp.columns if c.startswith(base) and 'students' in c]
            if len(candidates) != 1:
                raise KeyError(f"For {col}, expected one students‐col, found {candidates}")
            wcol = candidates[0]
            x = pd.to_numeric(grp[col],  errors='coerce')
            w = pd.to_numeric(grp[wcol], errors='coerce')
            stats[f'wstd_between_{base}'] = weighted_std(x, w)
        else:
            stats[f'wstd_between_{col}'] = np.nan
    group_stats.append(stats)

stats_df = pd.DataFrame(group_stats)

# 7) Merge the group‐level stats back onto your main DataFrame
df = df.merge(stats_df, on=['year', 'typ_placowki'], how='left')

# 8) Save to a new Excel file
df.to_excel(output_path, index=False)
print("Done — grouped metrics (with liceum merge) saved to:", output_path)















