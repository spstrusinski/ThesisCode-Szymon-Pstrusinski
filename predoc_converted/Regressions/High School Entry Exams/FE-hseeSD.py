import pandas as pd
import numpy as np
import statsmodels.api as sm
import numpy as np
import statsmodels.api as sm

# 1) File paths — UPDATE to your directories
input_path  = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\Only_OKE3 — kopia (główna).xlsx"
output_path = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\SD_reform_effects-final.xlsx"

# 2) Load & sanitize column names
df = pd.read_excel(input_path)
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r'\W+', '_', regex=True)
      .str.strip('_')
)

# 3) Ensure key columns exist and are typed
df['T']            = df['T'].astype(int)
df['gmina_class']  = df['gmina_class'].astype(str).str.strip()
df['School_Type']  = df['School_Type'].astype(str).str.strip()

# 4) Build Gym_dummy from School_Type
df['Gym_dummy'] = (df['School_Type'] == 'Gymasium').astype(float)

# 5) Identify all between‐school SD columns
sd_cols = [c for c in df.columns if '_SD_' in c]

# (Optional debug) show how each sd_col splits:
print("Detected SD columns and split parts:")
for c in sd_cols:
    print(" ", c, "->", c.split('_'))

# 6) Build a series‐level table with delta‐method SEs and counts
records = []
for col in sd_cols:
    parts    = col.split('_')
    subj     = parts[0]
    weight   = parts[-2]
    group_raw= parts[-1]
    # normalize All→Overall
    group = 'Overall' if group_raw in ('Overall','All') else group_raw

    for T_val, block in df.groupby('T'):
        if group == 'Overall':
            subdf = block
        else:
            # match group (Rural/Urban) to gmina_class
            subdf = block[block['gmina_class'].str.lower() == group.lower()]

        N = len(subdf)
        if N < 2:
            continue

        sigma    = subdf[col].iloc[0]
        se_delta = sigma / np.sqrt(2 * N)

        records.append({
            'T':           T_val,
            'Subject':     subj,
            'WeightType':  weight,
            'Group':       group,
            'SD':          sigma,
            'SE_delta':    se_delta,
            'N':           N
        })

df_sigma = pd.DataFrame(records)
df_sigma.sort_values(['Subject','WeightType','Group','T'], inplace=True)

# 7) Create the post‐reform dummy (T ≥ 6)
df_sigma['Post'] = (df_sigma['T'] >= 6).astype(int)

# 8) Interrupted‐time‐series WLS + HAC(1) for each series
results = []
for subj in df_sigma['Subject'].unique():
    for wt in df_sigma['WeightType'].unique():
        for grp in ['Overall','Rural','Urban']:
            s = df_sigma[
                (df_sigma['Subject']==subj) &
                (df_sigma['WeightType']==wt) &
                (df_sigma['Group']==grp)
            ].sort_values('T')

            # require variation and at least 5 points
            if s['Post'].nunique()<2 or s['SD'].nunique()<2 or len(s)<5:
                continue

            X = sm.add_constant(s['Post'])
            y = s['SD']
            w = 1.0 / (s['SE_delta']**2)

            try:
                mod = sm.WLS(y, X, weights=w).fit(
                    cov_type='HAC', cov_kwds={'maxlags':1}
                )
            except np.linalg.LinAlgError:
                mod = sm.WLS(y, X, weights=w).fit()

            coef = mod.params.get('Post', np.nan)
            se   = mod.bse.get('Post',   np.nan)
            pval = mod.pvalues.get('Post',np.nan)

            if   pval < 0.01: stars='***'
            elif pval < 0.05: stars='**'
            elif pval < 0.1:  stars='*'
            else:             stars=''

            results.append({
                'Subject':    subj,
                'WeightType': wt,
                'Group':      grp,
                'N':          int(s['N'].iloc[0]),
                'Coef_Post':  f"{coef:.3f}{stars}",
                'HAC_SE':     f"{se:.3f}",
                'p_value':    f"{pval:.3f}"
            })

# 9) Export final results to Excel
out = pd.DataFrame(results)
out.to_excel(output_path, index=False)
print("Done — wrote", output_path)
print(out)









import pandas as pd
import numpy as np
import statsmodels.api as sm
import numpy as np
import statsmodels.api as sm

# 1) File paths — UPDATE to your directories
input_path  = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\Only_OKE3 — kopia (główna).xlsx"
output_path = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\SD_reform_effects-final_nocovid.xlsx"


# 2) Load & sanitize column names
df = pd.read_excel(input_path)
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r'\W+', '_', regex=True)
      .str.strip('_')
)

# 3) Ensure key columns exist and are typed
df['T']            = df['T'].astype(int)
df['gmina_class']  = df['gmina_class'].astype(str).str.strip()
df['School_Type']  = df['School_Type'].astype(str).str.strip()

# 4) Build Gym_dummy from School_Type
df['Gym_dummy'] = (df['School_Type'] == 'Gymasium').astype(float)

# 5) Identify all between-school SD columns
sd_cols = [c for c in df.columns if '_SD_' in c]

# 6) Build the series-level table with delta-method SEs & N
records = []
for col in sd_cols:
    parts     = col.split('_')
    subj      = parts[0]
    weight    = parts[-2]
    group_raw = parts[-1]
    group     = 'Overall' if group_raw in ('Overall','All') else group_raw

    for T_val, block in df.groupby('T'):
        if group == 'Overall':
            subdf   = block
        else:
            subdf   = block[block['gmina_class'].str.lower() == group.lower()]

        N = len(subdf)
        if N < 2:
            continue

        sigma    = subdf[col].iloc[0]
        se_delta = sigma / np.sqrt(2 * N)

        records.append({
            'T':          T_val,
            'Subject':    subj,
            'WeightType': weight,
            'Group':      group,
            'SD':         sigma,
            'SE_delta':   se_delta,
            'N':          N
        })

df_sigma = pd.DataFrame(records)
df_sigma.sort_values(['Subject','WeightType','Group','T'], inplace=True)

# 7) Drop COVID years T = 7, 8, 9
df_sigma = df_sigma[~df_sigma['T'].isin([7, 8, 9])].copy()

# 8) Create the post-reform dummy (T ≥ 6)
df_sigma['Post'] = (df_sigma['T'] >= 6).astype(int)

# 9) Interrupted-time-series WLS + HAC(1)
results = []
for subj in df_sigma['Subject'].unique():
    for wt in df_sigma['WeightType'].unique():
        for grp in ['Overall','Rural','Urban']:
            s = df_sigma[
                (df_sigma['Subject']==subj) &
                (df_sigma['WeightType']==wt) &
                (df_sigma['Group']==grp)
            ].sort_values('T')
            if s['Post'].nunique()<2 or s['SD'].nunique()<2 or len(s)<5:
                continue

            X = sm.add_constant(s['Post'])
            y = s['SD']
            w = 1.0 / (s['SE_delta']**2)

            try:
                mod = sm.WLS(y, X, weights=w).fit(
                    cov_type='HAC', cov_kwds={'maxlags':1}
                )
            except np.linalg.LinAlgError:
                mod = sm.WLS(y, X, weights=w).fit()

            coef = mod.params.get('Post', np.nan)
            se   = mod.bse.get('Post',   np.nan)
            pval = mod.pvalues.get('Post',np.nan)

            if   pval < 0.01: stars='***'
            elif pval < 0.05: stars='**'
            elif pval < 0.1:  stars='*'
            else:             stars=''

            results.append({
                'Subject':    subj,
                'WeightType': wt,
                'Group':      grp,
                'N':          int(s['N'].iloc[0]),
                'Coef_Post':  f"{coef:.3f}{stars}",
                'HAC_SE':     f"{se:.3f}",
                'p_value':    f"{pval:.3f}"
            })

# 10) Export final results
out = pd.DataFrame(results)
out.to_excel(output_path, index=False)
print("Done — wrote", output_path)
print(out)