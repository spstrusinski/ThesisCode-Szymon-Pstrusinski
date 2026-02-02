import pandas as pd
import numpy as np
import statsmodels.api as sm

# 1) Paths — update these as needed
INPUT  = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_with_ids.xlsx"
OUTPUT = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\SD_subject_delta_hac_with_N.xlsx"

# 2) Load & sanitize column names
df = pd.read_excel(INPUT)
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r'\W+', '_', regex=True)
      .str.strip('_')
      .str.lower()
)

# 3) Key variables
df['entity_index'] = df['entity_index'].astype(str)
df['year']         = df['year'].astype(int)
df['postreform']   = df['postreform'].astype(int)

# 4) Identify between‐school SD columns
sd_cols = [c for c in df.columns
           if c.startswith('std_between_') or c.startswith('wstd_between_')]

def parse_sd(col):
    """Return (subject key, pretty subject name, weight type)."""
    parts = col.split('_')
    weight = 'weighted' if col.startswith('wstd_') else 'unweighted'
    subj_key = "_".join(parts[2:5])  # e.g. "math_poziom_advanced"
    pretty   = subj_key.replace('_',' ').title()
    return subj_key, pretty, weight

results = []
post_by_year = df.groupby('year')['postreform'].first()

for col in sd_cols:
    subj_key, subj, weight = parse_sd(col)

    # locate the raw‐percent‐SD column for filtering
    raw_col = subj_key + '_m_odchylenie_standardowe_percent'
    if raw_col not in df.columns:
        raw_col = subj_key + '_odchylenie_standardowe_percent'
    if raw_col not in df.columns:
        continue  # skip if no matching raw‐percent column

    # 5) filter to positive raw‐percent SD
    df_f = df[df[raw_col] > 0]

    # 6) compute N_t and sigma_t by year
    grp     = df_f[['year', col]].dropna().groupby('year')[col]
    Nt      = grp.count()       # number of schools in that year
    if Nt.empty:
        continue
    sigma_t = grp.first()

    # record average N across years
    avg_N = int(round(Nt.mean()))

    # 7) delta‐method SE per year
    se_t = sigma_t / np.sqrt(2 * Nt)

    # 8) build time series DataFrame
    ts = pd.DataFrame({'sigma': sigma_t, 'se_delta': se_t})
    ts = ts.join(post_by_year.rename('postreform'), how='left')
    ts = ts.dropna(subset=['postreform'])
    ts['w'] = 1.0 / ts['se_delta']**2

    # 9) WLS with Newey–West(1)
    X   = sm.add_constant(ts['postreform'])
    mod = sm.WLS(ts['sigma'], X, weights=ts['w'])
    fit = mod.fit(cov_type='HAC', cov_kwds={'maxlags':1})

    rho    = fit.params['postreform']
    se_hac = fit.bse['postreform']
    pval   = fit.pvalues['postreform']
    stars  = '***' if pval<0.01 else '**' if pval<0.05 else '*' if pval<0.1 else ''

    results.append({
        'Subject':     subj,
        'WeightType':  weight,
        'Avg_N':       avg_N,
        'Coef_post':   rho,
        'SE_HAC1':     se_hac,
        'p_value':     pval,
        'Stars':       stars
    })

# 10) Export summary with N included
out = pd.DataFrame(results)
out.to_excel(OUTPUT, index=False)
print("Done — wrote", OUTPUT)
print(out)
