import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# 1) Paths — adjust these to your environment
INPUT  = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_with_ids.xlsx"
OUTPUT = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\FE_regressions_full_outcomes.xlsx"

# 2) Load & sanitize column names
df = pd.read_excel(INPUT)
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r'\W+', '_', regex=True)
      .str.strip('_')
      .str.lower()
)

# 3) Detect and recode the “Tak/Nie” and “rodzaj placówki” columns
public_col = [c for c in df.columns if 'czy_publicz' in c][0]
rodzaj_col = [c for c in df.columns if 'rodzaj_placow' in c][0]
df['is_public']  = df[public_col].astype(str).str.strip().str.lower().eq('tak').astype(int)
df['youth_only'] = df[rodzaj_col].astype(str).str.strip().str.lower().eq('dla młodzieży').astype(int)
df['is_liceum']  = df['typ_placowki'].astype(str).str.strip().str.lower().eq('liceum').astype(int)

# 4) Panel keys and additional regressors
df['entity_index'] = df['entity_index'].astype(str)
df['year']         = df['year'].astype(int)
df['postreform']   = df['postreform'].astype(int)
df['covid']        = df['year'].isin([2021, 2022]).astype(int)
# gmina_type already 0=rural, 1=urban
df['gmina_type']   = df['gmina_type'].astype(int)

# 5) Auto‐detect all outcome variables ending in '_percent'
y_cols = [c for c in df.columns if c.endswith('_percent')]
print("Outcomes found:", y_cols)

# 6) Function to generate significance stars
def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

# 7) Run regressions and collect results
results = []
for y in y_cols:
    data = (
        df.set_index(['entity_index','year'])
          [[y, 'postreform', 'gmina_type', 'covid']]
          .dropna()
    )
    if data.empty:
        continue

    clusters = data.index.get_level_values('entity_index').nunique()
    obs      = len(data)

    # Model 1
    ex1 = sm.add_constant(data['postreform'])
    m1  = PanelOLS(data[y], ex1,
                   entity_effects=True,
                   time_effects=True,
                   drop_absorbed=True)
    r1  = m1.fit(cov_type='clustered', cluster_entity=True)
    b1   = r1.params['postreform']
    se1  = r1.std_errors['postreform']
    p1   = r1.pvalues['postreform']
    b1s  = f"{b1:.3f}{stars(p1)}"

    # Model 2
    data['post_x_gmina'] = data['postreform'] * data['gmina_type']
    ex2 = sm.add_constant(data[['postreform','gmina_type','post_x_gmina']])
    m2  = PanelOLS(data[y], ex2,
                   entity_effects=True,
                   time_effects=True,
                   drop_absorbed=True)
    r2  = m2.fit(cov_type='clustered', cluster_entity=True)
    b2_base = r2.params['postreform']
    se2_base= r2.std_errors['postreform']
    p2_base = r2.pvalues['postreform']
    b2_base_s = f"{b2_base:.3f}{stars(p2_base)}"
    b2_int  = r2.params['post_x_gmina']
    se2_int = r2.std_errors['post_x_gmina']
    p2_int  = r2.pvalues['post_x_gmina']
    b2_int_s = f"{b2_int:.3f}{stars(p2_int)}"

    # Model 3
    data['covid_x_gmina'] = data['covid'] * data['gmina_type']
    ex3 = sm.add_constant(data[[
        'postreform','gmina_type','post_x_gmina',
        'covid','covid_x_gmina'
    ]])
    m3  = PanelOLS(data[y], ex3,
                   entity_effects=True,
                   time_effects=True,
                   drop_absorbed=True)
    r3  = m3.fit(cov_type='clustered', cluster_entity=True)
    b3_post   = r3.params['postreform'];   se3_post   = r3.std_errors['postreform'];   p3_post   = r3.pvalues['postreform']
    b3_pint   = r3.params['post_x_gmina']; se3_pint   = r3.std_errors['post_x_gmina']; p3_pint   = r3.pvalues['post_x_gmina']
    b3_covid  = r3.params.get('covid',np.nan);     se3_covid  = r3.std_errors.get('covid',np.nan);     p3_covid  = r3.pvalues.get('covid',np.nan)
    b3_cint   = r3.params.get('covid_x_gmina',np.nan); se3_cint   = r3.std_errors.get('covid_x_gmina',np.nan); p3_cint   = r3.pvalues.get('covid_x_gmina',np.nan)
    b3_post_s  = f"{b3_post:.3f}{stars(p3_post)}"
    b3_pint_s  = f"{b3_pint:.3f}{stars(p3_pint)}"
    b3_covid_s = f"{b3_covid:.3f}{stars(p3_covid)}"
    b3_cint_s  = f"{b3_cint:.3f}{stars(p3_cint)}"

    results.append({
        'Outcome':        y,
        'Clusters':       clusters,
        'Obs':            obs,
        'M1_coef_post':   b1s,
        'M1_se_post':     f"{se1:.3f}",
        'M2_coef_base':   b2_base_s,
        'M2_se_base':     f"{se2_base:.3f}",
        'M2_coef_int':    b2_int_s,
        'M2_se_int':      f"{se2_int:.3f}",
        'M3_coef_post':   b3_post_s,
        'M3_se_post':     f"{se3_post:.3f}",
        'M3_coef_pint':   b3_pint_s,
        'M3_se_pint':     f"{se3_pint:.3f}",
        'M3_coef_covid':  b3_covid_s,
        'M3_se_covid':    f"{se3_covid:.3f}",
        'M3_coef_cint':   b3_cint_s,
        'M3_se_cint':     f"{se3_cint:.3f}"
    })

# 8) Export to Excel
pd.DataFrame(results).to_excel(OUTPUT, index=False)
print("Done — wrote", OUTPUT)








import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# 1) Paths — adjust these to your setup
INPUT  = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_with_ids.xlsx"
OUTPUT = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\FE_avg_basic_advanced.xlsx"

# 2) Load & sanitize column names
df = pd.read_excel(INPUT)
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r'\W+', '_', regex=True)
      .str.strip('_')
      .str.lower()
)

# 3) Recode panel keys & regressors
df['entity_index'] = df['entity_index'].astype(str)
df['year']         = df['year'].astype(int)
df['postreform']   = df['postreform'].astype(int)
# COVID is collinear with year‐FE and will be absorbed if included alone
df['covid']        = df['year'].isin([2021, 2022]).astype(int)
# gmina_type already 0=rural,1=urban
df['gmina_type']   = df['gmina_type'].astype(int)

# 4) Explicitly list the two outcomes
y_cols = ['avg_basic_score', 'avg_advanced_score']

# 5) Helper for stars
def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

# 6) Run the three FE regressions for each outcome
results = []

for y in y_cols:
    panel = (
        df.set_index(['entity_index','year'])
          [[y, 'postreform','gmina_type','covid']]
          .dropna()
    )
    if panel.empty:
        continue

    clusters = panel.index.get_level_values('entity_index').nunique()
    obs      = len(panel)

    # --- Model 1: postreform only
    ex1 = sm.add_constant(panel['postreform'])
    m1  = PanelOLS(panel[y], ex1,
                   entity_effects=True,
                   time_effects=True,
                   drop_absorbed=True)
    r1  = m1.fit(cov_type='clustered', cluster_entity=True)
    b1, se1, p1 = r1.params['postreform'], r1.std_errors['postreform'], r1.pvalues['postreform']
    b1s = f"{b1:.3f}{stars(p1)}"

    # --- Model 2: add postreform×urban
    panel['post_x_urb'] = panel['postreform'] * panel['gmina_type']
    ex2 = sm.add_constant(panel[['postreform','gmina_type','post_x_urb']])
    m2  = PanelOLS(panel[y], ex2,
                   entity_effects=True,
                   time_effects=True,
                   drop_absorbed=True)
    r2  = m2.fit(cov_type='clustered', cluster_entity=True)
    b2b, se2b, p2b = r2.params['postreform'],    r2.std_errors['postreform'],    r2.pvalues['postreform']
    b2i, se2i, p2i = r2.params['post_x_urb'],    r2.std_errors['post_x_urb'],    r2.pvalues['post_x_urb']
    b2b_s = f"{b2b:.3f}{stars(p2b)}"
    b2i_s = f"{b2i:.3f}{stars(p2i)}"

    # --- Model 3: add covid×urban (covid main will be absorbed)
    panel['covid_x_urb'] = panel['covid'] * panel['gmina_type']
    ex3 = sm.add_constant(panel[['postreform','gmina_type','post_x_urb','covid','covid_x_urb']])
    m3  = PanelOLS(panel[y], ex3,
                   entity_effects=True,
                   time_effects=True,
                   drop_absorbed=True)
    r3  = m3.fit(cov_type='clustered', cluster_entity=True)
    b3p, se3p, p3p = r3.params['postreform'],      r3.std_errors['postreform'],      r3.pvalues['postreform']
    b3i, se3i, p3i = r3.params['post_x_urb'],      r3.std_errors['post_x_urb'],      r3.pvalues['post_x_urb']
    b3c, se3c, p3c = r3.params.get('covid',np.nan), r3.std_errors.get('covid',np.nan), r3.pvalues.get('covid',np.nan)
    b3ci,se3ci,p3ci= r3.params.get('covid_x_urb',np.nan), r3.std_errors.get('covid_x_urb',np.nan), r3.pvalues.get('covid_x_urb',np.nan)
    b3p_s  = f"{b3p:.3f}{stars(p3p)}"
    b3i_s  = f"{b3i:.3f}{stars(p3i)}"
    b3c_s  = f"{b3c:.3f}{stars(p3c)}"
    b3ci_s = f"{b3ci:.3f}{stars(p3ci)}"

    results.append({
        'Outcome':          y,
        'Clusters':         clusters,
        'Obs':              obs,
        'M1_coef_post':     b1s,
        'M1_se_post':       f"{se1:.3f}",
        'M2_coef_base':     b2b_s,
        'M2_se_base':       f"{se2b:.3f}",
        'M2_coef_int':      b2i_s,
        'M2_se_int':        f"{se2i:.3f}",
        'M3_coef_post':     b3p_s,
        'M3_se_post':       f"{se3p:.3f}",
        'M3_coef_pint':     b3i_s,
        'M3_se_pint':       f"{se3i:.3f}",
        'M3_coef_covid':    b3c_s,
        'M3_se_covid':      f"{se3c:.3f}",
        'M3_coef_cint':     b3ci_s,
        'M3_se_cint':       f"{se3ci:.3f}"
    })

# 7) Export to Excel
pd.DataFrame(results).to_excel(OUTPUT, index=False)
print("Done — wrote", OUTPUT)












import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# 1) Paths — adjust to your environment
INPUT  = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_with_ids.xlsx"
OUTPUT = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\FE_regressions_school_types.xlsx"

# 2) Load & sanitize column names
df = pd.read_excel(INPUT)
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r'\W+', '_', regex=True)
      .str.strip('_')
      .str.lower()
)

# 3) Detect original columns
public_col = [c for c in df.columns if 'czy_publicz' in c][0]
rodzaj_col = [c for c in df.columns if 'rodzaj_placow' in c][0]

# 4) Recode your three binaries
df['is_public']   = (df[public_col].astype(str).str.lower() == 'tak').astype(int)
df['youth_only']  = (df[rodzaj_col].astype(str).str.lower() == 'dla młodzieży').astype(int)
df['is_liceum']   = (df['typ_placowki'].astype(str).str.lower() == 'liceum').astype(int)

# 5) Panel keys + other dummies
df['entity_index'] = df['entity_index'].astype(str)
df['year']         = df['year'].astype(int)
df['postreform']   = df['postreform'].astype(int)
df['covid']        = df['year'].isin([2021, 2022]).astype(int)
df['gmina_type']   = df['gmina_type'].astype(int)  # already 0=rural,1=urban

# 6) Only the three recoded outcomes
y_cols = ['is_liceum', 'youth_only', 'is_public']

# 7) Stars helper
def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

# 8) Loop over outcomes and run FE specs
results = []
for y in y_cols:
    panel = (
        df.set_index(['entity_index', 'year'])
          [[y, 'postreform', 'gmina_type', 'covid']]
          .dropna()
    )
    if panel.empty:
        continue

    clusters = panel.index.get_level_values('entity_index').nunique()
    obs      = len(panel)

    # Model 1: postreform
    ex1 = sm.add_constant(panel['postreform'])
    m1  = PanelOLS(panel[y], ex1,
                   entity_effects=True,
                   time_effects=True,
                   drop_absorbed=True)
    r1  = m1.fit(cov_type='clustered', cluster_entity=True)
    b1, se1, p1 = r1.params['postreform'], r1.std_errors['postreform'], r1.pvalues['postreform']
    b1s = f"{b1:.3f}{stars(p1)}"

    # Model 2: + post × urban
    panel['post_x_urb'] = panel['postreform'] * panel['gmina_type']
    ex2 = sm.add_constant(panel[['postreform','gmina_type','post_x_urb']])
    m2  = PanelOLS(panel[y], ex2,
                   entity_effects=True,
                   time_effects=True,
                   drop_absorbed=True)
    r2 = m2.fit(cov_type='clustered', cluster_entity=True)
    b2b, se2b, p2b = r2.params['postreform'],    r2.std_errors['postreform'],    r2.pvalues['postreform']
    b2i, se2i, p2i = r2.params['post_x_urb'],    r2.std_errors['post_x_urb'],    r2.pvalues['post_x_urb']
    b2b_s = f"{b2b:.3f}{stars(p2b)}"
    b2i_s = f"{b2i:.3f}{stars(p2i)}"

    # Model 3: + covid + covid × urban
    panel['covid_x_urb'] = panel['covid'] * panel['gmina_type']
    ex3 = sm.add_constant(panel[[
        'postreform','gmina_type','post_x_urb',
        'covid','covid_x_urb'
    ]])
    m3  = PanelOLS(panel[y], ex3,
                   entity_effects=True,
                   time_effects=True,
                   drop_absorbed=True)
    r3 = m3.fit(cov_type='clustered', cluster_entity=True)
    b3p, se3p, p3p  = (r3.params.get('postreform',np.nan),
                      r3.std_errors.get('postreform',np.nan),
                      r3.pvalues.get('postreform',np.nan))
    b3i, se3i, p3i  = (r3.params.get('post_x_urb',np.nan),
                      r3.std_errors.get('post_x_urb',np.nan),
                      r3.pvalues.get('post_x_urb',np.nan))
    b3c, se3c, p3c  = (r3.params.get('covid',np.nan),
                      r3.std_errors.get('covid',np.nan),
                      r3.pvalues.get('covid',np.nan))
    b3ci,se3ci,p3ci = (r3.params.get('covid_x_urb',np.nan),
                       r3.std_errors.get('covid_x_urb',np.nan),
                       r3.pvalues.get('covid_x_urb',np.nan))
    b3p_s  = f"{b3p:.3f}{stars(p3p)}"
    b3i_s  = f"{b3i:.3f}{stars(p3i)}"
    b3c_s  = f"{b3c:.3f}{stars(p3c)}"
    b3ci_s = f"{b3ci:.3f}{stars(p3ci)}"

    results.append({
        'Outcome':       y,
        'Clusters':      clusters,
        'Obs':           obs,
        'M1_coef_post':  b1s,
        'M1_se_post':    f"{se1:.3f}",
        'M2_coef_base':  b2b_s,
        'M2_se_base':    f"{se2b:.3f}",
        'M2_coef_int':   b2i_s,
        'M2_se_int':     f"{se2i:.3f}",
        'M3_coef_post':  b3p_s,
        'M3_se_post':    f"{se3p:.3f}",
        'M3_coef_p_g':   b3i_s,
        'M3_se_p_g':     f"{se3i:.3f}",
        'M3_coef_covid': b3c_s,
        'M3_se_covid':   f"{se3c:.3f}",
        'M3_coef_c_g':   b3ci_s,
        'M3_se_c_g':     f"{se3ci:.3f}"
    })

# 9) Export to Excel
pd.DataFrame(results).to_excel(OUTPUT, index=False)
print("Done — wrote", OUTPUT)


