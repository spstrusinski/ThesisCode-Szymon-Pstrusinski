import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1) Load & clean column names
path = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_grouped_metrics.xlsx'
df = pd.read_excel(path, engine='openpyxl')

def clean_col(name):
    s = re.sub(r'[^0-9a-zA-Z]', '_', name)
    return re.sub(r'__+', '_', s).strip('_').lower()

df.columns = [clean_col(c) for c in df.columns]

# 2) Keep only liceum & technikum
df = df[df['typ_placowki'].isin(['liceum', 'technikum'])]

# 3) Rebuild identifiers & reform indicator
df['gmina_type'] = df['unnamed_4_level_0_typ_gminy'].isin(['gmina_miejska','miasto']).astype(int)
df['powiat_clean'] = df['unnamed_2_level_0_powiat_nazwa'].str.replace(r'_', '', regex=True)
df['gmina_powiat_id'] = df['gmina_type'].astype(str) + df['powiat_clean']

df['reform'] = 0
df.loc[(df['typ_placowki']=='liceum') & (df['year']>=2023), 'reform'] = 1
df.loc[(df['typ_placowki']=='technikum') & (df['year']>=2024), 'reform'] = 1

# 4) Compute urban–rural gaps
score_cols = [c for c in df.columns if c.endswith('_sredni_score_percent')]
score_cols += ['avg_basic_score', 'avg_advanced_score']
grp = df.groupby(['typ_placowki','year','gmina_type'])[score_cols].mean()
un = grp.unstack('gmina_type')
gap_df = pd.DataFrame({
    f"{var}_gap": un[(var,1)] - un[(var,0)]
    for var in score_cols
}).reset_index()

# 5) Identify proxy for Overall Basic Avg student count
basic_count_col = next(
    c for c in df.columns
    if 'dla_calego_egzaminu_dojrzalosci' in c and 'students' in c
)

# 6) Define subjects & their metrics
subjects = {
    'Math Advanced': {
        'Average Score (%)': 'math_poziom_advanced_m_sredni_score_percent',
        'Between-School SD': 'std_between_math_poziom_advanced_m_sredni_score_percent',
        'Student Count':     'math_poziom_advanced_m_students',
        'Urban-Rural Gap':   'math_poziom_advanced_m_sredni_score_percent_gap',
    },
    'Overall Basic Average': {
        'Average Basic Score': 'avg_basic_score',
        'Between-School SD':    'std_between_avg_basic_score',
        'Student Count':        basic_count_col,
        'Urban-Rural Gap':      'avg_basic_score_gap',
    },
    'Overall Advanced Average': {
        'Average Advanced Score': 'avg_advanced_score',
        'Between-School SD':       'std_between_avg_advanced_score',
        'Urban-Rural Gap':         'avg_advanced_score_gap',
    },
}

# 7) Plot each subject’s metrics
for subj, metrics in subjects.items():
    m = len(metrics)
    # layout: 2x2 for 4, 1x3 for 3
    if m == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()
    else:  # m == 3
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
        axes = axes.flatten()
    for ax, (label, col) in zip(axes, metrics.items()):
        # skip missing
        if col not in df.columns and col not in gap_df.columns:
            ax.set_visible(False)
            continue
        # choose data source
        if col.endswith('_gap'):
            data = gap_df
            grp2 = data.groupby(['year','typ_placowki'])[col]
        else:
            data = df
            grp2 = data.groupby(['year','typ_placowki'])[col]
        mean = grp2.mean().unstack('typ_placowki')
        sem  = grp2.sem().unstack('typ_placowki')
        years = mean.index.astype(int)
        ax.set_xticks(years)
        for typ in mean.columns:
            ax.plot(years, mean[typ], marker='o', label=typ)
            lo = mean[typ] - 1.96*sem[typ]
            hi = mean[typ] + 1.96*sem[typ]
            ax.fill_between(years, lo, hi, alpha=0.2)
        ax.axvline(2023, color='grey', linestyle='--')
        ax.axvline(2024, color='grey', linestyle='--')
        ax.set_title(label)
        ax.set_xlabel('Year')
        ax.set_ylabel(label)
        ax.legend(title='School type')
    fig.suptitle(f'Trends: {subj}', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()














import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1) Load & clean column names
path = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_grouped_metrics.xlsx'
df = pd.read_excel(path, engine='openpyxl')

def clean_col(name):
    s = re.sub(r'[^0-9a-zA-Z]', '_', name)
    return re.sub(r'__+', '_', s).strip('_').lower()

df.columns = [clean_col(c) for c in df.columns]

# 2) Filter to liceum & technikum
df = df[df['typ_placowki'].isin(['liceum', 'technikum'])]

# 3) Rebuild identifiers & reform indicator
df['gmina_type'] = df['unnamed_4_level_0_typ_gminy'] \
    .isin(['gmina_miejska','miasto']).astype(int)
df['powiat_clean'] = df['unnamed_2_level_0_powiat_nazwa'] \
    .str.replace(r'_', '', regex=True)
df['gmina_powiat_id'] = df['gmina_type'].astype(str) + df['powiat_clean']

df['reform'] = 0
df.loc[(df['typ_placowki']=='liceum') & (df['year']>=2023), 'reform'] = 1
df.loc[(df['typ_placowki']=='technikum') & (df['year']>=2024), 'reform'] = 1

# 4) Compute urban–rural gaps for score variables
score_cols = [c for c in df.columns if c.endswith('_sredni_score_percent')]
score_cols += ['avg_basic_score', 'avg_advanced_score']
grp = df.groupby(['typ_placowki','year','gmina_type'])[score_cols].mean()
un = grp.unstack('gmina_type')
gap_df = pd.DataFrame({
    f"{var}_gap": un[(var,1)] - un[(var,0)]
    for var in score_cols
}).reset_index()

# 5) Identify proxy for Overall Basic Avg student count
basic_count_col = next(
    c for c in df.columns
    if 'dla_calego_egzaminu_dojrzalosci' in c and 'students' in c
)

# 6) Define subjects & metrics
subjects = {
    'Math Advanced': {
        'Average Score (%)': 'math_poziom_advanced_m_sredni_score_percent',
        'Between-School SD': 'std_between_math_poziom_advanced_m_sredni_score_percent',
        'Student Count':     'math_poziom_advanced_m_students',
        'Urban-Rural Gap':   'math_poziom_advanced_m_sredni_score_percent_gap',
    },
    'Overall Basic Average': {
        'Average Basic Score': 'avg_basic_score',
        'Between-School SD':    'std_between_avg_basic_score',
        'Student Count':        basic_count_col,
        'Urban-Rural Gap':      'avg_basic_score_gap',
    },
    'Overall Advanced Average': {
        'Average Advanced Score': 'avg_advanced_score',
        'Between-School SD':       'std_between_avg_advanced_score',
        'Urban-Rural Gap':         'avg_advanced_score_gap',
    },
}

# 7) Create a single big figure with subplots 3x4
fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharex=False)
axes = axes.flatten()

plot_idx = 0
for subj, metrics in subjects.items():
    for label, col in metrics.items():
        ax = axes[plot_idx]
        # select data and grouping
        if col.endswith('_gap'):
            data = gap_df
            grp2 = data.groupby(['year','typ_placowki'])[col]
        else:
            data = df
            grp2 = data.groupby(['year','typ_placowki'])[col]
        # compute mean & sem
        mean = grp2.mean().unstack('typ_placowki')
        sem  = grp2.sem().unstack('typ_placowki')
        years = mean.index.astype(int)
        ax.set_xticks(years)
        # plot lines & CIs
        for typ in mean.columns:
            ax.plot(years, mean[typ], marker='o', label=typ)
            lo = mean[typ] - 1.96*sem[typ]
            hi = mean[typ] + 1.96*sem[typ]
            ax.fill_between(years, lo, hi, alpha=0.2)
        # reform lines
        ax.axvline(2023, color='grey', linestyle='--')
        ax.axvline(2024, color='grey', linestyle='--')
        # labels
        ax.set_title(f"{subj}: {label}", fontsize=10)
        ax.set_xlabel('Year')
        ax.set_ylabel(label)
        ax.legend(title='School type', fontsize=8)
        plot_idx += 1

# hide any unused subplot (e.g., 12th ax)
for i in range(plot_idx, len(axes)):
    axes[i].set_visible(False)

fig.suptitle('Trends for Selected Metrics', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




















import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1) Load & clean column names
path = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_grouped_metrics.xlsx'
df = pd.read_excel(path, engine='openpyxl')

def clean_col(name):
    s = re.sub(r'[^0-9a-zA-Z]', '_', name)
    return re.sub(r'__+', '_', s).strip('_').lower()

df.columns = [clean_col(c) for c in df.columns]

# 2) Keep only liceum & technikum
df = df[df['typ_placowki'].isin(['liceum', 'technikum'])]

# 3) Rebuild identifiers & reform indicator
df['gmina_type']   = df['unnamed_4_level_0_typ_gminy']\
                        .isin(['gmina_miejska','miasto']).astype(int)
df['powiat_clean'] = df['unnamed_2_level_0_powiat_nazwa']\
                        .str.replace(r'_', '', regex=True)
df['gmina_powiat_id'] = df['gmina_type'].astype(str) + df['powiat_clean']
df['reform'] = 0
df.loc[(df['typ_placowki']=='liceum')    & (df['year']>=2023), 'reform'] = 1
df.loc[(df['typ_placowki']=='technikum') & (df['year']>=2024), 'reform'] = 1

# 4) Compute urban–rural gaps for scores
score_cols = [c for c in df.columns if c.endswith('_sredni_score_percent')]
score_cols += ['avg_basic_score', 'avg_advanced_score']
grp = df.groupby(['typ_placowki','year','gmina_type'])[score_cols].mean()
un = grp.unstack('gmina_type')
gap_df = pd.DataFrame({
    f"{var}_gap": un[(var,1)] - un[(var,0)]
    for var in score_cols
}).reset_index()

# 5) Identify proxy column for overall basic count
basic_count_col = next(
    c for c in df.columns
    if 'dla_calego_egzaminu_dojrzalosci' in c and 'students' in c
)

# 6) Ordered metrics per subject
subjects = {
    'Math Advanced': [
        ('Average Score (%)',
         'math_poziom_advanced_m_sredni_score_percent'),
        ('Between‑School SD',
         'std_between_math_poziom_advanced_m_sredni_score_percent'),
        ('Urban‑Rural Gap',
         'math_poziom_advanced_m_sredni_score_percent_gap'),
        ('Student Count',
         'math_poziom_advanced_m_students'),
    ],
    'Overall Basic Average': [
        ('Average Basic Score', 'avg_basic_score'),
        ('Between‑School SD',       'std_between_avg_basic_score'),
        ('Urban‑Rural Gap',         'avg_basic_score_gap'),
        ('Student Count',           basic_count_col),
    ],
    'Overall Advanced Average': [
        ('Average Advanced Score', 'avg_advanced_score'),
        ('Between‑School SD',        'std_between_avg_advanced_score'),
        ('Urban‑Rural Gap',          'avg_advanced_score_gap'),
        # no student count here, omit 4th panel
    ],
}

# 7) Create big 3×4 figure
nrows, ncols = 3, 4
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(20, 15),
                         sharex=False)
axes = axes.flatten()

plot_idx = 0
for row_idx, (subj, metrics) in enumerate(subjects.items()):
    for col_idx, (label, col) in enumerate(metrics):
        ax = axes[row_idx*ncols + col_idx]
        # choose data
        if col.endswith('_gap'):
            data = gap_df
            grp2 = data.groupby(['year','typ_placowki'])[col]
        else:
            data = df
            grp2 = data.groupby(['year','typ_placowki'])[col]
        mean = grp2.mean().unstack('typ_placowki')
        sem  = grp2.sem().unstack('typ_placowki')
        years = mean.index.astype(int)
        ax.set_xticks(years)
        # plot lines & CIs
        for typ in mean.columns:
            ax.plot(years, mean[typ], marker='o', label=typ)
            lo = mean[typ] - 1.96*sem[typ]
            hi = mean[typ] + 1.96*sem[typ]
            ax.fill_between(years, lo, hi, alpha=0.2)
        # reform verticals
        ax.axvline(2023, color='grey', linestyle='--')
        ax.axvline(2024, color='grey', linestyle='--')
        # labels
        ax.set_title(f"{label}", fontsize=10)
        ax.set_xlabel('Year')
        ax.set_ylabel(label)
        if row_idx == 0 and col_idx == 0:
            # only one legend overall
            ax.legend(title='School type', fontsize=8)

    # hide unused panels in last row if any
    if len(metrics) < ncols:
        for empty_j in range(len(metrics), ncols):
            axes[row_idx*ncols + empty_j].set_visible(False)

# 8) Add bold column headers atop the grid
col_titles = ["Average Score", "Between‑School SD",
              "Urban‑Rural Gap", "Student Count"]
for j, title in enumerate(col_titles):
    x = (j + 0.5) / ncols
    fig.text(x, 0.97, title, ha='center',
             fontsize=14, fontweight='bold')

# 9) Add bold row labels to the left of each row
row_titles = list(subjects.keys())
for i, title in enumerate(row_titles):
    # center of each row in figure coords
    y = 1 - (i + 0.5) / nrows
    fig.text(0.02, y, title, va='center',
             fontsize=14, fontweight='bold')

fig.suptitle('Trends for Selected Metrics', fontsize=16)
fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
plt.show()


























import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1) File path
path = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_grouped_metrics.xlsx'

# 2) Load & clean names
df = pd.read_excel(path, engine='openpyxl')
def clean_col(name):
    s = re.sub(r'[^0-9a-zA-Z]', '_', name)
    return re.sub(r'__+', '_', s).strip('_').lower()
df.columns = [clean_col(c) for c in df.columns]

# 3) Keep only liceum & technikum
df = df[df['typ_placowki'].isin(['liceum','technikum'])]

# 4) Build the urban-rural gap DF (needed for the gap metric)
score_cols = [c for c in df.columns if c.endswith('_sredni_score_percent')]
score_cols += ['avg_basic_score','avg_advanced_score']
grp = df.groupby(['typ_placowki','year','gmina_type'])[score_cols].mean()
un  = grp.unstack('gmina_type')
gap_df = pd.DataFrame({
    f"{var}_gap": un[(var,1)] - un[(var,0)]
    for var in score_cols
}).reset_index()

# 5) Proxy for Overall Basic student‐count
basic_count_col = next(
    c for c in df.columns
    if 'dla_calego_egzaminu_dojrzalosci' in c and 'students' in c
)

# 6) Define subj × metrics
subjects = [
    ("Advanced Math", {
        "Average Score": "math_poziom_advanced_m_sredni_score_percent",
        "Between-School SD": "std_between_math_poziom_advanced_m_sredni_score_percent",
        "Student Count": "math_poziom_advanced_m_students",
        "Urban-Rural Gap": "math_poziom_advanced_m_sredni_score_percent_gap",
    }),
    ("Overall Basic Average", {
        "Average Score": "avg_basic_score",
        "Between-School SD": "std_between_avg_basic_score",
        "Student Count": basic_count_col,
        "Urban-Rural Gap": "avg_basic_score_gap",
    }),
    ("Overall Advanced Average", {
        "Average Score": "avg_advanced_score",
        "Between-School SD": "std_between_avg_advanced_score",
        "Urban-Rural Gap": "avg_advanced_score_gap",
    })
]

# 7) Compute year‐by‐year differences + CIs
diffs = {}
for subj_label, metrics in subjects:
    diffs[subj_label] = {}
    for metric, col in metrics.items():
        # pick DataFrame + groupby
        if col.endswith('_gap'):
            data = gap_df
            g = data.groupby(['year','typ_placowki'])[col]
        else:
            data = df
            g = data.groupby(['year','typ_placowki'])[col]
        mean = g.mean().unstack('typ_placowki')
        sem  = g.sem().unstack('typ_placowki')
        # difference + CI
        diff     = mean['liceum'] - mean['technikum']
        se_diff  = np.sqrt(sem['liceum']**2 + sem['technikum']**2)
        lower    = diff - 1.96*se_diff
        upper    = diff + 1.96*se_diff
        diffs[subj_label][metric] = pd.DataFrame({
            'year':  diff.index.astype(int),
            'diff':  diff.values,
            'lower': lower.values,
            'upper': upper.values
        })

# 8) Plot big grid
metrics_list = ["Average Score","Between-School SD","Student Count","Urban-Rural Gap"]
n_rows = len(metrics_list)
n_cols = len(subjects)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols,4*n_rows))
for c, (subj_label, _) in enumerate(subjects):
    for r, metric in enumerate(metrics_list):
        ax = axes[r, c]
        dfp = diffs[subj_label].get(metric)
        # if no panel (e.g. Student Count for Overall Advanced), blank
        if dfp is None:
            ax.axis('off')
            continue
        yrs = dfp['year']
        ax.plot(yrs, dfp['diff'], 'o-', color='black')
        ax.fill_between(yrs, dfp['lower'], dfp['upper'], color='gray', alpha=0.3)
        ax.axhline(0, color='red', linestyle='--')
        ax.axvline(2023, color='grey', linestyle='--')
        ax.axvline(2024, color='grey', linestyle='--')
        if r==0:
            ax.set_title(subjects[c][0], fontweight='bold')
        if c==0:
            ax.set_ylabel(metric, fontweight='bold')
        ax.set_xticks(yrs)
        if r==n_rows-1:
            ax.set_xlabel('Year')
plt.tight_layout()
plt.show()




















import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import statsmodels.formula.api as smf

# 1) Load & clean column names
path = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_grouped_metrics.xlsx'
df = pd.read_excel(path, engine='openpyxl')
def clean_col(name):
    s = re.sub(r'[^0-9a-zA-Z]', '_', name)
    return re.sub(r'__+', '_', s).strip('_').lower()
df.columns = [clean_col(c) for c in df.columns]

# 2) Filter to liceum & technikum
df = df[df['typ_placowki'].isin(['liceum', 'technikum'])].copy()

# 3) Build identifiers & reform info
df['gmina_type']   = df['unnamed_4_level_0_typ_gminy']\
                        .isin(['gmina_miejska','miasto']).astype(int)
df['powiat_clean'] = df['unnamed_2_level_0_powiat_nazwa']\
                        .str.replace(r'_', '', regex=True)
df['gmina_powiat_id'] = df['gmina_type'].astype(str) + df['powiat_clean']
df['reform_year'] = np.where(df['typ_placowki']=='liceum', 2023, 2024)
df['event_time']  = df['year'] - df['reform_year']

# 4) Create event-time dummies (omit e_m1)
df['e_m2'] = (df['event_time'] == -2).astype(int)
df['e_m1'] = (df['event_time'] == -1).astype(int)
df['e0']  = (df['event_time'] ==  0).astype(int)
df['e1']  = (df['event_time'] ==  1).astype(int)
df['e2']  = (df['event_time'] ==  2).astype(int)
event_terms = ['e_m2','e0','e1','e2']
time_map    = {'e_m2': -2, 'e0': 0, 'e1': 1, 'e2': 2}

# 5) Compute urban-rural gaps
score_cols = [c for c in df.columns if c.endswith('_sredni_score_percent')]
score_cols += ['avg_basic_score','avg_advanced_score']
grp = df.groupby(['typ_placowki','year','gmina_type'])[score_cols].mean()
un  = grp.unstack('gmina_type')
gap_df = pd.DataFrame({f"{var}_gap": un[(var,1)]-un[(var,0)]
                       for var in score_cols}).reset_index()

# 6) Identify proxy for basic student count
basic_count_col = next(c for c in df.columns
                       if 'dla_calego_egzaminu_dojrzalosci' in c
                       and 'students' in c)

# 7) Define subjects & metrics, dropping Advanced Math "Average Score"
subjects = [
    ("Advanced Math", {
        "Between-School SD": "std_between_math_poziom_advanced_m_sredni_score_percent",
        "Student Count":     "math_poziom_advanced_m_students",
        "Urban-Rural Gap":   "math_poziom_advanced_m_sredni_score_percent_gap",
    }),
    ("Overall Basic Average", {
        "Average Score":      "avg_basic_score",
        "Between-School SD":  "std_between_avg_basic_score",
        "Student Count":      basic_count_col,
        "Urban-Rural Gap":    "avg_basic_score_gap",
    }),
    ("Overall Advanced Average", {
        "Average Score":     "avg_advanced_score",
        "Between-School SD": "std_between_avg_advanced_score",
        "Urban-Rural Gap":   "avg_advanced_score_gap",
    }),
]

# 8) Run event-study regressions and collect coefficients
records = []
for subj_label, metrics in subjects:
    for metric_label, col in metrics.items():
        if col.endswith('_gap'):
            data = gap_df.merge(df[['typ_placowki','year'] + event_terms],
                                on=['typ_placowki','year'], how='left')
            fe = ""  # no gmina_powiat FE for gap, only year
        else:
            data = df.copy()
            fe = " + C(gmina_powiat_id)"
        formula = f"{col} ~ {' + '.join(event_terms)} + C(year){fe}"
        mod = smf.ols(formula, data=data).fit(cov_type='HC1')
        ci = mod.conf_int().loc[event_terms]
        for term in event_terms:
            records.append({
                'Subject':   subj_label,
                'Metric':    metric_label,
                'EventTime': time_map[term],
                'Coef':      mod.params.get(term, np.nan),
                'CI_low':    ci.loc[term,0],
                'CI_high':   ci.loc[term,1]
            })
es_df = pd.DataFrame(records)

# 9) Plot combined event-study figure (4 rows × 3 cols)
metric_order = ["Average Score","Between-School SD","Student Count","Urban-Rural Gap"]
n_rows = len(metric_order)
n_cols = len(subjects)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True)

for col_idx, (subj_label, _) in enumerate(subjects):
    for row_idx, metric in enumerate(metric_order):
        ax = axes[row_idx, col_idx]
        sub = es_df[(es_df.Subject==subj_label) & (es_df.Metric==metric)]
        if sub.empty:
            ax.axis('off')
            continue
        x = sub.EventTime
        y = sub.Coef
        lo = sub.CI_low
        hi = sub.CI_high
        ax.plot(x, y, marker='o')
        ax.fill_between(x, lo, hi, alpha=0.2)
        ax.axhline(0, color='grey', linewidth=1)
        ax.axvline(0, color='grey', linestyle='--')
        if row_idx==0:
            ax.set_title(subj_label, fontsize=14, fontweight='bold')
        if col_idx==0:
            ax.set_ylabel(metric, fontsize=14, fontweight='bold')
        ax.set_xticks([-2,0,1,2])
        ax.set_xlabel('Event time')
fig.tight_layout()
plt.show()












