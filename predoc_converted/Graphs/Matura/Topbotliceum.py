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

# 2) Define subjects mapping
subjects = {
    'Advanced Math': 'math_poziom_advanced_m_sredni_score_percent',
    'Overall Basic Average': 'avg_basic_score',
    'Overall Advanced Average':'avg_advanced_score'
}

# 3) Copy for “all” schools
df_all = df.copy()

# 4) Quantiles
quantiles = [0.15, 0.50, 0.85]
years = sorted(df_all['year'].unique())
n_boot = 200

# 5) Helper to compute quantiles + bootstrap CIs
def compute_qci(data, col):
    recs = []
    for yr in years:
        arr = data.loc[data['year']==yr, col].dropna().values
        if len(arr)==0:
            recs.append([yr] + [np.nan]*len(quantiles)*3)
            continue
        qs = np.quantile(arr, quantiles)
        boots = np.array([
            np.quantile(np.random.choice(arr, len(arr), True), quantiles)
            for _ in range(n_boot)
        ])
        lows = np.percentile(boots, 2.5, axis=0)
        highs= np.percentile(boots, 97.5, axis=0)
        recs.append([yr, *qs, *lows, *highs])
    cols = ['year'] + [f'q{int(q*100)}' for q in quantiles] \
                 + [f'low{int(q*100)}' for q in quantiles] \
                 + [f'high{int(q*100)}' for q in quantiles]
    return pd.DataFrame(recs, columns=cols)

# 6) Precompute for liceum, technikum, all
dfs = {}
for grp,color in [('liceum','blue'),('technikum','orange')]:
    dfs[grp] = {subj: compute_qci(df_all[df_all['typ_placowki']==grp], col)
                for subj,col in subjects.items()}
dfs['all'] = {subj: compute_qci(df_all, col) for subj,col in subjects.items()}

# 7) Plot 2×3 grid, tighter spacing, with legends
fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)

# Row 0: liceum & technikum
for j,(subj,_) in enumerate(subjects.items()):
    ax = axes[0,j]
    for grp,color in [('liceum','blue'),('technikum','orange')]:
        dfq = dfs[grp][subj]
        yrs = dfq['year'].astype(int)
        ax.set_xticks(yrs)
        for q,ls in zip([15,50,85], ['-','--',':']):
            ax.plot(yrs, dfq[f'q{q}'], linestyle=ls,
                    color=color, marker='o',
                    label=f'{grp.title()} {q}th pct')
            ax.fill_between(yrs,
                            dfq[f'low{q}'],
                            dfq[f'high{q}'],
                            color=color, alpha=0.2)
    ax.axvline(2023, color='grey', linestyle='--')
    ax.axvline(2024, color='grey', linestyle='--')
    ax.set_title(subj, fontsize=14, fontweight='bold')
    if j==0:
        ax.set_ylabel('Liceum & Technikum', fontsize=12, fontweight='bold')
# unified legend for row 0
axes[0,2].legend(loc='upper left', bbox_to_anchor=(1.02,1))

# Row 1: all schools
for j,(subj,_) in enumerate(subjects.items()):
    ax = axes[1,j]
    dfq = dfs['all'][subj]
    yrs = dfq['year'].astype(int)
    ax.set_xticks(yrs)
    for q,color in zip([15,50,85], ['green','grey','red']):
        ax.plot(yrs, dfq[f'q{q}'], '-', color=color, marker='o',
                label=f'All {q}th pct')
        ax.fill_between(yrs,
                        dfq[f'low{q}'],
                        dfq[f'high{q}'],
                        color=color, alpha=0.2)
    ax.axvline(2023, color='grey', linestyle='--')
    ax.axvline(2024, color='grey', linestyle='--')
    if j==0:
        ax.set_ylabel('All Schools', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year')
# unified legend for row 1
axes[1,2].legend(loc='upper left', bbox_to_anchor=(1.02,1))

# adjust spacing
fig.subplots_adjust(wspace=0.2, hspace=0.3, right=0.85)
plt.tight_layout()
plt.show()
