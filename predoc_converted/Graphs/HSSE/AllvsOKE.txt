import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 1) Load & clean
path = r'C:\Users\Pstrrr...ICYK\Desktop\Teza\Matury\Pomoc_dla_Szymona_grouped_metrics.xlsx'
df = pd.read_excel(path, engine='openpyxl')
def clean_col(name):
    s = re.sub(r'[^0-9a-zA-Z]', '_', name)
    return re.sub(r'__+', '_', s).strip('_').lower()
df.columns = [clean_col(c) for c in df.columns]

# 2) Subjects and year positions
subjects = {
    'Advanced Math':            'math_poziom_advanced_m_sredni_score_percent',
    'Overall Basic Average':    'avg_basic_score',
    'Overall Advanced Average': 'avg_advanced_score'
}
years = sorted(df['year'].unique())
xpos  = np.arange(len(years))

# 3) IQR summary function
def summary_iqr(data, col):
    g   = data.groupby('year')[col]
    med = g.median().reindex(years).values
    q1  = g.quantile(0.25).reindex(years).values
    q3  = g.quantile(0.75).reindex(years).values
    return med, q1, q3

palette = {'liceum':'C0', 'technikum':'C1'}

# 4) Plotting
fig, axes = plt.subplots(2, 3, figsize=(18,10), sharex=True)

for j, (title, col) in enumerate(subjects.items()):
    # Top: IQR ribbons per school type
    top_ax = axes[0, j]
    for typ in ['liceum','technikum']:
        sub = df[df['typ_placowki']==typ]
        med, q1, q3 = summary_iqr(sub, col)
        top_ax.plot(xpos, med, color=palette[typ], lw=2, label=typ.title())
        top_ax.fill_between(xpos, q1, q3, color=palette[typ], alpha=0.3)

    top_ax.axvline(x=2, color='k', ls='--')
    top_ax.axvline(x=3, color='k', ls='--')
    top_ax.set_title(title, fontsize=14, fontweight='bold')
    if j==0:
        top_ax.set_ylabel('Liceum & Technikum', fontsize=12, fontweight='bold')
    if j==2:
        top_ax.legend(title='School type', bbox_to_anchor=(1.05,1))

    # Bottom: single boxplot for all schools
    bot_ax = axes[1, j]
    sns.boxplot(
        data=df, x='year', y=col,
        showfliers=True,
        color='lightgray',
        ax=bot_ax
    )
    bot_ax.axvline(x=2, color='k', ls='--')
    bot_ax.axvline(x=3, color='k', ls='--')
    bot_ax.set_xticks(xpos)
    bot_ax.set_xticklabels(years)
    bot_ax.set_ylim(0, 100)
    if j==0:
        bot_ax.set_ylabel('All schools', fontsize=12, fontweight='bold')
    bot_ax.set_xlabel('Year')

plt.tight_layout()
plt.show()
