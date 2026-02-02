import pandas as pd
import numpy as np
from scipy import stats

# Load file
file_path = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\Working_with_SD_and_Stats_All_OKE_with_Unweighted_Weighted_SD.xlsx"
df = pd.read_excel(file_path)

# --- Handle special dummy variables ---
df["gmina_class"] = df["gmina_class"].map({"urban": 1, "rural": 0})
df["czy publiczna"] = df["czy publiczna"].map({"Tak": 1, "Nie": 0})

# Keep only numeric columns for most of the analysis
df_numeric = df.select_dtypes(include=[np.number])
df_all = df_numeric.copy()
df_oke3 = df[df["ID OKE"] == 3][df_numeric.columns]

# Helper function for summary stats
def get_stats(series):
    count = series.count()
    mean = series.mean()
    std = series.std(ddof=1)
    se = std / np.sqrt(count) if count > 0 else np.nan
    return count, mean, se, std, series.min(), series.max()

# Function to compare two groups
def format_diff(mean_all, se_all, mean_oke3, se_oke3):
    if pd.isna(mean_all) or pd.isna(mean_oke3):
        return ""
    diff = mean_all - mean_oke3
    se_diff = np.sqrt(se_all ** 2 + se_oke3 ** 2)
    if pd.isna(se_diff) or se_diff == 0:
        return f"{diff:.3f} (NA)"
    t = diff / se_diff
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = ""
    return f"{diff:.3f}{stars} ({se_diff:.3f})"

# Start building summary
summary_rows = []

# Identify special variable pairs:
# SD_XXX in OKE 3 ↔ _Unweighted_SD in ALL
sd_cols = [col for col in df_oke3.columns if col.startswith("SD_")]
unweighted_sd_cols = [col for col in df_all.columns if col.endswith("_Unweighted_SD")]
unweighted_pairs = {
    sd_col: next((uc for uc in unweighted_sd_cols if uc.split("_")[0] in sd_col), None)
    for sd_col in sd_cols
    if any(sd_col.startswith(prefix) for prefix in ["SD_All", "SD_Rural", "SD_Urban"])
}

# Weighted_SD XXX in OKE 3 ↔ plain _SD in ALL
weighted_sd_cols = [col for col in df_oke3.columns if col.endswith("_Weighted_SD")]
sd_base_cols = [col for col in df_all.columns if col.endswith("_SD") and not col.endswith("_Unweighted_SD") and not col.endswith("_Weighted_SD")]
weighted_pairs = {
    w_col: next((sc for sc in sd_base_cols if sc.split("_")[0] in w_col), None)
    for w_col in weighted_sd_cols
}

# Process normal variables
for col in df_numeric.columns:
    count_all, mean_all, se_all, sd_all, min_all, max_all = get_stats(df_all[col])
    count_oke3, mean_oke3, se_oke3, sd_oke3, min_oke3, max_oke3 = get_stats(df_oke3[col])

    # Default diff
    diff_str = format_diff(mean_all, se_all, mean_oke3, se_oke3)

    # Special handling for special pairs
    if col in unweighted_pairs:
        compare_col = unweighted_pairs[col]
        mean_all = df_all[compare_col].mean()
        se_all = df_all[compare_col].std(ddof=1) / np.sqrt(df_all[compare_col].count())
        mean_oke3 = df_oke3[col].mean()
        se_oke3 = df_oke3[col].std(ddof=1) / np.sqrt(df_oke3[col].count())
        diff_str = format_diff(mean_all, se_all, mean_oke3, se_oke3)
    elif col in weighted_pairs:
        compare_col = weighted_pairs[col]
        mean_all = df_all[compare_col].mean()
        se_all = df_all[compare_col].std(ddof=1) / np.sqrt(df_all[compare_col].count())
        mean_oke3 = df_oke3[col].mean()
        se_oke3 = df_oke3[col].std(ddof=1) / np.sqrt(df_oke3[col].count())
        diff_str = format_diff(mean_all, se_all, mean_oke3, se_oke3)

    summary_rows.append({
        "Variable": col,
        "Count (All)": count_all,
        "Mean (All)": f"{mean_all:.3f}" if not pd.isna(mean_all) else "",
        "SD (All)": f"{sd_all:.3f}" if not pd.isna(sd_all) else "",
        "Min (All)": f"{min_all:.3f}" if not pd.isna(min_all) else "",
        "Max (All)": f"{max_all:.3f}" if not pd.isna(max_all) else "",
        "SE (All)": f"{se_all:.3f}" if not pd.isna(se_all) else "",

        "Count (OKE 3)": count_oke3,
        "Mean (OKE 3)": f"{mean_oke3:.3f}" if not pd.isna(mean_oke3) else "",
        "SD (OKE 3)": f"{sd_oke3:.3f}" if not pd.isna(sd_oke3) else "",
        "Min (OKE 3)": f"{min_oke3:.3f}" if not pd.isna(min_oke3) else "",
        "Max (OKE 3)": f"{max_oke3:.3f}" if not pd.isna(max_oke3) else "",
        "SE (OKE 3)": f"{se_oke3:.3f}" if not pd.isna(se_oke3) else "",

        "Difference (All - OKE 3)": diff_str
    })

# Export to Excel
summary_df = pd.DataFrame(summary_rows)
output_path = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\Summary_Full_Stats_OKE_vs_OKE3.xlsx"
summary_df.to_excel(output_path, index=False)

print("✅ Full summary with counts, means, SDs, differences, and significance saved.")







import pandas as pd
import numpy as np
from scipy import stats

# Load full dataset
file_path = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\Working_with_SD_and_Stats_All_OKE_with_Unweighted_Weighted_SD.xlsx"
df = pd.read_excel(file_path)

# List of all 18 variables from your message
all_sd_vars = [
    "SD_Math_Overall", "SD_Polish_Overall", "SD_English_Overall",
    "SD_Math_Rural", "SD_Polish_Rural", "SD_English_Rural",
    "SD_Math_Urban", "SD_Polish_Urban", "SD_English_Urban",
    "All_Math_SD", "All_Polish_SD", "All_English_SD",
    "Rural_Math_SD", "Rural_Polish_SD", "Rural_English_SD",
    "Urban_Math_SD", "Urban_Polish_SD", "Urban_English_SD"
]

# Define mapping for variable pairing
unweighted_pairs = {
    "SD_Math_Overall": "All_Math_Unweighted_SD",
    "SD_Polish_Overall": "All_Polish_Unweighted_SD",
    "SD_English_Overall": "All_English_Unweighted_SD",
    "SD_Math_Rural": "Rural_Math_Unweighted_SD",
    "SD_Polish_Rural": "Rural_Polish_Unweighted_SD",
    "SD_English_Rural": "Rural_English_Unweighted_SD",
    "SD_Math_Urban": "Urban_Math_Unweighted_SD",
    "SD_Polish_Urban": "Urban_Polish_Unweighted_SD",
    "SD_English_Urban": "Urban_English_Unweighted_SD"
}

weighted_pairs = {
    "All_OKE3_Math_Weighted_SD": "All_Math_SD",
    "All_OKE3_Polish_Weighted_SD": "All_Polish_SD",
    "All_OKE3_English_Weighted_SD": "All_English_SD",
    "Rural_OKE3_Math_Weighted_SD": "Rural_Math_SD",
    "Rural_OKE3_Polish_Weighted_SD": "Rural_Polish_SD",
    "Rural_OKE3_English_Weighted_SD": "Rural_English_SD",
    "Urban_OKE3_Math_Weighted_SD": "Urban_Math_SD",
    "Urban_OKE3_Polish_Weighted_SD": "Urban_Polish_SD",
    "Urban_OKE3_English_Weighted_SD": "Urban_English_SD"
}

# Combine all pairs into one lookup dictionary: OKE3_col → ALL_col
all_pairs = {**unweighted_pairs, **weighted_pairs}

# Filter to numeric columns for safety
df_numeric = df.select_dtypes(include=[np.number])
df_oke3 = df[df["ID OKE"] == 3]

# Helper for formatting diff
def format_diff(mean_all, se_all, mean_oke3, se_oke3):
    if pd.isna(mean_all) or pd.isna(mean_oke3):
        return ""
    diff = mean_all - mean_oke3
    se_diff = np.sqrt(se_all ** 2 + se_oke3 ** 2)
    if pd.isna(se_diff) or se_diff == 0:
        return f"{diff:.3f} (NA)"
    t_stat = diff / se_diff
    p = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{diff:.3f}{stars} ({se_diff:.3f})"

# Compute summary
summary_rows = []
for oke3_col, all_col in all_pairs.items():
    # Ensure both columns exist
    if oke3_col not in df_oke3.columns or all_col not in df_numeric.columns:
        continue

    all_values = df_numeric[all_col].dropna()
    oke3_values = df_oke3[oke3_col].dropna()

    mean_all = all_values.mean()
    se_all = all_values.std(ddof=1) / np.sqrt(len(all_values)) if len(all_values) > 0 else np.nan
    mean_oke3 = oke3_values.mean()
    se_oke3 = oke3_values.std(ddof=1) / np.sqrt(len(oke3_values)) if len(oke3_values) > 0 else np.nan

    summary_rows.append({
        "Variable (All)": all_col,
        "Variable (OKE 3)": oke3_col,
        "Mean (All)": f"{mean_all:.3f}" if not pd.isna(mean_all) else "",
        "SE (All)": f"{se_all:.3f}" if not pd.isna(se_all) else "",
        "Mean (OKE 3)": f"{mean_oke3:.3f}" if not pd.isna(mean_oke3) else "",
        "SE (OKE 3)": f"{se_oke3:.3f}" if not pd.isna(se_oke3) else "",
        "Difference (All - OKE 3)": format_diff(mean_all, se_all, mean_oke3, se_oke3)
    })

# Export to Excel
summary_df = pd.DataFrame(summary_rows)
output_path = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\Summary_Weighted_Unweighted_SD_Comparison.xlsx"
summary_df.to_excel(output_path, index=False)

print("✅ SD summary (18 variables) with custom pairing saved.")







# === Load summary and main data ===
summary_path = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\Summary_Full_Stats_OKE_vs_OKE3.xlsx"
data_path = r"C:\Users\Pstrrr...ICYK\Desktop\Teza\Egzamin Ósmoklasistygimnazjalisty\Workingfolder\Working_with_SD_and_Stats_All_OKE_with_Unweighted_Weighted_SD.xlsx"

summary_df = pd.read_excel(summary_path)
df = pd.read_excel(data_path)

# === Clean column names ===
df.columns = df.columns.str.strip()

# === Dummy encode 'rodzaj placówki' ===
df["rodzaj placówki"] = df["rodzaj placówki"].map(lambda x: 1 if x == "dla młodzieży" else 0)
df["rodzaj placówki"] = pd.to_numeric(df["rodzaj placówki"], errors="coerce")

# === Convert Stanina columns to numeric ===
stanina_cols = ["Stanina-polski", "Stanina-matematyka", "Stanina-angielski"]
for col in stanina_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# === Ensure 'czy publiczna' and 'gmina_class' are binary numeric ===
df["czy publiczna"] = df["czy publiczna"].map({"Tak": 1, "Nie": 0})
df["czy publiczna"] = pd.to_numeric(df["czy publiczna"], errors="coerce")

df["gmina_class"] = df["gmina_class"].map({"urban": 1, "rural": 0})
df["gmina_class"] = pd.to_numeric(df["gmina_class"], errors="coerce")

# === Helper: format difference with significance stars ===
def format_diff(mean_all, se_all, mean_oke3, se_oke3):
    if pd.isna(mean_all) or pd.isna(mean_oke3):
        return ""
    diff = mean_all - mean_oke3
    se_diff = np.sqrt(se_all ** 2 + se_oke3 ** 2)
    if pd.isna(se_diff) or se_diff == 0:
        return f"{diff:.3f} (NA)"
    t = diff / se_diff
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{diff:.3f}{stars} ({se_diff:.3f})"

# === Helper: build summary row ===
def build_summary_row(var, df_all, df_oke3):
    series_all = df_all[var] if var in df_all.columns else pd.Series(dtype='float64')
    series_oke3 = df_oke3[var] if var in df_oke3.columns else pd.Series(dtype='float64')

    count_all = series_all.count()
    mean_all = series_all.mean()
    std_all = series_all.std(ddof=1)
    se_all = std_all / np.sqrt(count_all) if count_all > 0 else np.nan
    min_all = series_all.min()
    max_all = series_all.max()

    count_oke3 = series_oke3.count()
    mean_oke3 = series_oke3.mean()
    std_oke3 = series_oke3.std(ddof=1)
    se_oke3 = std_oke3 / np.sqrt(count_oke3) if count_oke3 > 0 else np.nan
    min_oke3 = series_oke3.min()
    max_oke3 = series_oke3.max()

    diff_str = ""
    if not pd.isna(mean_all) and not pd.isna(mean_oke3):
        diff_str = format_diff(mean_all, se_all, mean_oke3, se_oke3)

    return {
        "Variable": var,
        "Count (All)": count_all if count_all > 0 else "",
        "Mean (All)": f"{mean_all:.3f}" if not pd.isna(mean_all) else "",
        "SD (All)": f"{std_all:.3f}" if not pd.isna(std_all) else "",
        "Min (All)": f"{min_all:.3f}" if not pd.isna(min_all) else "",
        "Max (All)": f"{max_all:.3f}" if not pd.isna(max_all) else "",
        "SE (All)": f"{se_all:.3f}" if not pd.isna(se_all) else "",

        "Count (OKE 3)": count_oke3 if count_oke3 > 0 else "",
        "Mean (OKE 3)": f"{mean_oke3:.3f}" if not pd.isna(mean_oke3) else "",
        "SD (OKE 3)": f"{std_oke3:.3f}" if not pd.isna(std_oke3) else "",
        "Min (OKE 3)": f"{min_oke3:.3f}" if not pd.isna(min_oke3) else "",
        "Max (OKE 3)": f"{max_oke3:.3f}" if not pd.isna(max_oke3) else "",
        "SE (OKE 3)": f"{se_oke3:.3f}" if not pd.isna(se_oke3) else "",

        "Difference (All - OKE 3)": diff_str
    }

# === Build rows for the extra variables ===
extra_vars = ["rodzaj placówki", "Stanina-polski", "Stanina-matematyka", "Stanina-angielski"]
df_all = df.copy()
df_oke3 = df[df["ID OKE"] == 3]
rows_to_add = [build_summary_row(var, df_all, df_oke3) for var in extra_vars]

# === Append to summary and save ===
summary_df = pd.concat([summary_df, pd.DataFrame(rows_to_add)], ignore_index=True)
summary_df.to_excel(summary_path, index=False)

print("✅ Final summary updated with 'rodzaj placówki' and Stanina variables.")






# Ensure column names are clean
df.columns = df.columns.str.strip()

# Filter to OKE 3
oke3_df = df[df["ID OKE"] == 3]

# Calculate percentage from dummy (1 = "dla młodzieży")
total = oke3_df["rodzaj placówki"].notna().sum()
count_youth = oke3_df["rodzaj placówki"].sum()

percentage = (count_youth / total) * 100 if total > 0 else 0

print(f"✅ 'dla młodzieży' in OKE 3: {int(count_youth)} out of {int(total)} ({percentage:.2f}%)")








