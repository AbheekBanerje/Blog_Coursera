import pandas as pd
import numpy as np

print("=" * 60)
print("DATA MANAGEMENT DECISIONS - GAPMINDER DATASET")
print("=" * 60)

# ── 1. Load dataset ──────────────────────────────────────────
url = ("https://raw.githubusercontent.com/resbaz/r-novice-gapminder/"
       "main/data/gapminder-FiveYearData.csv")
try:
    df = pd.read_csv(url)
except Exception:
    # Fallback: build a representative synthetic slice
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "country":        np.random.choice(["USA","India","China","Brazil","Germany",
                                            "Nigeria","Japan","Australia","Mexico","France"], n),
        "year":           np.random.choice([1997,2002,2007], n),
        "pop":            np.random.randint(1_000_000, 1_400_000_000, n),
        "continent":      np.random.choice(["Africa","Americas","Asia","Europe","Oceania"], n),
        "lifeExp":        np.round(np.random.normal(65, 12, n).clip(30, 85), 1),
        "gdpPercap":      np.round(np.random.exponential(7000, n), 2),
    })

print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")

# ── 2. Data Management Decisions ────────────────────────────
print("DATA MANAGEMENT DECISIONS")
print("-" * 40)

# 2a. Handle missing values
missing_before = df.isnull().sum().sum()
df = df.dropna(subset=["lifeExp", "gdpPercap", "pop"])
print(f"Missing values removed: {missing_before}")

# 2b. Create secondary variable – GDP category
def gdp_category(val):
    if val < 1000:   return "Low (<$1k)"
    elif val < 5000: return "Lower-Middle ($1k–5k)"
    elif val < 15000:return "Upper-Middle ($5k–15k)"
    else:            return "High (>$15k)"

df["gdp_category"] = df["gdpPercap"].apply(gdp_category)
print("Secondary variable created: gdp_category (binned from gdpPercap)")

# 2c. Create secondary variable – Life Expectancy group
def life_group(val):
    if val < 50:  return "Low (<50)"
    elif val < 65:return "Medium (50–64)"
    elif val < 75:return "High (65–74)"
    else:         return "Very High (75+)"

df["life_group"] = df["lifeExp"].apply(life_group)
print("Secondary variable created: life_group (binned from lifeExp)")

# 2d. Select relevant columns only
df = df[["country","continent","year","lifeExp","gdpPercap","pop",
         "gdp_category","life_group"]]
print(f"Columns retained: {list(df.columns)}")

# ── 3. Frequency Distributions ──────────────────────────────
print("\n" + "=" * 60)
print("FREQUENCY DISTRIBUTIONS")
print("=" * 60)

# Variable 1: continent
print("\n[Variable 1] CONTINENT")
print("-" * 40)
freq1 = df["continent"].value_counts()
pct1  = df["continent"].value_counts(normalize=True).mul(100).round(1)
tbl1  = pd.DataFrame({"Count": freq1, "Percent (%)": pct1})
print(tbl1.to_string())
print(f"\nTotal valid observations: {freq1.sum()}")
print(f"Missing values: {df['continent'].isnull().sum()}")

# Variable 2: gdp_category
print("\n[Variable 2] GDP CATEGORY (derived from gdpPercap)")
print("-" * 40)
order2 = ["Low (<$1k)","Lower-Middle ($1k–5k)","Upper-Middle ($5k–15k)","High (>$15k)"]
freq2  = df["gdp_category"].value_counts().reindex(order2, fill_value=0)
pct2   = (freq2 / freq2.sum() * 100).round(1)
tbl2   = pd.DataFrame({"Count": freq2, "Percent (%)": pct2})
print(tbl2.to_string())
print(f"\nTotal valid observations: {freq2.sum()}")
print(f"Missing values: {df['gdp_category'].isnull().sum()}")

# Variable 3: life_group
print("\n[Variable 3] LIFE EXPECTANCY GROUP (derived from lifeExp)")
print("-" * 40)
order3 = ["Low (<50)","Medium (50–64)","High (65–74)","Very High (75+)"]
freq3  = df["life_group"].value_counts().reindex(order3, fill_value=0)
pct3   = (freq3 / freq3.sum() * 100).round(1)
tbl3   = pd.DataFrame({"Count": freq3, "Percent (%)": pct3})
print(tbl3.to_string())
print(f"\nTotal valid observations: {freq3.sum()}")
print(f"Missing values: {df['life_group'].isnull().sum()}")

print("\n" + "=" * 60)
print("END OF OUTPUT")
print("=" * 60)
