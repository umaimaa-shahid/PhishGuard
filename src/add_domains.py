import pandas as pd
import tldextract

# Offline mode: uses the bundled public-suffix list, no download needed
extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

def registered(domain):
    e = extract(domain)
    if e.domain and e.suffix:
        return f"{e.domain}.{e.suffix}".lower()
    return str(domain).lower()   # IP addresses or odd hosts: keep as-is

df = pd.read_csv("data/cleaned.csv")
df["registered_domain"] = df["Domain"].map(registered)

print("rows:", len(df))
print("distinct Domain:", df["Domain"].nunique())
print("distinct registered_domain:", df["registered_domain"].nunique())
print("largest groups:\n", df["registered_domain"].value_counts().head(10))

df.to_csv("data/cleaned_with_domains.csv", index=False)