import ipaddress
from urllib.parse import urlsplit
import pandas as pd
import tldextract

# Same definition as notebooks/03_preprocessing.ipynb
extractor = tldextract.TLDExtract(include_psl_private_domains=True)

def extract_registered_domain(url):
    if pd.isna(url) or not str(url).strip():
        return pd.NA
    url = str(url).strip()
    parsed = urlsplit(url if "://" in url else "//" + url)
    hostname = parsed.hostname
    if not hostname:
        return pd.NA
    hostname = hostname.lower().rstrip(".")
    try:
        return ipaddress.ip_address(hostname).compressed
    except ValueError:
        pass
    e = extractor(hostname)
    if e.domain and e.suffix:
        return f"{e.domain}.{e.suffix}"
    return hostname

df = pd.read_csv("data/cleaned.csv")
df["registered_domain"] = df["URL"].apply(extract_registered_domain)

print("rows:", len(df))
print("missing:", df["registered_domain"].isna().sum())
print("distinct registered_domain:", df["registered_domain"].nunique())
print(df["registered_domain"].value_counts().head(10))

df.to_csv("data/cleaned_with_domains.csv", index=False)