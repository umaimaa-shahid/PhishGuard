from src.common import *

df = load_data()
print(df.shape, "phishing rate:", df[TARGET].mean())
print(df["registered_domain"].nunique(), "unique domains")

Xtr, Xte, ytr, yte = domain_disjoint_split(df)
print("domain-disjoint:", Xtr.shape, Xte.shape, ytr.mean(), yte.mean())

Xtr, Xte, ytr, yte = random_split(df)
print("random:", Xtr.shape, Xte.shape, ytr.mean(), yte.mean())