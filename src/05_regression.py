"""
05_regression.py  -  Régression pondérée (WLS) du log(dépense/équivalent-adulte)  -  niveau de vie.
Sortie : reports/regression_summary.txt, reports/figures/regression_coefficients.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE, "data", "processed", "menage_clusters.csv")
REPORT_DIR = os.path.join(BASE, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

EXPL = ["hhsize", "hage", "hgender", "heduc", "hdiploma", "halfab", "hactiv7j",
        "hbranch", "milieu", "region", "electricite", "eau_potable"]


def run():
    df = pd.read_csv(DATA_PATH)
    expl = [c for c in EXPL if c in df.columns]
    df = df.dropna(subset=["pcexp"])
    df = df[df["pcexp"] > 0]
    y = np.log(df["pcexp"])

    X = df[expl].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    X = sm.add_constant(X)
    weights = df["hhweight"]

    model = sm.WLS(y, X, weights=weights).fit()
    print(model.summary())

    with open(os.path.join(REPORT_DIR, "regression_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Variable dépendante : log(pcexp)  -  dépense par équivalent-adulte (niveau de vie)\n\n")
        f.write(str(model.summary()))

    coefs = model.params.drop("const").sort_values()
    plt.figure(figsize=(7, 5))
    coefs.plot(kind="barh", color=["tab:red" if v < 0 else "tab:green" for v in coefs])
    plt.title("Effets estimés sur log(dépense/équivalent-adulte)  -  EHCVM 2018-2019")
    plt.xlabel("Coefficient (WLS)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "regression_coefficients.png"), dpi=150)
    plt.close()

    print(f"R² ajusté : {model.rsquared_adj:.3f}")
    return model


if __name__ == "__main__":
    run()
