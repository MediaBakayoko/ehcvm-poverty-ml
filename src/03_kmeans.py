"""
03_kmeans.py — Segmentation K-Means des ménages sur les scores ACP (conditions de vie).
Sortie : data/processed/menage_clusters.csv, reports/figures/elbow_silhouette.png,
         reports/cluster_profiles.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE = os.path.join(os.path.dirname(__file__), "..")
PCA_PATH = os.path.join(BASE, "data", "processed", "pca_scores.csv")
DATA_PATH = os.path.join(BASE, "data", "processed", "menage_dataset.csv")
OUT_DIR = os.path.join(BASE, "data", "processed")
FIG_DIR = os.path.join(BASE, "reports", "figures")
REPORT_DIR = os.path.join(BASE, "reports")
os.makedirs(FIG_DIR, exist_ok=True)

PROFILE_VARS = ["pauvre_officiel", "region", "milieu", "heduc", "hactiv7j",
                 "hhsize", "electricite", "eau_potable", "pcexp"]


def run_kmeans(k_range=range(2, 8)):
    pca_scores = pd.read_csv(PCA_PATH)
    pc_cols = [c for c in pca_scores.columns if c.startswith("PC")]
    X = pca_scores[pc_cols].values

    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(list(k_range), inertias, marker="o", color="tab:blue")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertie", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(list(k_range), sils, marker="s", color="tab:red")
    ax2.set_ylabel("Silhouette", color="tab:red")
    plt.title("Choix du nombre de clusters (K-Means)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "elbow_silhouette.png"), dpi=150)
    plt.close()

    k = 5  # retenu pour l'interprétabilité (silhouette proche du max observé en k=6)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    pca_scores["cluster"] = km.labels_

    df = pd.read_csv(DATA_PATH)
    merged = pd.concat([df.reset_index(drop=True), pca_scores[["cluster"]]], axis=1)
    merged.to_csv(os.path.join(OUT_DIR, "menage_clusters.csv"), index=False)

    profile_vars = [c for c in PROFILE_VARS if c in merged.columns]
    profiles = merged.groupby("cluster")[profile_vars].agg(
        lambda s: s.mean() if np.issubdtype(s.dtype, np.number) else s.mode().iloc[0]
    )
    profiles["n_menages"] = merged.groupby("cluster").size()
    profiles.to_csv(os.path.join(REPORT_DIR, "cluster_profiles.csv"))
    print(profiles)
    return merged, profiles


if __name__ == "__main__":
    run_kmeans()
