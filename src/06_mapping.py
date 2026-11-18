"""
06_mapping.py — Carte des disparités régionales de pauvreté et typologie de ménages.

Agrège le taux de pauvreté officiel (pondéré) et la distribution des clusters K-Means
par région administrative, puis produit une carte choroplèthe.

Nécessite un fond de carte des régions de Côte d'Ivoire (GeoJSON/Shapefile) — à placer
dans data/raw/civ_regions.geojson (ex. source : GADM, humdata.org, ou limites officielles INS).

Sortie : reports/maps/poverty_rate_by_region.png, reports/regional_disparities.csv
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE, "data", "processed", "menage_clusters.csv")
GEO_PATH = os.path.join(BASE, "data", "raw", "civ_regions.geojson")
REPORT_DIR = os.path.join(BASE, "reports")
MAP_DIR = os.path.join(REPORT_DIR, "maps")
os.makedirs(MAP_DIR, exist_ok=True)

WEIGHT_CANDIDATES = ["hhweight", "poids_menage", "weight"]


def weighted_mean(group, col, weight_col):
    return np.average(group[col], weights=group[weight_col]) if weight_col else group[col].mean()


def run():
    df = pd.read_csv(DATA_PATH)
    if "region" not in df.columns or "pauvre_officiel" not in df.columns:
        raise ValueError("Colonnes 'region' et/ou 'pauvre_officiel' absentes.")
    weight_col = next((w for w in WEIGHT_CANDIDATES if w in df.columns), None)

    agg = df.groupby("region").apply(
        lambda g: pd.Series({
            "taux_pauvrete_pct": 100 * weighted_mean(g, "pauvre_officiel", weight_col),
            "n_menages": len(g),
            "cluster_dominant": g["cluster"].mode().iloc[0] if "cluster" in g.columns else np.nan,
        })
    ).reset_index()
    agg.to_csv(os.path.join(REPORT_DIR, "regional_disparities.csv"), index=False)
    print(agg.sort_values("taux_pauvrete_pct", ascending=False))

    try:
        import geopandas as gpd
        if os.path.exists(GEO_PATH):
            gdf = gpd.read_file(GEO_PATH)
            region_key = next((c for c in gdf.columns if "region" in c.lower() or "name" in c.lower()), gdf.columns[0])
            merged = gdf.merge(agg, left_on=region_key, right_on="region", how="left")
            fig, ax = plt.subplots(figsize=(8, 8))
            merged.plot(column="taux_pauvrete_pct", cmap="OrRd", legend=True, ax=ax,
                        edgecolor="black", linewidth=0.5,
                        legend_kwds={"label": "Taux de pauvreté officiel (%)"})
            ax.set_title("Disparités régionales de pauvreté — Côte d'Ivoire (EHCVM 2018-2019)")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(MAP_DIR, "poverty_rate_by_region.png"), dpi=200)
            plt.close()
            print(f"Carte sauvegardée : {MAP_DIR}/poverty_rate_by_region.png")
        else:
            print(f"Fond de carte introuvable ({GEO_PATH}) — carte non générée, "
                  f"seul le tableau regional_disparities.csv est produit.")
    except ImportError:
        print("geopandas non installé — tableau seul produit (pip install geopandas).")

    return agg


if __name__ == "__main__":
    run()
