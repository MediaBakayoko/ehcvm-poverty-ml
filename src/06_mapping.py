"""
06_mapping.py  -  Carte des disparités régionales de pauvreté (33 régions, EHCVM 2018-2019).

Fond de carte : GADM 4.1 niveau 2 (régions administratives Côte d'Ivoire), à placer dans
data/raw/gadm_civ2/gadm41_CIV_2.json (téléchargeable librement sur https://gadm.org).

Sortie : reports/maps/poverty_rate_by_region.png, reports/regional_disparities.csv,
         data/processed/poverty_by_region.geojson
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE, "data", "processed", "menage_clusters.csv")
GEO_PATH = os.path.join(BASE, "data", "raw", "gadm_civ2", "gadm41_CIV_2.json")
REPORT_DIR = os.path.join(BASE, "reports")
MAP_DIR = os.path.join(REPORT_DIR, "maps")
os.makedirs(MAP_DIR, exist_ok=True)

# Codes région EHCVM (variable 'region', 1-33) -> noms officiels
REGION_MAP = {
    1: "Autonome d'Abidjan", 2: "Haut-Sassandra", 3: "Poro", 4: "Gbêkê", 5: "Indénié-Djuablin",
    6: "Tonkpi", 7: "Yamoussoukro", 8: "Gontougo", 9: "San-Pédro", 10: "Kabadougou",
    11: "N'Zi", 12: "Marahoué", 13: "Sud-Comoé", 14: "Worodougou", 15: "Lôh-Djiboua",
    16: "Agnéby-Tiassa", 17: "Gôh", 18: "Cavally", 19: "Bafing", 20: "Bagoué",
    21: "Bélier", 22: "Béré", 23: "Bounkani", 24: "Folon", 25: "Gbôklé",
    26: "Grands-Ponts", 27: "Guémon", 28: "Hambol", 29: "Iffou", 30: "La Mé",
    31: "Nawa", 32: "Tchologo", 33: "Moronou",
}


def normalize(s):
    s = s.lower().strip().replace('é', 'e').replace('ô', 'o').replace('ê', 'e').replace("'", "")
    return re.sub(r'[^a-z0-9]', '', s)


def run():
    df = pd.read_csv(DATA_PATH)

    def weighted_mean(g, col):
        return np.average(g[col], weights=g["hhweight"])

    agg = df.groupby("region").apply(
        lambda g: pd.Series({
            "taux_pauvrete_pct": 100 * weighted_mean(g, "pauvre_officiel"),
            "n_menages": len(g),
            "pcexp_moyen": weighted_mean(g, "pcexp"),
        }), include_groups=False
    ).reset_index()
    agg["region_nom"] = agg["region"].map(REGION_MAP)
    agg = agg.sort_values("taux_pauvrete_pct", ascending=False)
    agg.to_csv(os.path.join(REPORT_DIR, "regional_disparities.csv"), index=False)
    print(agg.to_string())

    try:
        import geopandas as gpd
        if os.path.exists(GEO_PATH):
            gdf = gpd.read_file(GEO_PATH)
            gdf["norm"] = gdf["NAME_2"].apply(normalize)
            agg["norm"] = agg["region_nom"].apply(normalize)
            agg.loc[agg["region_nom"] == "Autonome d'Abidjan", "norm"] = "abidjan"
            merged = gdf.merge(agg, on="norm", how="left")

            fig, ax = plt.subplots(figsize=(9, 9))
            merged.plot(column="taux_pauvrete_pct", cmap="OrRd", legend=True, ax=ax,
                        edgecolor="black", linewidth=0.5,
                        legend_kwds={"label": "Taux de pauvreté officiel (%)", "shrink": 0.6})
            for _, row in merged.iterrows():
                c = row.geometry.centroid
                ax.annotate(f"{row['taux_pauvrete_pct']:.0f}%", (c.x, c.y), fontsize=6, ha="center")
            ax.set_title("Disparités régionales de pauvreté monétaire  -  Côte d'Ivoire\n"
                         "(EHCVM 2018-2019, seuil officiel INS, 33 régions)", fontsize=12)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(MAP_DIR, "poverty_rate_by_region.png"), dpi=200, bbox_inches="tight")
            plt.close()
            merged.drop(columns=["norm"]).to_file(
                os.path.join(BASE, "data", "processed", "poverty_by_region.geojson"), driver="GeoJSON")
            print(f"Carte sauvegardée : {MAP_DIR}/poverty_rate_by_region.png")
        else:
            print(f"Fond de carte introuvable ({GEO_PATH})  -  télécharger GADM CIV niveau 2.")
    except ImportError:
        print("geopandas non installé  -  tableau seul produit (pip install geopandas).")

    return agg


if __name__ == "__main__":
    run()
