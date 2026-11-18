
# SYNTHESE DES RESULTATS — Machine Learning for Multidimensional Poverty Profiling, Côte d'Ivoire
## Données réelles EHCVM 2018-2019 (INS/Banque Mondiale), 12 992 ménages

## 1. Taux de pauvreté monétaire officiel (national, pondéré)
28.0% des ménages ivoiriens sont sous le seuil national de pauvreté (dépense par équivalent-adulte < seuil EHCVM).

## 2. Disparités régionales (33 régions)
- Les plus touchées : Tonkpi (53%), Kabadougou (51%), Bafing (51%), Cavally (51%), N'Zi (49%)
- Les moins touchées : Autonome d'Abidjan (6%), Sud-Comoé (17%), Gontougo (20%), Yamoussoukro (20%)
- Écart de 1 à 9 entre la région la plus pauvre et Abidjan.

## 3. Typologie de ménages (K-Means, k=5, sur ACP des conditions de vie)
- Cluster 3 (aisé, n=469) : 2% pauvreté, forte dépense (1.38M FCFA/an), quasi-totalité électrifiée/eau potable, chef très éduqué.
- Cluster 1 (très vulnérable, n=1920) : 52% pauvreté, 6% électrifié, 37% eau potable, chef peu éduqué.
- Cluster 0 (intermédiaire supérieur, n=3401) : 17% pauvreté, 90%+ accès eau/électricité.
- Clusters 2 et 4 (intermédiaires, n=3832 et 3370) : 39-43% pauvreté, accès partiel aux services.

## 4. Classification supervisée de la pauvreté officielle
Meilleur modèle : Gradient Boosting (AUC=0.892, F1=0.669)
Variables les plus discriminantes : taille du ménage, possession d'un téléviseur (proxy niveau de vie),
accès à l'assainissement amélioré, région, branche d'activité du chef de ménage.

## 5. Régression du niveau de vie (WLS, R²=0.51)
Effets significatifs (p<0.001) sur log(dépense/équivalent-adulte) :
- Taille du ménage : effet négatif fort (-0.118) — économies d'échelle négatives / charge démographique
- Milieu rural : effet négatif (-0.129) vs urbain
- Électricité : effet positif fort (+0.174)
- Eau potable : effet positif (+0.136)
- Alphabétisation du chef : effet positif (+0.138)
- Diplôme du chef : effet positif (+0.085)
- Niveau d'éducation du chef : effet négatif léger (-0.024, à interpréter avec prudence — colinéarité avec diplôme)

## 6. Urbain vs rural
Milieu rural systématiquement associé à une pauvreté plus élevée et un accès moindre aux infrastructures,
confirmé à la fois par le clustering et la régression.

## 7. Recommandations pour le ciblage des politiques sociales
1. Prioriser les régions du Nord-Ouest et de l'Ouest (Tonkpi, Kabadougou, Bafing, Cavally, N'Zi) où la pauvreté dépasse 48%.
2. Cibler les ménages du cluster 1 (très vulnérable) : accès eau/électricité quasi inexistant — investir en infrastructures de base avant tout transfert monétaire.
3. L'éducation (alphabétisation, diplôme) du chef de ménage reste le levier le plus robuste et actionnable à moyen terme.
4. Les programmes de ciblage géographique simples (région + milieu rural) captent déjà une grande part du signal de vulnérabilité, en complément du profil socioéconomique du ménage.
5. Rappel méthodologique : le quintile de richesse (patrimoine) ne doit PAS servir de critère d'éligibilité à des
   programmes de lutte contre la pauvreté monétaire — ce projet confirme que la pauvreté officielle (EHCVM) recoupe
   partiellement mais pas totalement les quintiles de richesse basés sur les actifs.
