# Post LinkedIn — brouillon final (à valider)

🔍 **28% des ménages ivoiriens sous le seuil de pauvreté : ce que le Machine Learning révèle**

J'ai mené une analyse multidimensionnelle de la pauvreté en Côte d'Ivoire à partir des microdonnées
officielles EHCVM 2018-2019 (INS / Banque Mondiale) — 12 992 ménages, échantillon national.

🎯 Objectif : identifier les profils de ménages les plus vulnérables à partir de leurs conditions de
vie, niveau d'éducation, situation professionnelle et dépenses — en distinguant rigoureusement la
pauvreté monétaire officielle (seuil EHCVM appliqué à la dépense/équivalent-adulte) du simple
quintile de richesse (patrimoine), trop souvent confondus.

📊 Méthode : ACP sur les conditions de vie → K-Means (5 typologies de ménages) → classification
supervisée du statut de pauvreté officiel (AUC = 0,89) → régression pondérée du niveau de vie (R² = 0,51).

📌 Résultats clés :
• Taux de pauvreté national : 28%, avec un écart de 1 à 9 entre régions (53% à Tonkpi vs 6% à Abidjan)
• 5 profils de ménages identifiés, du plus vulnérable (52% de pauvreté, 6% d'accès à l'électricité)
  au plus aisé (2% de pauvreté, quasi-totalité équipée)
• Variables les plus discriminantes : taille du ménage, équipement du logement, accès à
  l'assainissement, branche d'activité du chef de ménage
• L'alphabétisation et le diplôme du chef de ménage restent les leviers les plus robustes sur le
  niveau de vie, après contrôle du milieu urbain/rural

💡 Pour le ciblage des politiques sociales : les régions du Nord-Ouest et de l'Ouest (Tonkpi,
Kabadougou, Bafing, Cavally) concentrent la pauvreté la plus sévère — un ciblage géographique simple
capte déjà une grande part du signal de vulnérabilité, à affiner par le profil socioéconomique du ménage.

💻 Code, méthodologie et résultats complets en open-source :
🔗 https://github.com/MediaBakayoko/ehcvm-poverty-ml

#DataScience #MachineLearning #Pauvreté #CôtedIvoire #EHCVM #GeoAI #SuiviEvaluation #PolitiquesSociales #INS
