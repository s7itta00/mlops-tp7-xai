# Model Card - fraude (version servie par l'alias production)

## Modele
Nom et version dans le registre MLflow : `fraude`, version 1, alias `production` (registre reconstruit selon l'Annexe A du TP07 : le registre du TP2 n'etait plus disponible).
Famille et reglages : RandomForestClassifier (scikit-learn 1.9.1), n_estimators = 500, max_depth = 16, random_state = 42 ; serialisation skops.
Run d'origine (MLflow) : `f8bb3eea007347d99ba42fa8b74fd2ad` (experience `tp2-registre`, run `rf-500-16`).

## Objectif et usage prevu
Attribuer a chaque paiement par carte une probabilite de fraude (P(fraude)) a partir de 4 variables connues au moment du paiement (montant, anciennete du client, nombre de transactions sur 24 h, paiement de nuit), afin de **suspendre** le paiement et de declencher une verification (SMS, appel du conseiller).

## Usages exclus
- Bloquer definitivement un compte ou refuser un client sans revue humaine (5 % des fraudes predites sont des faux positifs : 15 clients honnetes bloques sur 1 250).
- Scoring de credit, tarification ou toute decision sur la personne (le modele ne voit que des transactions, pas la solvabilite).
- Utilisation sur d'autres populations / devises / canaux que ceux des donnees d'entrainement (donnees synthetiques, MAD).

## Donnees
Fichier, nombre de lignes, part de fraudes, empreinte (tag donnees_md5) : `data/transactions.csv`, 5 000 lignes, 568 fraudes (11,4 %), md5 `cf9f6f86c141e7df32c61715c493483f`.
Origine des donnees (reelles ou synthetiques) : **synthetiques**, generees par `donnees.py` (graine 7) ; decoupage train/test 75/25, random_state = 42.

## Performances (jeu de test, 1 250 transactions)
F1 : 0,8669   precision : 0,8837   rappel : 0,8507   AUC : 0,9872
Matrice de confusion : [[1101, 15], [20, 114]] -> 114 fraudes detectees sur 134, 15 faux positifs.

## Methodes d'explicabilite
Globale : importance par permutation (perte de F1, 10 melanges) et SHAP TreeExplainer (moyenne des |valeurs SHAP| sur 400 transactions de test).
Locale : SHAP waterfall (exact, base + somme = prediction) et LIME (2 000 voisins, graine 0) pour Mme Benali ; LIME instable sur les petites contributions, SHAP ne mesure pas une causalite.
Variables les plus influentes : montant (permutation 0,593 ; SHAP 0,116), puis anciennete, nuit et freq_24h a egalite (~0,056 en SHAP). Effet du montant positif au-dela d'environ 1 000 MAD ; anciennete elevee -> moins de fraude.

## Comparaison avec un modele interpretable
Regression logistique sur log10(montant) : F1 = 0,8593, 113 fraudes detectees / 134 (contre 0,8669 et 114 pour la foret), difference d'une seule fraude, dans le bruit du jeu de test. Arbre de profondeur 3 : lisible mais F1 = 0,5737 seulement.

## Limites et risques connus
- Donnees synthetiques : performances non garanties sur des donnees reelles.
- Variable `nuit` : effet fort et dependant du contexte (+0,37 pour Mme Benali, -0,04 pour la transaction 668) ; risque de penaliser les clients qui paient legitimement de nuit (travail de nuit).
- Peu de fraudes en test (134) : un ecart de F1 de 0,01 n'est pas significatif.
- Explications LIME variables d'une graine a l'autre pour anciennete et freq_24h.

## Recommandations d'utilisation et de surveillance
- Toujours une revue humaine avant blocage ; communiquer au client l'explication SHAP (deterministe), uniquement sur les causes principales.
- Surveiller la derive des distributions (montant, nuit) et le taux de faux positifs ; recalculer les explications (`src/journaliser.py`) a chaque changement de l'alias production.
- Envisager la regression logistique log10(montant) comme alternative lisible.

## Tracabilite
Run des explications (experience tp7-explicabilite) : `fa1d2890d4224667a5bfa2e405c7c3b1` (run `xai-fraude-v1`), tag `explications_run` sur fraude v1.
Auteurs : Sahtellil Adam (5IASD)   Date : 12/11/2026   Prochaine revue : 12/02/2027 ou a tout changement de l'alias production.
