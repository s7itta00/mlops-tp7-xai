# TP07 – Explicabilité et auditabilité : réponses et tableaux

> **Note sur le registre** : le dossier `tp2-mlflow` (mlflow.db + mlartifacts) n'existait plus sur ce poste. Le modèle a été recréé selon l'**Annexe A** dans un registre neuf (`tp7-xai/mlflow.db`) : mêmes données (graine 7, 5 000 lignes, 568 fraudes), même forêt 500/16, même F1 = 0,8669. Les identifiants diffèrent de ceux du TP2 : run du modèle `f8bb3eea…`, run des explications `fa1d2890…`.
> Environnement : Python 3.13, scikit-learn 1.9.1, shap 0.54.1, mlflow 3.18.0 (aucun conflit de versions).

## Tableau SHAP / LIME (activité 4)

| Variable | SHAP (activité 3) | LIME, 2 000 voisins, graine 0 | Stable sur 5 graines ? |
|---|---|---|---|
| montant | +0,367 | +0,482 (`montant > 1007.36`) | Oui : +0,38 à +0,50 (500) ; +0,43 à +0,48 (2 000) |
| nuit | +0,370 | +0,183 (`nuit > 0.00`) | Oui en signe : +0,18 à +0,25 ; mais rang 2, pas 1 |
| anciennete | +0,090 | +0,026 (`29 < anciennete <= 60`) | Non à 500 voisins (−0,008 / +0,060) ; oui à 2 000 (+0,02 à +0,03) |
| freq_24h | −0,018 | +0,002 (`3 < freq_24h <= 4`) | Non : change de signe à 2 000 (−0,037 à +0,012) |

## Tableau du compromis (activité 5)

| Modèle | F1 | Fraudes détectées / 134 | Ce qu'on peut expliquer à la cliente sans outil |
|---|---|---|---|
| Forêt fraude@production (500 arbres) | 0,8669 | 114 | Rien directement : il faut SHAP/LIME |
| Logistique, montant brut | 0,7407 | 90 | Coefficients lisibles, mais modèle mal spécifié |
| Logistique, log10(montant) | 0,8593 | 113 | « Chaque transaction de plus en 24 h multiplie la cote de fraude par 4,76 ; chaque mois d'ancienneté la multiplie par 0,928 » |
| Arbre profondeur 2 (4 feuilles) | 0,2293 | 18 | Deux règles, mais quasi inutile |
| Arbre profondeur 3 (8 feuilles) | 0,5737 | 72 | Règles SI…ALORS lisibles (ci-dessous) |
| Arbre profondeur 6 (50 feuilles) | 0,7510 | 95 | Déjà trop de règles pour un humain |
| Arbre sans limite (206 feuilles) | 0,7556 | 102 | Illisible |

**Arbre de profondeur 3 en règles :**
- SI montant > 1 505,7 ET nuit = 1 ALORS fraude (feuilles 13 et 14)
- SI montant > 1 505,7 ET nuit = 0 ET ancienneté ≤ 69,5 mois ALORS fraude
- SINON normal (tout paiement ≤ 1 505,7 MAD, et les gros paiements de jour de clients anciens)

**Chemin de Mme Benali** : montant 1 800 > 1 505,7 → nuit = 1 > 0,5 → ancienneté 30 ≤ 64 → **feuille 13, fraude, P = 0,983**.

## Phrase pour Mme Benali
« Madame, votre paiement de 1 800 MAD a été suspendu par précaution parce qu'il a été fait en pleine nuit et que son montant est nettement supérieur à vos dépenses habituelles ; votre compte encore récent a joué un peu. Il suffit de nous confirmer que c'est bien vous pour le débloquer. »

## Questions

**Q1 – Registre et alias.** Charger `models:/fraude@production` découple le code de la version : c'est exactement ce que sert la production, sans chemin de fichier ni numéro codé en dur ; promouvoir un modèle consiste à déplacer l'alias, sans toucher au code. Si l'alias passait à la version 2, les explications de l'activité 6 resteraient valides **pour la version 1** seulement : elles portent les tags `modele_version = 1` et `modele_run_id = f8bb3eea…`, et le tag `explications_run` est posé sur la version 1, pas sur la version 2. On le détecterait en comparant `get_model_version_by_alias('fraude','production').version` (= 2) au tag `modele_version` du run (= 1), ou en voyant que la version 2 n'a pas de tag `explications_run`. Il faudrait alors relancer `journaliser.py`.

**Q2 – Deux importances.** Le classement est presque le même : montant est premier dans les deux cas. En revanche, anciennete passe devant freq_24h en MDI (0,237 contre 0,191), alors qu'elles sont à égalité en permutation (0,268 et 0,269), et nuit est quatrième dans les deux cas, mais avec 0,093 contre 0,196. La MDI additionne la baisse d'impureté apportée par chaque coupure, **sur les données d'entraînement**. Une variable binaire comme nuit ne permet qu'une seule coupure par chemin, alors qu'une variable continue (montant, ancienneté) peut être coupée de nombreuses fois : la MDI favorise donc les variables continues à forte cardinalité. La permutation mesure la **perte de F1 sur le test** quand on mélange la colonne : elle mesure la dépendance réelle du modèle, sans ce biais. Une seule coupure sur nuit change fortement la prédiction (+1,2 dans le score générateur), d'où 0,196.

**Q3 – Beeswarm.** Les points d'ancienneté situés à droite de zéro (contribution positive, donc vers la fraude) sont **bleus** : ce sont les faibles anciennetés. Les anciennetés élevées (rouges) sont à gauche. Une ancienneté faible augmente donc le risque, et une ancienneté élevée le diminue. Le beeswarm donne le **sens** et la **forme** de l'effet, ainsi que sa dispersion d'une transaction à l'autre. La permutation ne donne qu'un nombre positif, l'ampleur, sans direction. Le graphique `shap_montant.png` montre en plus que la contribution du montant devient positive vers 1 000 MAD.

**Q4 – Global et local.** Ce n'est pas une contradiction. L'importance globale est la **moyenne** des |SHAP| sur 400 transactions, or 82 % des paiements ont lieu de jour, et pour eux nuit a une petite contribution (environ −0,04). Pour la minorité de paiements de nuit, la contribution est forte (environ +0,2 à +0,37). La moyenne dilue donc cet effet. Efficacité : 0,1157 + 0,3697 + 0,3665 + 0,0903 − 0,0183 = **0,9239 ≈ 0,924** = prédiction. Pour la transaction 668 : 0,1157 + 0,5221 + 0,2185 + 0,1807 − 0,0410 = 0,996.

**Q5 – Phrase et transaction 668.** La phrase est donnée plus haut. La transaction 668 a lieu **de jour** (nuit = 0). Or la valeur de base (0,1157) intègre déjà le fait que 18 % des paiements sont de nuit, donc risqués. Être de jour est un peu plus rassurant que la moyenne, d'où une contribution légèrement négative (−0,041). Pour Mme Benali, nuit = 1, ce qui est rare et associé à la fraude, d'où +0,37 : avec SHAP, une même variable peut pousser dans un sens ou dans l'autre selon sa valeur et selon les autres variables (interactions dans la forêt).

**Q6 – Stabilité de LIME.** Stables sur les 5 graines : **montant** (+0,38 à +0,50) et **nuit** (+0,18 à +0,25), toujours positifs et toujours aux deux premiers rangs. Instables : **anciennete**, qui change de signe à 500 voisins (−0,008, +0,060, −0,006…), et **freq_24h**, qui change de signe à 2 000 voisins (+0,002, −0,037, +0,012…). Augmenter le voisinage réduit la variance sans la supprimer. Règle : ne communiquer au client que les causes **dont le signe et le rang sont stables sur plusieurs graines** (et de poids non négligeable), fixer et journaliser la graine, et préférer SHAP, qui est déterministe pour ce modèle.

**Q7 – SHAP ou LIME.** Les deux méthodes désignent les **mêmes deux causes principales** (montant et nuit, positives), mais pas dans le même ordre : SHAP met nuit en premier (+0,370 contre +0,367), LIME met montant en premier (+0,48 contre +0,18). On ne peut pas comparer les nombres directement. SHAP répartit exactement l'écart prédiction − valeur de base, en unités de probabilité, avec pour référence la moyenne. LIME donne les coefficients d'un modèle linéaire local ajusté sur des variables **discrétisées** (des règles du type `montant > 1007.36`) et pondéré par un noyau de proximité : ses poids dépendent de la discrétisation, du noyau et de la graine, et leur somme ne redonne pas la prédiction. Pour ce modèle, je choisirais **SHAP (TreeExplainer)**, qui est exact, rapide, reproductible, et dont les contributions s'additionnent jusqu'à la prédiction, ce qui est auditable. LIME reste utile comme contrôle croisé, ou pour un modèle non arborescent.

**Q8 – Compromis.** La forêt détecte 114 fraudes sur 134 et la logistique log10 en détecte 113 : l'écart est d'**une seule fraude**, et ΔF1 = 0,008. Avec 134 fraudes en test, l'écart-type du rappel vaut environ √(0,85 × 0,15 / 134) ≈ 0,03, soit environ ±4 fraudes : l'écart est bien inférieur au bruit. À performance équivalente, je mettrais en production la **régression logistique sur log10(montant)**. Ses décisions sont lisibles sans explicateur (coefficients, multiplicateurs de cote), sa dépendance monotone est facile à auditer, et elle se surveille mieux. Bloquer un paiement exige de pouvoir justifier la décision au client. La forêt pourrait rester comme challenger.

**Q9 – Reconstituer une décision.** Éléments enregistrés : le modèle exact (fraude v1, run `f8bb3eea`, artefact skops, `requirements_modele.txt`), les données d'entraînement (empreinte md5 `cf9f6f86…`, `donnees.py` avec sa graine, découpage `random_state=42`), le run d'explications `xai-fraude-v1` (tags modèle/données, paramètres SHAP/LIME et graines, graphiques, `lime_benali.html`), la Model Card attachée à la version et le code versionné dans Git. Il manque encore le **journal de la décision elle-même** : les variables d'entrée exactes de la transaction de Mme Benali telles que reçues en production, l'horodatage, la version et l'alias effectivement servis à cet instant, le score, le seuil appliqué et l'explication calculée à ce moment (prediction logging). Il manque aussi la version DVC des données et l'identité du décideur humain.

**Q10 – Model Card.** *Usage exclu* : bloquer définitivement un compte sans revue humaine. Justification : la matrice [[1101, 15], [20, 114]] contient 15 faux positifs, donc 15 clients honnêtes bloqués sur 1 250, et la précision vaut 0,88, si bien qu'environ 1 alerte sur 9 est fausse. *Limite* : l'effet de la variable nuit dépend du contexte et peut pénaliser des clients légitimes. Justification : sa contribution SHAP va de +0,37 (Mme Benali) à −0,04 (transaction 668), et LIME la place de façon variable (0,18 à 0,25). Autre limite : les données sont synthétiques, et le F1 n'est pas garanti sur des données réelles.

## Captures à faire (non automatisables)
1. Sortie de `src/verifier_modele.py` + onglet Models de MLflow (alias production) – http://127.0.0.1:5000
2. Tableau de permutation + `reports/shap_global.png`
3. Sortie de `src/xai_local.py` + `reports/shap_benali.png`
4. Sortie de `src/xai_lime.py` + `reports/lime_benali.html` dans le navigateur
5. Run `xai-fraude-v1` (tags + dossier explications/) + tag `explications_run` sur fraude v1
6. Page de la version 1 de fraude (description = Model Card, tags)

Toutes les sorties texte sont dans `reports/sortie_*.txt`.
