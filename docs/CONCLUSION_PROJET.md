# Conclusion du projet

7 novembre 2026. Le parcours expérimental et les livrables locaux sont terminés,
y compris la session autonome de l’utilisateur et le contrôle de ses résultats.
La publication publique, une vidéo enregistrée, une écoute subjective annotée et
une réinstallation depuis Internet sur une autre machine ne sont pas réalisées.

## Réponse à la question du projet

Sur FMA-small, le projet compare les informations musicales explicites et les
représentations d’un Vision Transformer préentraîné sur des images puis gelé.
Le meilleur candidat choisi sur validation combine l’extraction au bloc 6 et des
canaux original/harmonique/percussif issus de HPSS. Un classifieur linéaire apprend
ensuite à partir de ces représentations. Le projet n’entraîne pas un nouveau
Transformer de bout en bout et ne propose pas une nouvelle architecture.

L’évaluation V1 verrouillée porte sur 778 morceaux test de 202 artistes :
**macro-F1 0,5035, accuracy 51,16 %**. Le gain de macro-F1 face au ViT original est
de **0,0394**, intervalle apparié à 95 % **[0,0054 ; 0,0715]**, obtenu par bootstrap
des artistes. Les comparaisons secondaires restent incertaines. Cette observation
est propre au protocole et au corpus ; elle n’établit pas une supériorité générale
du ViT, de HPSS ou de l’analyse musicale explicite.

La fusion initiale des descripteurs et du ViT n’a pas apporté de gain convaincant
sur validation. Ce résultat négatif est conservé. La baisse du candidat de 0,5975
sur validation à 0,5035 sur test et sa fragilité au bruit limitent son utilisation.
Les huit classes FMA ne couvrent pas tout l’objectif initial : Hip-Hop représente
une partie du besoin « rap », mais le gospel n’a pas de classe dédiée.

## Apport de la dernière session

L’utilisateur a exécuté le CNN pendant trente époques avec patience cinq. Son
meilleur modèle, à l’époque 26, atteint **0,5707 de macro-F1 validation** et 57,82 %
d’accuracy. Il gagne 2,42 points de F1 par rapport à l’ancien CNN de même graine,
arrêté à l’époque 18 avec patience trois. La trajectoire initiale est reproduite.
Cette expérience illustre l’effet d’un arrêt trop précoce pour cette graine,
sans démontrer qu’un budget plus grand améliore toujours la généralisation.

Le contrôle recharge les poids, reproduit l’inférence train/validation et recalcule
les métriques. Le test V1 reste intact : la session est un développement
complémentaire, choisi après sa consultation, sans nouvelle estimation test.
[Résultats et preuves de la session](../reports/SESSION_UTILISATEUR.md).

## Ce que le portfolio peut montrer

- Une revue bibliographique reliée aux choix expérimentaux et à leurs limites.
- Un audit de données et des partitions par artistes, avec exclusions traçables.
- Une analyse du signal, des représentations et de l’entrée réellement reçue par le ViT.
- Des références, ablations, répétitions CNN et comparaisons sans masquer les résultats négatifs.
- Des métriques par morceau, des incertitudes par artiste et une évaluation verrouillée.
- Des calculs reprenables, des exports contrôlés, trois notebooks et une démo locale.
- Une expérience lancée par l’utilisateur, vérifiée puis interprétée avec ses limites.

Les supports sont accessibles depuis [le guide](GUIDE_LIVRAISON.md),
[le parcours pédagogique](../notebooks/00_parcours_projet.ipynb) et
[la trame de présentation](PRESENTATION_PORTFOLIO.md).

## Limites et prolongements possibles

La classification reste imparfaite, notamment pour Pop. Les scores ne sont pas
calibrés et aucun rejet des musiques hors des huit classes n’est validé. L’écoute
humaine des erreurs n’a pas fourni d’annotations utilisables. Les vérifications
locales ne prouvent pas une installation reproductible sur une machine vierge.
La démo fonctionne localement ; elle n’est pas un service public validé.

Une suite serait un nouveau projet expérimental : étendre les genres avec des
données autorisées, réserver une nouvelle évaluation indépendante, comparer un
préentraînement audio spécialisé, ou mesurer calibration et rejet hors domaine.
Chaque piste demanderait son protocole et son budget. Aucun calcul supplémentaire
n’est nécessaire pour présenter honnêtement les résultats de cette version.
