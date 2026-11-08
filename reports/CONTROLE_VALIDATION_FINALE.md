# Contrôle indépendant de la répétition finale sur validation

29 octobre 2026. **Contrôle réussi** ; aucun fichier audio décodé par ce contrôle.

La répétition terminée à 17 h 02 (Paris) a produit neuf sorties par morceau sur
les 780 morceaux de validation : 7 020 prédictions, soit 21 060 vecteurs de fenêtres.
Le script `scripts/verify_final_validation.py` les relit sans importer les fonctions
de stockage, d’agrégation ou de métrique de l’évaluateur.

Il vérifie les empreintes et la provenance des enregistrements, l’identité des
modèles, l’ordre des labels, la partition exacte et les limites des fenêtres.
Les moyennes sont recalculées avec `math.fsum`. Les probabilités sont comparées
aux fichiers historiques, alignés par identifiant. Les matrices de confusion,
la macro-F1 à huit labels et l’accuracy sont recalculées explicitement.

Résultat : aucun désaccord de classe pour les neuf systèmes ; écarts de
probabilités inférieurs à 10⁻⁶ ; métriques et matrices historiques reproduites.
Le rapport détaille séparément l’écart des probabilités enregistrées et celui
des moyennes recalculées. Tous les fichiers attendus sont présents, sans résultat
supplémentaire. Les cinq tests du vérificateur couvrent un exemple calculable à
la main et le refus de moyennes, identités, valeurs ou géométries incohérentes.

Preuve : [rapport JSON](FINAL_VALIDATION_RECOUNT.json), incluant l’empreinte du
script et des preuves utilisées. Les sources de l’évaluateur et les modèles
verrouillés restent inchangés.

Ce contrôle confirme la fidélité du pipeline sur validation, pas une performance
de généralisation indépendante. Il ne rejoue pas l’inférence et ne recalcule pas
toutes les caractéristiques. La campagne test vérifiera de nouveau l’intégrité
complète du protocole avant son premier décodage.

La suite est l’exécution des neuf exports sur les 778 morceaux test, puis le
rapport prévu : majoritaire de train, métriques par genre, confusions, bootstrap
apparié par artiste, comparaisons fixées et moyenne des métriques CNN. Aucun
nouveau choix de modèle ou de paramètre ne sera fondé sur les scores test.
