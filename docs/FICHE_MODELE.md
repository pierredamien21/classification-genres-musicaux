# Fiche du modèle — ViT bloc 6 + HPSS, V1

État au 29 octobre 2026 : export évalué, résultats test vérifiés ; interface non livrée.

## Usage et vocabulaire

Projet pédagogique et portfolio : attribuer un genre principal à un extrait
musical parmi Electronic, Experimental, Folk, Hip-Hop, Instrumental,
International, Pop et Rock. Le vocabulaire vient de FMA-small. Aucun détecteur
de genre inconnu, gospel ou multigenre. Aucun usage de modération ou de décision
sur des personnes n’a été évalué.

## Données et méthode

FMA-small retenu : 6 299 morceaux train, 780 validation, 778 test, partitions
préservées et contrôlées. Le candidat et C = 0,1 ont été sélectionnés sur
validation. Aucun ajustement sur test ni réentraînement train + validation.

Audio mono 16 kHz ; évaluation FMA sur trois fenêtres de dix secondes à 0/10/20 s,
avec complément de zéros à droite si nécessaire. Log-Mel 128 bandes ; séparation
harmonique/percussive appliquée au spectre complexe avant les spectrogrammes.
Canaux original/harmonique/percussif, bornage −80/+20 dB et redimensionnement
bicubique 224 × 224. ViT-Tiny préentraîné sur images, gelé : moyenne des tokens
de patches du bloc 6 avec normalisation finale, vecteur de 192 composantes.
Standardisation et régression logistique apprises sur train. Moyenne des
probabilités de fenêtres, puis classe de score maximal.

L’identité exacte des poids, versions et exports est consignée dans le
[manifeste verrouillé](../reports/final-evaluation-lock-20261029T085227817846Z.json).
Voir [l’inférence et ses commandes](INFERENCE_CANDIDAT.md) pour les contrats réels.

## Performances et limites

Test : 778 morceaux, 202 artistes ; macro-F1 **0,5035**, IC artiste 95 %
**[0,4133 ; 0,5449]**, accuracy **51,16 %**. Gain apparié de macro-F1 sur
ViT original : **+0,0394**, IC **[+0,0054 ; +0,0715]**. Pas de gain établi
sur toutes les variantes secondaires. Validation : 0,5975 de macro-F1 ;
la baisse test de 9,40 points doit rester visible. Pop F1 0,3258 contre
Hip-Hop 0,7525 ; performances inégales selon les genres.

Scores non calibrés, vocabulaire fermé, transfert depuis des images et corpus
restreint. Sensibilité au bruit constatée sur le panel de validation : perte
de macro-F1 0,1831 à SNR 20 dB. Ce panel ne représente pas toutes les conditions
d’enregistrement. Aucune écoute subjective annotée n’a été validée.

CPU local vérifié ; GPU Adreno incompatible CUDA, autre accélération non testée.
Médiane enregistrée 1,092 s par morceau pour le candidat, hors ensemble des coûts
de chargement et contrôle. Cela ne mesure pas encore la latence de la démo.

## Présentation et suivi

La future interface doit afficher version, huit classes, top-3, positions des
fenêtres et scores sans les présenter comme une certitude. Le mode longs fichiers
début/milieu/fin reste distinct du protocole FMA et doit être testé séparément.
Une validation de la démo ne remplace pas l’évaluation statistique.
Les licences des poids et de chaque morceau doivent être conservées et vérifiées
pour toute redistribution ; l’évaluation n’autorise pas à republier le corpus.

[Résultats complets et preuves](../reports/EVALUATION_FINALE.md).
Les extensions futures conservent cette V1 ; son test déjà consulté ne peut plus
servir à une nouvelle sélection présentée comme indépendante.
