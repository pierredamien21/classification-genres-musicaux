# Trame de présentation — environ trois minutes

Support préparé le 5 novembre 2026 ; ce n’est pas une vidéo enregistrée.
Complété le 7 novembre : session utilisateur vérifiée, trente époques, macro-F1
validation 0,5707 au meilleur checkpoint (26). Aucune nouvelle mesure test.

| Temps | À montrer | Message |
|---|---|---|
| 0:00–0:20 | Question du notebook 00 | Comparer les indices musicaux explicites et les représentations d’un ViT pour huit genres FMA. |
| 0:20–0:45 | Partitions et artistes | Nettoyage, partitions et absence d’ajustement sur test rendent la comparaison interprétable. |
| 0:45–1:15 | STFT, HPSS et patches du notebook 02 | Le passage du son au tenseur transforme l’information. HPSS ne sépare pas les instruments ; le ViT reste gelé. |
| 1:15–1:45 | Validation et test | Fusion initiale non concluante ; candidat à 0,5035 de macro-F1 test, gain principal apparié de 0,0394. |
| 1:45–2:10 | Confusion et robustesse | Pop reste difficile, le bruit dégrade les résultats, le score test est inférieur à la validation. |
| 2:10–2:40 | Démo sur un fichier autorisé | Chargement, top-3, positions et spectrogramme ; les scores ne sont pas des certitudes. |
| 2:40–3:00 | Provenance et suite | Apprentissage de l’audit, comparaison, reprise de calcul, vérification et interprétation. Session autonome vérifiée : gain validation de 2,42 points pour la graine 42, sans conclure sur le test. |

Pour enregistrer, utiliser un fichier dont la diffusion est autorisée ou couper
l’audio. Ne pas publier de chemins personnels, de jetons Jupyter ou le corpus
complet. Figures PNG/SVG : `reports/notebooks-v1/figures`. Le projet présente
une démarche expérimentale, pas une nouvelle architecture ni un état de l’art.
