# Fiche du modèle V2

Mis à jour : 11 novembre 2026. La [fiche V1](FICHE_MODELE.md) reste inchangée : c'est un instantané du 29 octobre, épinglé par les preuves V1.

## Résumé

**MAEST bloc 12 gelé et sonde logistique**, choisi sur le seul développement par validation croisée imbriquée. Il classe un extrait musical en 8 genres FMA.

Sur le test V2 indépendant (501 morceaux, un par artiste), il obtient une **macro-F1 de 0,572** [0,534 ; 0,615] et une **accuracy de 57,9 %**.

## Fonctionnement

1. Audio mono rééchantillonné à 16 kHz, avec trois fenêtres de 10 s à 0, 10 et 20 s.
2. MAEST (`mtg-upf/discogs-maest-10s-pw-129e`, révision fixée) sur chaque fenêtre : tokens [CLS], [DIST] et moyenne des patchs après la norme finale, soit 2 304 dimensions.
3. Standardisation et régression logistique (C = 1e-4), ajustée sur les 23 571 fenêtres des 7 857 morceaux du développement.
4. Score du morceau : moyenne des probabilités des fenêtres.
5. En option, une température de 0,824 et un seuil de rejet à 0,337 sur la probabilité maximale. Ces valeurs viennent du développement ; voir les limites.

L'entrée MAEST a été vérifiée contre le code d'origine (`tests/test_audio_pretrained.py`). Code : `src/music_genre/inference/predict_v2.py`. Poids figés par le [verrou 3](../reports/v2/lock3.json).

## Performances

| Mesure | Développement (CV 5 plis) | Test V2 |
|---|---:|---:|
| Macro-F1 | 0,684 ± 0,027 | 0,572 [0,534 ; 0,615] |
| Accuracy | 0,687 | 0,579 |
| Meilleur genre | Hip-Hop 0,85 | Hip-Hop 0,91 |
| Pire genre | Pop 0,48 | Instrumental 0,27 |

- **Comparaisons sur le même test** :
  - meilleur système V1 : +0,091 [+0,049 ; +0,134] ;
  - CNN renforcé : +0,117 [+0,083 ; +0,154] ;
  - les deux restent significatifs après correction de Holm.
- **Robustesse** (développement, hors pli) :
  - gain de ±6 dB : −0,006 ;
  - MP3 à 64 kb/s : −0,004 ;
  - bruit blanc à 30, 20 et 10 dB : −0,018, −0,066 et −0,193.

Rapport complet : [RAPPORT_V2.md](RAPPORT_V2.md).

## Usages

- **Prévus** : démonstration de portfolio, recherche, comparaison de représentations audio.
- **Non prévus** :
  - usage commercial (les poids MAEST sont sous licence CC-BY-NC-SA-4.0) ;
  - décisions sur des artistes ;
  - genres hors des 8 classes ;
  - audio bruité.

## Limites connues

- **Changement de distribution.** Le score chute de 11 points entre le développement et le test : environ 3 points viennent de la pondération par artiste, le reste n'est pas expliqué par les métadonnées. Instrumental et Electronic basculent vers Experimental.
- **Calibration non transférable.** La température améliore l'ECE sur le développement (0,065 → 0,011) mais la dégrade sur le test (0,045 → 0,085).
- **Rejet inefficace.** Sur le test, le seuil accepte 89 % des morceaux hors domaine (AUROC 0,67) et 11 morceaux Gospel sur 12.
- **Fragilité au bruit fort.** C'est une limite déclarée : l'augmentation qui la corrigeait n'a pas été adoptée, en application de la règle préenregistrée.
- **Recouvrement possible.** Celui entre Discogs20, les données d'entraînement de MAEST, et FMA est invérifiable.
- **Coût.** Environ 12 s de calcul CPU par morceau (x86-64 émulé sur ARM64, un thread), dont l'essentiel pour MAEST.

## Licences

| Élément | Licence |
|---|---|
| Poids MAEST | CC-BY-NC-SA-4.0, non commercial |
| Tête logistique | Ajustée sur l'audio FMA, sous licences propres à chaque morceau, majoritairement non commerciales |
| Code du projet | Aucune licence choisie à ce jour, choix laissé à l'auteur |

Aucun poids n'est versionné : il faut les retélécharger avec leurs empreintes ([reproduction](REPRODUCTION.md)).
