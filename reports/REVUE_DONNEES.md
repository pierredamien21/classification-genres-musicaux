# Revue technique et sélection pour les premières expériences

Profil `conservative_baseline_v1` appliqué le 2026-10-20.

Sur 8 000 morceaux : **7857 retenus**, **126 en quarantaine réversible**, **17 exclus de cette sélection**.

Il s’agit d’une revue automatisée suivant une politique documentée. Aucune écoute humaine ni résolution subjective des genres n’est revendiquée.

- Manifeste pour les expériences (`data/manifests/baseline-20261020T142750008115Z/manifest.csv`, non versionné)
- Journal des 8 000 décisions (`data/manifests/baseline-20261020T142750008115Z/decisions.csv`, non versionné)
- Cas encore en attente (`data/manifests/baseline-20261020T142750008115Z/pending_review.csv`, non versionné)
- Paires candidates détaillées (`data/manifests/baseline-20261020T142750008115Z/fingerprint_candidates.csv`, non versionné)
- Provenance complète (`data/manifests/baseline-20261020T142750008115Z/report.json`, non versionné)

## Politique appliquée

- Erreurs du décodeur de référence : exclusion technique pour ce pipeline.
- Diagnostics, silence majoritaire, licence manquante ou empreinte insuffisante : quarantaine.
- Alerte de niveau seule : conservation avec son indicateur ; aucune normalisation ou correction destructive du signal.
- Sons exactement identiques, même genre et même partition : plus petit identifiant éligible conservé, répétitions exclues avec lien vers le représentant.
- Conflit de genres sur un contenu identique : groupe isolé, sans réétiquetage arbitraire.
- Candidats non exacts : composante entière mise en quarantaine, sans les déclarer doublons confirmés.

Les fichiers et partitions d’origine sont inchangés. La sélection des données ne repose sur aucun score de modèle. Les futures performances devront être annoncées comme celles de ce sous-ensemble filtré, avec ses effectifs, et non comme un résultat directement comparable à FMA-small intégral.

## Effectifs retenus

| Genre | Train | Validation | Test |
|---|---:|---:|---:|
| Electronic | 792 | 99 | 95 |
| Experimental | 794 | 100 | 97 |
| Folk | 786 | 99 | 100 |
| Hip-Hop | 772 | 100 | 98 |
| Instrumental | 786 | 93 | 96 |
| International | 794 | 98 | 100 |
| Pop | 782 | 94 | 92 |
| Rock | 793 | 97 | 100 |
| **Total** | **6299** | **780** | **778** |

## Empreintes perceptuelles

Outil vérifié : `fpcalc version 1.6.2 (FFmpeg Lavc62.11.100 Lavf62.3.100 SwR6.1.100)`. Les 8 000 fichiers ont été soumis à l’outil.
6 sorties en erreur ; 9 empreintes trop courtes ou peu informatives (catégories pouvant se recouvrir).
45 paires candidates, dont 33 non identiques au niveau PCM et 0 non exactes entre partitions.

La recherche utilise des mots Chromaprint communs peu fréquents, puis compare les bits après alignement temporel. Seuils fixés avant la recherche : dix mots distincts communs, au moins 120 mots alignés, au plus 10 % de bits différents. Les mots présents dans plus de 25 morceaux sont ignorés à la recherche. Ces seuils sont des heuristiques locales ; ils ne reproduisent pas le service AcoustID.

Contrôle positif : 6/6 transformations de 3 morceaux de **training** retrouvées (mono 16 kHz avec gain réduit ou décalage de deux secondes) ; 0 rapprochement(s) entre sources distinctes. [Résultats des contrôles](FINGERPRINT_CONTROLS.json). Ce petit contrôle ne mesure pas le rappel sur tout le dataset.

### Candidats non exacts (au plus trente affichés)

| A | B | Entre partitions | Taux de bits différents |
|---|---|---|---:|
| 040541 | 040542 | non | 0.0000 |
| 035182 | 035550 | non | 0.0001 |
| 088861 | 088873 | non | 0.0003 |
| 070301 | 073763 | non | 0.0006 |
| 085307 | 085596 | non | 0.0008 |
| 035544 | 036959 | non | 0.0014 |
| 085487 | 125196 | non | 0.0017 |
| 038955 | 039530 | non | 0.0044 |
| 122354 | 133100 | non | 0.0070 |
| 074546 | 082505 | non | 0.0100 |
| 093521 | 093522 | non | 0.0137 |
| 085598 | 086116 | non | 0.0151 |
| 093522 | 093523 | non | 0.0175 |
| 084198 | 084202 | non | 0.0184 |
| 061174 | 065752 | non | 0.0201 |
| 038353 | 054443 | non | 0.0274 |
| 003534 | 044793 | non | 0.0281 |
| 093520 | 093521 | non | 0.0312 |
| 028477 | 038961 | non | 0.0326 |
| 128441 | 140623 | non | 0.0368 |
| 093518 | 093523 | non | 0.0377 |
| 093520 | 093522 | non | 0.0425 |
| 056460 | 059449 | non | 0.0483 |
| 114539 | 115891 | non | 0.0528 |
| 114541 | 115892 | non | 0.0556 |
| 119831 | 128880 | non | 0.0618 |
| 144941 | 145703 | non | 0.0619 |
| 091319 | 146689 | non | 0.0634 |
| 088864 | 088877 | non | 0.0707 |
| 148113 | 148114 | non | 0.0778 |

## Vérifications et limites

Le constructeur vérifie les empreintes de l’audit, des signatures et de la politique, l’identité des listes de morceaux, l’absence de doublons PCM retenus et l’absence d’artiste ou de groupe de contenu retenu dans plusieurs partitions.

La paire exacte **057371 / 072149** porte deux genres différents (Instrumental / Experimental) : elle reste isolée. Les six échecs initiaux du décodeur sont également analysés par fpcalc/FFmpeg ; voir les diagnostics du scan.

Les empreintes peuvent manquer des réencodages fortement altérés, des modifications de tempo, des recouvrements courts ou des extraits non recouvrants du même morceau. Une sortie normale du second outil ne prouve pas qu’un MP3 soit intact. Les avertissements du premier décodeur restent donc pris en compte.

La bibliothèque [Chromaprint](https://github.com/acoustid/chromaprint) vise les sons presque identiques ; son [outil fpcalc](https://acoustid.org/chromaprint) calcule les empreintes localement. Aucune donnée audio ou empreinte n’a été téléversée.

## Prochaine étape

Extraire les descripteurs de timbre, rythme et harmonie sur ce manifeste versionné, puis entraîner une première référence. Ajuster toute normalisation sur train seulement, sélectionner sur validation et garder test pour l’évaluation finale. La file de quarantaine pourra faire l’objet d’une revue ultérieure et d’une analyse de sensibilité documentée.
