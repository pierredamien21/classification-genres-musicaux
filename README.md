# Genres musicaux : ViT, modèles audio préentraînés et réplication

Projet de portfolio. Il classe des extraits de 30 s de la Free Music Archive en 8 genres, puis vérifie sur un test neuf, une seule fois et sous verrou, ce que valaient les conclusions d'une première version.

## Résultats (test V2 : 501 morceaux, un par artiste, artistes absents de FMA-small)

| Système | Macro-F1 [IC 95 %] |
|---|---:|
| **MAEST bloc 12 gelé + sonde logistique (V2)** | **0,572** [0,534 ; 0,615] |
| ViT d'images bloc 6 + HPSS (candidat V1) | 0,481 [0,442 ; 0,525] |
| ViT d'images original | 0,453 [0,418 ; 0,496] |
| CNN renforcé, 3 graines | 0,444–0,462 |

- **H1, réplication de la V1** : +0,028 [−0,007 ; +0,064]. Non concluante : même sens qu'en V1, mais l'intervalle contient 0.
- **H2 et H3** : MAEST bat le candidat V1 (+0,091) et le CNN (+0,117), p Holm = 0,0003.
- **Résultats défavorables, rapportés tels quels** :
  - le score baisse de 11 points entre le développement et le test ;
  - la calibration ne se transfère pas ;
  - le rejet des genres inconnus est inefficace.

## Documents

| Document | Contenu |
|---|---|
| [Rapport V2](docs/RAPPORT_V2.md) | Question, protocole, résultats, conclusions |
| [Fiche des données](docs/FICHE_DONNEES.md) | Cohortes, audit, licences, écarts connus |
| [Fiche du modèle V2](docs/FICHE_MODELE_V2.md) | Fonctionnement, performances, usages, limites |
| [Interprétation](docs/INTERPRETATION_V2.md) | Écarts, analyse par artiste et par genre, écoute |
| [Architecture](docs/ARCHITECTURE.md) | Modules, traçabilité, verrous |
| [Reproduction](docs/REPRODUCTION.md) | Installation vérifiée, données, poids, chaîne de commandes |
| [Contrôle final](docs/CONTROLE_FINAL.md) | Chaque critique : corrigée, mesurée ou limite assumée |
| [Index des résumés](docs/resumes/INDEX.md) | État condensé de chaque étape, pour reprendre le travail |

Le protocole préenregistré est dans [PROTOCOLE_V2.md](docs/PROTOCOLE_V2.md), et le déroulé dans [PLAN_V2.md](docs/PLAN_V2.md).

**Historique V1** :
- l'étiquette Git `v1-final` ;
- les documents de [docs/archive/](docs/archive/) ;
- quelques documents laissés en place parce que des preuves V1 les épinglent : [fiche du modèle V1](docs/FICHE_MODELE.md), [conclusion V1](docs/CONCLUSION_PROJET.md), [guide de livraison V1](docs/GUIDE_LIVRAISON.md), [décisions](docs/DECISIONS_EXPERIMENTALES.md), [revue bibliographique](docs/REVUE_BIBLIOGRAPHIQUE.md) ;
- les rapports V1 : [évaluation finale V1](reports/EVALUATION_FINALE.md).

## Démarrage rapide

Installation : voir [REPRODUCTION.md](docs/REPRODUCTION.md). Ensuite :

```powershell
.\.venv\Scripts\python.exe -m pytest -q                      # 309 tests sur le poste complet
.\.venv\Scripts\python.exe -m music_genre doctor --root .    # diagnostic, sans ML
.\.venv\Scripts\python.exe -m music_genre predict morceau.mp3 --root .   # exports V1 locaux requis
```

La démo Gradio (`app/`) et les notebooks (`notebooks/`) datent de la V1. Ils ont leurs propres fichiers `requirements/`.

## Organisation

```text
src/music_genre/   audio, données, caractéristiques, modèles, entraînement, évaluation, inférence
scripts/           chaîne V2 (extraction, CV, robustesse, évaluation unique, recomptage) et outils V1
configs/           une configuration par représentation et par étape ; protocole V2
reports/           rapports versionnés ; reports/v2/ porte les verrous et les preuves V2
docs/              documents ci-dessus ; docs/archive/ : historique V1
tests/             tests unitaires et de contrats
data/ runs/ artifacts/pretrained/   locaux, jamais versionnés (audio, caches, poids)
```

## Licences et publication

- **Audio FMA** : licences propres à chaque morceau, à 82–87 % non commerciales. Il n'est jamais versionné.
- **Poids** :
  - MAEST : CC-BY-NC-SA-4.0, non commercial ;
  - AST : BSD-3-Clause ;
  - PANNs : CC-BY-4.0 ;
  - ViT tiny : Apache-2.0.
  Aucun n'est versionné.
- **Code** : aucune licence n'a encore été choisie ; c'est à l'auteur de le faire avant publication.

Environnement de référence : Python 3.13 x86-64, émulé sur Windows ARM64, CPU seul.
