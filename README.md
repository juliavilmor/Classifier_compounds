# Classifier_compounds

ML model to classify compounds between actives and inactives.

It is trained with DUD-E database.




---
**NOTE**

This repository is under development!!

---







### Structure of the repository
```bash
├── classifier
│   ├── classifier_embedding.py
│   ├── classifier_model.py
│   ├── __init__.py
│   └── __pycache__
│       ├── classifier_embedding.cpython-311.pyc
│       ├── classifier_model.cpython-311.pyc
│       └── __init__.cpython-311.pyc
├── data
│   ├── data_retrospective
│   │   ├── IL36g_docking_0_enamine.csv
│   │   ├── IL36g_docking_1_enamine.csv
│   │   ├── IL36g_docking_2_enamine.csv
│   │   ├── IL36g_retrospective.csv
│   │   └── rcsb_pdb_6P9E.fasta
│   └── dude_full.tsv
├── nohup.out
├── README.md
├── results
│   ├── data_retrospective
│   │   ├── compound_df.pkl
│   │   ├── data_embed.csv
│   │   ├── data_embed.pkl
│   │   ├── IL36g_docking_scores.png
│   │   ├── new_compound_df.csv
│   │   ├── new_compound_df.pkl
│   │   ├── PR_curve_xf.png
│   │   ├── ROC_curve_xf.png
│   │   ├── statistics_IL36g.csv
│   │   └── statistics_prediction_IL36g_with_dude.csv
│   ├── ESM23B-fingerprints
│   │   ├── data_embed_100000.pkl
│   │   ├── data_embed_10000.pkl
│   │   ├── data_embed.csv
│   │   ├── data_embed.pkl
│   │   ├── Dataset_tSNE.png
│   │   ├── new_compound_df.csv
│   │   ├── new_compound_df.pkl
│   │   ├── PR_curve_RF_balanced.png
│   │   ├── Precision-Recall_curve_RF_balanced.png
│   │   ├── ROC_curve_RF_balanced.png
│   │   ├── statistics_DT.csv
│   │   ├── statistics_RF_balanced.csv
│   │   └── statistics_RF.csv
│   ├── ESM2650M-ChemBERT2
│   │   ├── compound_df.pkl
│   │   ├── data_embed_100000.pkl
│   │   ├── data_embed.csv
│   │   ├── data_embed.pkl
│   │   ├── new_compound_df.csv
│   │   ├── new_compound_df.pkl
│   │   ├── PR_curve_XGB_hyperparam.png
│   │   ├── ROC_curve_XGB_hyperparam.png
│   │   └── statistics_XGB_hyperparam.csv
│   ├── ESM2650M-ChemBerta
│   │   ├── compound_df.pkl
│   │   ├── new_compound_df.csv
│   │   └── new_compound_df.pkl
│   ├── ESM2650M-fingerprints
│   │   ├── data_embed_100000.pkl
│   │   ├── data_embed_10000.pkl
│   │   ├── data_embed_1000.pkl
│   │   ├── data_embed.csv
│   │   ├── data_embed.pkl
│   │   ├── data_embed_test.csv
│   │   ├── data_embed_test.pkl
│   │   ├── new_compound_df.csv
│   │   ├── new_compound_df.pkl
│   │   ├── PR_curve_DT.png
│   │   ├── PR_curve_RF_balanced.png
│   │   ├── PR_curve_RF.png
│   │   ├── PR_curve_xf_res0.05.png
│   │   ├── PR_curve_xf_res0.5.png
│   │   ├── PR_curve_XGB.png
│   │   ├── Precision-Recall_curve_RF_balanced.png
│   │   ├── Precision-Recall_curve_RF.png
│   │   ├── ROC_curve_DT.png
│   │   ├── ROC_curve_RF_balanced.png
│   │   ├── ROC_curve_RF.png
│   │   ├── ROC_curve_xf_res0.05.png
│   │   ├── ROC_curve_xf_res0.5.png
│   │   ├── ROC_curve_XGB.png
│   │   ├── statistics_DT.csv
│   │   ├── statistics_RF_balanced.csv
│   │   ├── statistics_RF.csv
│   │   └── statistics_XGB.csv
│   ├── ESM2650M-MolFormer
│   │   ├── compound_df.pkl
│   │   ├── data_embed_100000.pkl
│   │   ├── data_embed.csv
│   │   ├── data_embed.pkl
│   │   ├── new_compound_df.csv
│   │   ├── new_compound_df.pkl
│   │   ├── PR_curve_XGB_hyperparam.png
│   │   ├── PR_curve_XGB.png
│   │   ├── ROC_curve_XGB_hyperparam.png
│   │   ├── ROC_curve_XGB.png
│   │   ├── statistics_XGB.csv
│   │   └── statistics_XGB_hyperparam.csv
│   ├── ESM2650M-SELFormer
│   │   ├── compound_df.pkl
│   │   ├── data_embed_100000.pkl
│   │   ├── data_embed.csv
│   │   ├── data_embed.pkl
│   │   ├── new_compound_df.csv
│   │   ├── new_compound_df.pkl
│   │   ├── PR_curve_XGB_hyperparam.png
│   │   ├── ROC_curve_XGB_hyperparam.png
│   │   └── statistics_XGB_hyperparam.csv
│   ├── HyperparametersXGB
│   │   ├── PR_curve_hyp_XGB.png
│   │   ├── PR_curve_xf_default.png
│   │   ├── PR_curve_xf_hyp.png
│   │   ├── ROC_curve_hyp_XGB.png
│   │   ├── ROC_curve_xf_default.png
│   │   ├── ROC_curve_xf_hyp.png
│   │   └── statistics_hyp_XGB.csv
│   └── MolecularEmbeds
│       ├── PR_curve_molecular_embeds.png
│       ├── PR_curve_xf_ChemBERT2.png
│       ├── PR_curve_xf_Fingerprints.png
│       ├── PR_curve_xf_MolFormer.png
│       ├── PR_curve_xf_SELFormer.png
│       ├── ROC_curve_molecularembeds.png
│       ├── ROC_curve_xf_ChemBERT2.png
│       ├── ROC_curve_xf_Fingerprints.png
│       ├── ROC_curve_xf_MolFormer.png
│       ├── ROC_curve_xf_SELFormer.png
│       └── statistics_molecularembeds.csv
├── tests
│   ├── test_dude_IL36g.ipynb
│   ├── test_IL36_retrospective.ipynb
│   ├── test_model_and_undersampling.py
│   └── test_molecularembeds.py
└── utils
    ├── prepare_data_retrospective.py
    └── tune_hyp_XGB.py
```


