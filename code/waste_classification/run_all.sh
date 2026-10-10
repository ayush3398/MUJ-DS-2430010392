#!/bin/bash
# Full pipeline, in order (about 10-40 minutes on CPU depending on your laptop; much faster with a GPU)
set -e
pip install -r requirements.txt
python download_data.py          # get TrashNet
python dataset_analysis.py       # CO1: stats, split, plots
python model.py                  # CO2: architecture, parameter count, diagram
python train.py                  # CO3/CO5: trains the 3 ablation experiments
python evaluate.py               # test metrics, confusion matrix, report, predictions
python compare_models.py         # CO4: HOG+SVM vs MLP vs CNN
python report.py                 # results/RESULTS.md with every real number
