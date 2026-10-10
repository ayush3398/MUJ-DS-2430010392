# Results (auto-generated from the actual experiments)

## Dataset statistics

- Source: TrashNet (dataset-resized), github.com/garythung/trashnet
- Images used: 2387 (raw images in chosen folders: 2390; exact duplicates removed before splitting)
- Classes (5): cardboard, glass, metal, paper, plastic
- Original size (WxH): {'512x384': 2387} -> resized to 128x128x3
- Images per class: {'cardboard': 403, 'glass': 501, 'metal': 409, 'paper': 594, 'plastic': 480}
- Class percentages: {'cardboard': 16.88, 'glass': 20.99, 'metal': 17.13, 'paper': 24.88, 'plastic': 20.11}
- Imbalance ratio (largest/smallest class): 1.47
- Split counts: {'train': 1670, 'val': 358, 'test': 359}
- Leakage check: {'files_in_more_than_one_split': 0, 'identical_image_groups_across_splits': 0, 'duplicate_groups_total': 0}

## Final CNN (augmentation + dropout) on the test set

Accuracy 0.7103 | Precision (macro) 0.7156 | Recall (macro) 0.7005 | F1 (macro) 0.7045 | test images: 359

```
              precision    recall  f1-score   support

   cardboard     0.8846    0.7667    0.8214        60
       glass     0.6022    0.7368    0.6627        76
       metal     0.6364    0.5645    0.5983        62
       paper     0.7938    0.8652    0.8280        89
     plastic     0.6613    0.5694    0.6119        72

    accuracy                         0.7103       359
   macro avg     0.7156    0.7005    0.7045       359
weighted avg     0.7146    0.7103    0.7089       359
```

## Ablation study (CO5)

| experiment | augmentation | dropout | test_accuracy | precision_macro | recall_macro | f1_macro | best_epoch | best_val_accuracy |
|---|---|---|---|---|---|---|---|---|
| baseline | False | 0.0 | 0.7075 | 0.7185 | 0.7 | 0.7025 | 4 | 0.6983 |
| augmentation | True | 0.0 | 0.6797 | 0.7001 | 0.6812 | 0.6775 | 10 | 0.7318 |
| augmentation_dropout | True | 0.5 | 0.7103 | 0.7156 | 0.7005 | 0.7045 | 10 | 0.7291 |

**Automatic reading of the table** (single run, one seed, so small differences are not conclusive):

- Adding augmentation reduced test accuracy by 2.8 percentage points (0.7075 -> 0.6797).
- Adding dropout on top of augmentation improved test accuracy by 3.1 percentage points (0.6797 -> 0.7103).
- Both together versus the baseline made almost no difference (under 1 percentage point, which is within normal run-to-run noise) (0.7075 -> 0.7103).

## Model comparison (CO4)

| model | accuracy | precision_macro | recall_macro | f1_macro | trainable_params | note |
|---|---|---|---|---|---|---|
| HOG + SVM | 0.6212 | 0.6259 | 0.6091 | 0.6129 | nan | HOG(1764 features)+RBF-SVM, C=1 |
| MLP (flatten) | 0.532 | 0.5468 | 0.5348 | 0.5323 | 12616709.0 | no augmentation, no convolution |
| CNN (aug + dropout) | 0.7103 | 0.7156 | 0.7005 | 0.7045 | 4288325.0 | final model from train.py |

Best test accuracy: **CNN (aug + dropout)** (0.7103).
