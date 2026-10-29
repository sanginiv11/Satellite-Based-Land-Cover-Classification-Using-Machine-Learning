# Beyond the Pixels: Satellite-Based Land-Cover Classification Using Machine Learning

Multiclass classification of land cover from Landsat multispectral pixel neighbourhoods, comparing 20+ classifiers from trivial baselines to boosted ensembles and stacking. The project also asks a specific question: **does looking at a pixel's neighbours help, compared with using the pixel alone?**

**Dataset:** [Statlog (Landsat Satellite)](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite), UCI Machine Learning Repository
**Environment:** Python, scikit-learn; runs top to bottom on Google Colab (the notebook downloads the data itself)

## Problem

Interpreting satellite imagery by hand is slow, and several land-cover classes (especially the grey soils) have overlapping spectral signatures. The goal is to classify the land cover of a pixel from numerical spectral measurements and to understand where and why models fail.

## Dataset

- Landsat MSS imagery of an 82 x 100 pixel area in Australia; labels come from an actual site visit.
- Each row is a **3 x 3 pixel neighbourhood x 4 spectral bands = 36 features** (8-bit values). The label is the class of the central pixel.
- 4,435 training rows and 2,000 test rows (the official split).
- 6 classes: red soil, cotton crop, grey soil, damp grey soil, soil with vegetation stubble, very damp grey soil.

**Evaluation protocol.** The dataset authors warn against cross-validation because neighbouring windows overlap (they share 6 of 9 pixels), so random re-splitting leaks near-duplicates. This project follows that advice:
- The supplied test set is used **only once per model** for final evaluation.
- Hyperparameters are tuned on a single stratified validation split from the training file.
- The champion model is chosen on **validation** accuracy, not test accuracy.

## Approach

1. **Data quality checks and EDA:** class balance, spectral signatures, correlation structure, outliers, PCA / LDA / t-SNE views.
2. **Three feature sets:**
   - `center`: 4 bands of the central pixel only
   - `full`: all 36 raw values
   - `engineered`: 36 raw values + 18 derived features (neighbourhood mean/std/range, local contrast, NDVI, brightness, NIR/Red ratio)
3. **Models, by family:**
   - Baselines (dummy classifiers)
   - Nearest neighbour (KNN)
   - Probabilistic (LDA, QDA, Gaussian Naive Bayes, a Gaussian Bayes classifier written from scratch, cost-sensitive Bayes decisions)
   - Linear (logistic regression, perceptron, ridge, linear SVM)
   - Decision trees (with pre- and post-pruning)
   - Ensembles (bagging, random forest, extra trees, AdaBoost, gradient boosting, HistGradientBoosting, XGBoost, LightGBM)
   - RBF SVM, MLP, voting and stacking
4. **Evaluation:** accuracy, balanced accuracy, macro F1, Cohen's kappa, log loss, ROC / PR curves, McNemar's test, bootstrap confidence intervals, calibration, learning curves.
5. **Error analysis** of the champion model.

## Results

Test set of 2,000 pixels:

| Model | Test accuracy | Macro F1 | Kappa |
|---|---|---|---|
| HistGradientBoosting (engineered features), champion | 0.9190 | 0.9066 | 0.9003 |
| SVM RBF (engineered features) | 0.9190 | 0.9040 | 0.9004 |
| Soft voting (KNN + SVM + RF + HGB + QDA) | 0.9160 | 0.9010 | 0.8965 |
| LightGBM (tuned) | 0.9140 | 0.9019 | 0.8942 |
| KNN, k = 5 (baseline) | 0.9045 | n/a | n/a |
| Unpruned decision tree | 0.8490 | n/a | n/a |
| Gaussian Naive Bayes | 0.7965 | n/a | n/a |
| Majority-class baseline | 0.2305 | n/a | n/a |

### Key findings

- **Land cover can be classified accurately.** The best models reach about 91 to 92% accuracy and kappa around 0.90, against 23% for always predicting the majority class.
- **Neighbours help every model family.** Adding the 8 surrounding pixels improved test accuracy by 8.6 points (gradient boosting), 8.1 (KNN), 7.8 (random forest) and 6.3 (SVM), but only 0.6 to 1.1 for Naive Bayes and logistic regression. The likely mechanisms are noise averaging and spatial context.
- **The top models are statistically tied.** Bootstrap intervals overlap heavily and McNemar's test usually cannot separate the champion from the next few models. The honest conclusion is "a group of flexible non-linear models reaches roughly 91 to 92%", not "model X is best".
- **Naive Bayes fails** because its independence assumption is badly violated: neighbouring pixels are correlated above 0.9.
- **Most errors are inherent to the data.** 67.9% of the champion's errors are confusions within the grey-soil group. Soil moisture is continuous but was cut into three classes, so pixels near the boundaries are genuinely ambiguous. Damp grey soil has the lowest F1 (0.745).
- **A reject option is practical.** Accepting only predictions with confidence of at least 0.9 keeps 91.3% of pixels at 95.95% accuracy, leaving the rest for manual review.

The results are also compared with the original 1994 Statlog study, where KNN was best at about 9.4% test error; here the best models reach about 8.1% error.


## How to run

**Colab (easiest):** upload the notebook and choose *Runtime > Run all*. The dataset is downloaded automatically.

**Locally:**
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost lightgbm
jupyter notebook satellite_land_cover_classification_executed.ipynb
```
XGBoost and LightGBM are optional; the notebook skips them if they are not installed.

## Repository contents

```
.
├── satellite_land_cover_classification_executed.ipynb   # full analysis with outputs
├── README.md
└── (generated when you run it: model_results.csv and fig_*.png)
```

## Limitations and future work

- The data covers a tiny area from one scene and one sensor, so results may not transfer to other regions or sensors.
- The validation split is mildly optimistic because windows overlap. In this run the validation-minus-test gap was essentially zero (+0.0005).
- Future work: spatially blocked splits on the full image, CNNs on larger patches, more spectral bands (Landsat 8/9, Sentinel-2) and time series, ordinal modelling of soil moisture.

## References

1. Srinivasan, A. (1993). *Statlog (Landsat Satellite)*. UCI Machine Learning Repository. https://doi.org/10.24432/C55887
2. Michie, D., Spiegelhalter, D. J., and Taylor, C. C. (1994). *Machine Learning, Neural and Statistical Classification*. Ellis Horwood.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.

## Author

Sangini Verma, B.Tech Electronics & Telecommunication Engineering, Symbiosis Institute of Technology, Pune
