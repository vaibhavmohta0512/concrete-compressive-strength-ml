# Project Report: Concrete Compressive Strength Prediction

> **Before submission:** Add student details and insert run-generated scores/figures. The pipeline deliberately does not fabricate results.

## Project title and student details
**Title:** Comparative Performance Analysis of Machine Learning Models for Concrete Compressive Strength Prediction  
**Student:** __________________ | **Roll No.:** __________________ | **Course/Faculty:** __________________

## Abstract
This project predicts concrete compressive strength (MPa) from seven material proportions and curing age using the UCI Concrete Compressive Strength dataset. It performs EDA, data-quality checks, leakage-aware preprocessing, compares Linear Regression, Random Forest and Gradient Boosting, tunes Random Forest with RandomizedSearchCV, evaluates with MAE/MSE/RMSE/R², and supplies a Streamlit interface. Insert actual results from `outputs/results_summary.md` after executing the pipeline.

## Literature review
Yeh (1998) introduced the concrete strength dataset and investigated neural networks for strength modelling, motivating nonlinear prediction. Linear Regression provides an interpretable baseline but may underfit nonlinear relationships. Random Forest (Breiman, 2001) averages randomized trees and can capture nonlinearities. Gradient Boosting (Friedman, 2001) builds trees sequentially to reduce residual error. Scikit-learn pipelines and cross-validation support reproducible, leakage-aware model selection (Pedregosa et al., 2011).

## Introduction and problem statement
Concrete compressive strength depends on mix proportions and curing age. The project estimates measured strength from known ingredient quantities and age. This is a **regression** problem because the target is continuous.

## Objectives
- Explore the data and create informative visualizations.
- Inspect missing values, types, duplicates and potential outliers.
- Preprocess without leakage.
- Compare three models on the same split.
- Tune a model and assess whether it improves test performance.
- Provide a Streamlit app and GitHub-ready repository.

## Dataset description
Source: [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), DOI [10.24432/C5PK67](https://doi.org/10.24432/C5PK67). It contains 1,030 rows, 8 quantitative input features and 1 quantitative target. Predictors: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate (all kg/m³), and age (days). Target: concrete compressive strength (MPa). UCI reports no missing attribute values. License: CC BY 4.0.

## EDA and visualization
The pipeline generates target distribution, input distributions, correlation heatmap, boxplots and feature-versus-target scatterplots in `outputs/figures/`. It writes actual row/column counts, dtypes, missing cells, duplicates and descriptive statistics to `outputs/data_audit.json`. Describe observed patterns after inspecting the generated figures; do not infer causality from correlations.

## Preprocessing
The data are split 80/20 with random state 42. Median imputation is inside each pipeline. Linear Regression uses StandardScaler; tree models do not require scaling. Imputation and scaling are fitted only on training data. All models use the same holdout set. RandomizedSearchCV uses five-fold CV on training data only. Outliers and duplicate rows are flagged for review rather than automatically deleted.

## Algorithms
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. Tuned Random Forest (improvement experiment)

## Experimental setup and workflow
Run `python src/train.py`. The script downloads the UCI dataset on first run, audits data, generates EDA, fits baselines, performs randomized hyperparameter search, compares metrics, saves plots and serializes the best model. The seed is 42.

## Performance evaluation and comparison
MAE, MSE and RMSE measure error (lower is better); R² measures performance relative to a mean-prediction baseline (higher is generally better). Accuracy, precision, recall, F1, confusion matrix and ROC-AUC are not appropriate for this continuous regression target.

Insert values from `outputs/final_model_comparison.csv` and figures from `outputs/figures/`. Do not use training scores as final performance.

## Hyperparameter tuning
RandomizedSearchCV samples 25 Random Forest configurations over number of trees, depth, split/leaf sizes and max features. It optimizes cross-validated RMSE on training data. Compare `outputs/tuning_comparison.csv` and `outputs/tuning_details.json`; tuning may not improve holdout performance and this should be reported honestly.

## Streamlit deployment
Run `streamlit run app.py` after training. The app accepts eight input features and returns an estimated strength in MPa. Add a hosted URL only after successful deployment: __________________.

## GitHub
Include the notebook, Python source, requirements.txt, README, report, source link, and generated model comparison tables/figures. Repository URL: __________________.

## Results and discussion
Complete this section with the actual run results: winning model, holdout MAE/RMSE/R², tuned-versus-baseline change, feature importance and residual behaviour. The model is an educational estimate and not a replacement for laboratory testing.

## Conclusion and future scope
The project demonstrates the full regression workflow from EDA to model comparison and deployment. Future work could use repeated cross-validation, grouped split by mix composition, uncertainty estimation and independent laboratory validation. Do not use this model for structural design without qualified review and appropriate testing.

## References
1. Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. *Cement and Concrete Research, 28*(12), 1797–1808. https://doi.org/10.1016/S0008-8846(98)00165-3
2. Yeh, I.-C. (1998). *Concrete Compressive Strength*. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK67
3. Breiman, L. (2001). Random Forests. *Machine Learning, 45*, 5–32. https://doi.org/10.1023/A:1010933404324
4. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
6. Streamlit documentation: https://docs.streamlit.io/
