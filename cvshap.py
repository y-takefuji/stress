import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from scipy.stats import spearmanr
import shap
import warnings
warnings.filterwarnings('ignore')

#─────────────────────────────────────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('StressLevelDataset.csv')
X  = df.drop('stress_level', axis=1)
y  = df['stress_level']
n_features = X.shape[1]

print("Dataset Shape:", X.shape)
print("\nTarget Distribution:")
print(y.value_counts().sort_index())
print()

# 100 shared random instances for SHAP
np.random.seed(42)
sample_idx = np.random.choice(len(X), size=100, replace=False)
X_sample   = X.iloc[sample_idx].reset_index(drop=True)

results = []

# ─────────────────────────────────────────────────────────────────────────────
# Helper – robust mean |SHAP| per feature
# ─────────────────────────────────────────────────────────────────────────────
def shap_top_features(shap_raw, feature_names):
    """
    Accepts every array layout SHAP may return.
    Finds whichever axis == n_features, moves it to axis-0,
    flattens everything else, returns mean |SHAP| per feature.
    """
    n = len(feature_names)

    # unwrap Explanation object (newest SHAP API)
    if hasattr(shap_raw, 'values'):
        shap_raw = shap_raw.values

    # list of arrays -> stack to ndarray
    if isinstance(shap_raw, list):
        shap_raw = np.array(shap_raw)

    arr = np.array(shap_raw, dtype=float)

    # find the feature axis (last axis whose size == n_features)
    feature_axes = [i for i, s in enumerate(arr.shape) if s == n]
    if len(feature_axes) == 0:
        raise ValueError(
            f"No axis of length n_features={n} in array shape {arr.shape}."
        )
    feat_ax = feature_axes[-1]

    # move feature axis to 0, flatten the rest
    arr      = np.moveaxis(arr, feat_ax, 0)   # (n_features, ...)
    arr      = arr.reshape(n, -1)             # (n_features, n_other)
    mean_abs = np.abs(arr).mean(axis=1)       # (n_features,)

    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Random Forest Feature Selection
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Random Forest Feature Selection")
print("=" * 60)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

rf_importance = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=False)

rf_top8     = list(rf_importance.head(8).index)
rf_cv_score = cross_val_score(
    rf_model, X[rf_top8], y, cv=5, scoring='accuracy'
).mean()

print(f"Top 8 features : {rf_top8}")
print(f"CV Accuracy    : {rf_cv_score:.4f}")

X_rf_reduced     = X.drop(rf_top8[0], axis=1)
rf_model_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_reduced.fit(X_rf_reduced, y)
rf_imp_reduced= pd.Series(
    rf_model_reduced.feature_importances_, index=X_rf_reduced.columns
).sort_values(ascending=False)
rf_top7 = list(rf_imp_reduced.head(7).index)
print(f"Top 7 (after removing '{rf_top8[0]}'): {rf_top7}\n")

results.append({
    'method'         : 'RF',
    'CV_accuracy'    : rf_cv_score,
    'top_8_features' : ', '.join(rf_top8),
    'top_7_features' : ', '.join(rf_top7),
})

# ─────────────────────────────────────────────────────────────────────────────
# 2. XGBoost Feature Selection
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("2. XGBoost Feature Selection")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=100, random_state=42, eval_metric='mlogloss'
)
xgb_model.fit(X, y)

xgb_importance = pd.Series(
    xgb_model.feature_importances_, index=X.columns
).sort_values(ascending=False)

xgb_top8     = list(xgb_importance.head(8).index)
xgb_cv_score = cross_val_score(
    xgb_model, X[xgb_top8], y, cv=5, scoring='accuracy'
).mean()

print(f"Top 8 features : {xgb_top8}")
print(f"CV Accuracy    : {xgb_cv_score:.4f}")

X_xgb_reduced     = X.drop(xgb_top8[0], axis=1)
xgb_model_reduced = XGBClassifier(
    n_estimators=100, random_state=42, eval_metric='mlogloss'
)
xgb_model_reduced.fit(X_xgb_reduced, y)
xgb_imp_reduced = pd.Series(
    xgb_model_reduced.feature_importances_, index=X_xgb_reduced.columns
).sort_values(ascending=False)
xgb_top7 = list(xgb_imp_reduced.head(7).index)
print(f"Top 7 (after removing '{xgb_top8[0]}'): {xgb_top7}\n")

results.append({
    'method'         : 'XGB',
    'CV_accuracy'    : xgb_cv_score,
    'top_8_features' : ', '.join(xgb_top8),
    'top_7_features' : ', '.join(xgb_top7),
})

# ─────────────────────────────────────────────────────────────────────────────
# 3. Logistic Regression Feature Selection
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("3. Logistic Regression Feature Selection")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X, y)

lr_importance = pd.Series(
    np.abs(lr_model.coef_).mean(axis=0), index=X.columns
).sort_values(ascending=False)

lr_top8     = list(lr_importance.head(8).index)
lr_cv_score = cross_val_score(
    lr_model, X[lr_top8], y, cv=5, scoring='accuracy'
).mean()

print(f"Top 8 features : {lr_top8}")
print(f"CV Accuracy    : {lr_cv_score:.4f}")

X_lr_reduced     = X.drop(lr_top8[0], axis=1)
lr_model_reduced = LogisticRegression(max_iter=1000, random_state=42)
lr_model_reduced.fit(X_lr_reduced, y)
lr_imp_reduced = pd.Series(
    np.abs(lr_model_reduced.coef_).mean(axis=0), index=X_lr_reduced.columns
).sort_values(ascending=False)
lr_top7 = list(lr_imp_reduced.head(7).index)
print(f"Top 7 (after removing '{lr_top8[0]}'): {lr_top7}\n")

results.append({
    'method'         : 'LR',
    'CV_accuracy'    : lr_cv_score,
    'top_8_features' : ', '.join(lr_top8),
    'top_7_features' : ', '.join(lr_top7),
})

# ─────────────────────────────────────────────────────────────────────────────
# 4. RF-SHAP
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("4. RF-SHAP Feature Selection")
print("=" * 60)

print("Computing SHAP for RF on100 instances...")
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_raw= rf_explainer.shap_values(X_sample)
print(f"  Raw SHAP shape : {np.array(rf_shap_raw).shape}")

rf_shap_imp= shap_top_features(rf_shap_raw, X.columns)
rf_shap_top8 = list(rf_shap_imp.head(8).index)
rf_shap_cv   = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X[rf_shap_top8], y, cv=5, scoring='accuracy'
).mean()

print(f"Top 8 features : {rf_shap_top8}")
print(f"CV Accuracy    : {rf_shap_cv:.4f}")

X_rf_shap_reduced = X.drop(rf_shap_top8[0], axis=1)
rf_model_shap_red = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_shap_red.fit(X_rf_shap_reduced, y)
rf_shap_raw_red= shap.TreeExplainer(rf_model_shap_red).shap_values(
    X_rf_shap_reduced.iloc[sample_idx].reset_index(drop=True)
)
rf_shap_imp_red  = shap_top_features(rf_shap_raw_red, X_rf_shap_reduced.columns)
rf_shap_top7     = list(rf_shap_imp_red.head(7).index)
print(f"Top 7 (after removing '{rf_shap_top8[0]}'): {rf_shap_top7}\n")

results.append({
    'method'         : 'RF-SHAP',
    'CV_accuracy'    : rf_shap_cv,
    'top_8_features' : ', '.join(rf_shap_top8),
    'top_7_features' : ', '.join(rf_shap_top7),
})

# ─────────────────────────────────────────────────────────────────────────────
# 5. XGB-SHAP
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("5. XGB-SHAP Feature Selection")
print("=" * 60)

print("Computing SHAP for XGB on 100 instances...")
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_raw  = xgb_explainer.shap_values(X_sample)
print(f"  Raw SHAP shape : {np.array(xgb_shap_raw).shape}")

xgb_shap_imp  = shap_top_features(xgb_shap_raw, X.columns)
xgb_shap_top8 = list(xgb_shap_imp.head(8).index)
xgb_shap_cv   = cross_val_score(
    XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
    X[xgb_shap_top8], y, cv=5, scoring='accuracy'
).mean()

print(f"Top 8 features : {xgb_shap_top8}")
print(f"CV Accuracy    : {xgb_shap_cv:.4f}")

X_xgb_shap_reduced  = X.drop(xgb_shap_top8[0], axis=1)
xgb_model_shap_red= XGBClassifier(
    n_estimators=100, random_state=42, eval_metric='mlogloss'
)
xgb_model_shap_red.fit(X_xgb_shap_reduced, y)
xgb_shap_raw_red    = shap.TreeExplainer(xgb_model_shap_red).shap_values(
    X_xgb_shap_reduced.iloc[sample_idx].reset_index(drop=True)
)
xgb_shap_imp_red = shap_top_features(xgb_shap_raw_red, X_xgb_shap_reduced.columns)
xgb_shap_top7= list(xgb_shap_imp_red.head(7).index)
print(f"Top 7 (after removing '{xgb_shap_top8[0]}'): {xgb_shap_top7}\n")

results.append({
    'method'         : 'XGB-SHAP',
    'CV_accuracy'    : xgb_shap_cv,
    'top_8_features' : ', '.join(xgb_shap_top8),
    'top_7_features' : ', '.join(xgb_shap_top7),
})

# ─────────────────────────────────────────────────────────────────────────────
# 6. LR-SHAP
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("6. LR-SHAP Feature Selection")
print("=" * 60)

print("Computing SHAP for LR on 100 instances...")
background= shap.sample(X, 50, random_state=42)
lr_explainer = shap.LinearExplainer(lr_model, background)
lr_shap_raw  = lr_explainer.shap_values(X_sample)
print(f"  Raw SHAP shape : {np.array(lr_shap_raw).shape}")

lr_shap_imp  = shap_top_features(lr_shap_raw, X.columns)
lr_shap_top8 = list(lr_shap_imp.head(8).index)
lr_shap_cv   = cross_val_score(
    LogisticRegression(max_iter=1000, random_state=42),
    X[lr_shap_top8], y, cv=5, scoring='accuracy'
).mean()

print(f"Top 8 features : {lr_shap_top8}")
print(f"CV Accuracy    : {lr_shap_cv:.4f}")

X_lr_shap_reduced  = X.drop(lr_shap_top8[0], axis=1)
lr_model_shap_red  = LogisticRegression(max_iter=1000, random_state=42)
lr_model_shap_red.fit(X_lr_shap_reduced, y)
background_red= shap.sample(X_lr_shap_reduced, 50, random_state=42)
lr_explainer_red   = shap.LinearExplainer(lr_model_shap_red, background_red)
lr_shap_raw_red    = lr_explainer_red.shap_values(
    X_lr_shap_reduced.iloc[sample_idx].reset_index(drop=True)
)
lr_shap_imp_red = shap_top_features(lr_shap_raw_red, X_lr_shap_reduced.columns)
lr_shap_top7    = list(lr_shap_imp_red.head(7).index)
print(f"Top 7 (after removing '{lr_shap_top8[0]}'): {lr_shap_top7}\n")

results.append({
    'method'         : 'LR-SHAP',
    'CV_accuracy'    : lr_shap_cv,
    'top_8_features' : ', '.join(lr_shap_top8),
    'top_7_features' : ', '.join(lr_shap_top7),
})

# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature Agglomeration
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("7. Feature Agglomeration Feature Selection")
print("=" * 60)

fa = FeatureAgglomeration(n_clusters=min(n_features, 20))
fa.fit(X)

fa_scores= {col: abs(np.corrcoef(X[col], y)[0, 1]) for col in X.columns}
fa_importance = pd.Series(fa_scores).sort_values(ascending=False)
fa_top8       = list(fa_importance.head(8).index)

fa_cv_score = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X[fa_top8], y, cv=5, scoring='accuracy'
).mean()

print(f"Top 8 features         : {fa_top8}")
print(f"CV Accuracy (using RF) : {fa_cv_score:.4f}")

X_fa_reduced   = X.drop(fa_top8[0], axis=1)
fa_scores_red  = {
    col: abs(np.corrcoef(X_fa_reduced[col], y)[0, 1])
    for col in X_fa_reduced.columns
}
fa_imp_reduced = pd.Series(fa_scores_red).sort_values(ascending=False)
fa_top7= list(fa_imp_reduced.head(7).index)
print(f"Top 7 (after removing '{fa_top8[0]}'): {fa_top7}\n")

results.append({
    'method'         : 'FA',
    'CV_accuracy'    : fa_cv_score,
    'top_8_features' : ', '.join(fa_top8),
    'top_7_features' : ', '.join(fa_top7),
})

# ─────────────────────────────────────────────────────────────────────────────
# 8. HVGS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("8. Highly Variable Gene Selection (HVGS)")
print("=" * 60)

hvgs_variance = X.var().sort_values(ascending=False)
hvgs_top8     = list(hvgs_variance.head(8).index)

hvgs_cv_score = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X[hvgs_top8], y, cv=5, scoring='accuracy'
).mean()

print(f"Top 8 features         : {hvgs_top8}")
print(f"CV Accuracy (using RF) : {hvgs_cv_score:.4f}")

X_hvgs_reduced   = X.drop(hvgs_top8[0], axis=1)
hvgs_var_reduced = X_hvgs_reduced.var().sort_values(ascending=False)
hvgs_top7        = list(hvgs_var_reduced.head(7).index)
print(f"Top 7 (after removing '{hvgs_top8[0]}'): {hvgs_top7}\n")

results.append({
    'method'         : 'HVGS',
    'CV_accuracy'    : hvgs_cv_score,
    'top_8_features' : ', '.join(hvgs_top8),
    'top_7_features' : ', '.join(hvgs_top7),
})

# ─────────────────────────────────────────────────────────────────────────────
# 9. Spearman Correlation
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("9. Spearman Correlation Feature Selection")
print("=" * 60)

spearman_scores = {col: abs(spearmanr(X[col], y)[0]) for col in X.columns}
spearman_imp= pd.Series(spearman_scores).sort_values(ascending=False)
spearman_top8   = list(spearman_imp.head(8).index)

spearman_cv_score = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X[spearman_top8], y, cv=5, scoring='accuracy'
).mean()

print(f"Top 8 features         : {spearman_top8}")
print(f"CV Accuracy (using RF) : {spearman_cv_score:.4f}")

X_spearman_reduced= X.drop(spearman_top8[0], axis=1)
spearman_scores_red = {
    col: abs(spearmanr(X_spearman_reduced[col], y)[0])
    for col in X_spearman_reduced.columns
}
spearman_imp_red = pd.Series(spearman_scores_red).sort_values(ascending=False)
spearman_top7    = list(spearman_imp_red.head(7).index)
print(f"Top 7 (after removing '{spearman_top8[0]}'): {spearman_top7}\n")

results.append({
    'method'         : 'Spearman',
    'CV_accuracy'    : spearman_cv_score,
    'top_8_features' : ', '.join(spearman_top8),
    'top_7_features' : ', '.join(spearman_top7),
})

# ─────────────────────────────────────────────────────────────────────────────
# Summary table –9 rows x 4 columns
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df['CV_accuracy'] = results_df['CV_accuracy'].round(4)

print("\n" + "=" * 60)
print("=== Summary Table ===")
print("=" * 60)
print(results_df.to_string(index=False))

results_df.to_csv('result.csv', index=False)
print("\nResults saved to result.csv")