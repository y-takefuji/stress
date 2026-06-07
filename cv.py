import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('StressLevelDataset.csv')
X = df.drop('stress_level', axis=1)
y = df['stress_level']

print("Dataset Shape:", X.shape)
print("\nTarget Distribution:")
print(y.value_counts().sort_index())
print()

# Initialize results storage
results = []

# 1. Random Forest Feature Selection
print("1. Random Forest Feature Selection")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_top8 = list(rf_importance.head(8).index)
rf_cv_score = cross_val_score(rf_model, X[rf_top8], y, cv=5, scoring='accuracy').mean()
print(f"Top 8 features: {rf_top8}")
print(f"CV Accuracy: {rf_cv_score:.4f}")

# Remove highest feature and re-select top 7
X_rf_reduced = X.drop(rf_top8[0], axis=1)
rf_model_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_reduced.fit(X_rf_reduced, y)
rf_importance_reduced = pd.Series(rf_model_reduced.feature_importances_, index=X_rf_reduced.columns).sort_values(ascending=False)
rf_top7 = list(rf_importance_reduced.head(7).index)
print(f"Top 7 features (after removing {rf_top8[0]}): {rf_top7}\n")

results.append({
    'method': 'RF',
    'CV_accuracy': rf_cv_score,
    'top_8_features': ', '.join(rf_top8),
    'top_7_features': ', '.join(rf_top7)
})

# 2. XGBoost Feature Selection
print("2. XGBoost Feature Selection")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
xgb_model.fit(X, y)
xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
xgb_top8 = list(xgb_importance.head(8).index)
xgb_cv_score = cross_val_score(xgb_model, X[xgb_top8], y, cv=5, scoring='accuracy').mean()
print(f"Top 8 features: {xgb_top8}")
print(f"CV Accuracy: {xgb_cv_score:.4f}")

# Remove highest feature and re-select top 7
X_xgb_reduced = X.drop(xgb_top8[0], axis=1)
xgb_model_reduced = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
xgb_model_reduced.fit(X_xgb_reduced, y)
xgb_importance_reduced = pd.Series(xgb_model_reduced.feature_importances_, index=X_xgb_reduced.columns).sort_values(ascending=False)
xgb_top7 = list(xgb_importance_reduced.head(7).index)
print(f"Top 7 features (after removing {xgb_top8[0]}): {xgb_top7}\n")

results.append({
    'method': 'XGBoost',
    'CV_accuracy': xgb_cv_score,
    'top_8_features': ', '.join(xgb_top8),
    'top_7_features': ', '.join(xgb_top7)
})

# 3. Linear Regression Feature Selection
print("3. Linear Regression Feature Selection")
lr_model = LinearRegression()
lr_model.fit(X, y)
lr_importance = pd.Series(np.abs(lr_model.coef_), index=X.columns).sort_values(ascending=False)
lr_top8 = list(lr_importance.head(8).index)
# Use regression for CV since LR is a regressor
from sklearn.metrics import make_scorer, mean_squared_error
lr_cv_score = -cross_val_score(lr_model, X[lr_top8], y, cv=5, scoring='neg_mean_squared_error').mean()
# Convert to accuracy-like metric (using R2 score instead)
from sklearn.metrics import r2_score
lr_cv_score = cross_val_score(lr_model, X[lr_top8], y, cv=5, scoring='r2').mean()
print(f"Top 8 features: {lr_top8}")
print(f"CV R2 Score: {lr_cv_score:.4f}")

# Remove highest feature and re-select top 7
X_lr_reduced = X.drop(lr_top8[0], axis=1)
lr_model_reduced = LinearRegression()
lr_model_reduced.fit(X_lr_reduced, y)
lr_importance_reduced = pd.Series(np.abs(lr_model_reduced.coef_), index=X_lr_reduced.columns).sort_values(ascending=False)
lr_top7 = list(lr_importance_reduced.head(7).index)
print(f"Top 7 features (after removing {lr_top8[0]}): {lr_top7}\n")

results.append({
    'method': 'LR',
    'CV_accuracy': lr_cv_score,
    'top_8_features': ', '.join(lr_top8),
    'top_7_features': ', '.join(lr_top7)
})

# 4. Feature Agglomeration
print("4. Feature Agglomeration Feature Selection")
# Fit FA with all features as clusters (no reduction yet)
fa = FeatureAgglomeration(n_clusters=min(X.shape[1], 20))  # Use reasonable number of clusters
fa.fit(X, y)

# Calculate feature importance based on correlation with target within each cluster
fa_labels = fa.labels_
feature_scores = {}

for feature_idx, feature_name in enumerate(X.columns):
    # Calculate correlation of each original feature with target
    corr = np.corrcoef(X[feature_name], y)[0, 1]
    feature_scores[feature_name] = abs(corr)

# Sort features by their individual correlation with target
fa_importance = pd.Series(feature_scores).sort_values(ascending=False)
fa_top8 = list(fa_importance.head(8).index)

# Use RF for cross-validation
rf_fa = RandomForestClassifier(n_estimators=100, random_state=42)
fa_cv_score = cross_val_score(rf_fa, X[fa_top8], y, cv=5, scoring='accuracy').mean()
print(f"Top 8 features: {fa_top8}")
print(f"CV Accuracy (using RF): {fa_cv_score:.4f}")

# Remove highest feature and re-select top 7
X_fa_reduced = X.drop(fa_top8[0], axis=1)
fa_reduced = FeatureAgglomeration(n_clusters=min(X_fa_reduced.shape[1], 20))
fa_reduced.fit(X_fa_reduced, y)

feature_scores_reduced = {}
for feature_idx, feature_name in enumerate(X_fa_reduced.columns):
    corr = np.corrcoef(X_fa_reduced[feature_name], y)[0, 1]
    feature_scores_reduced[feature_name] = abs(corr)

fa_importance_reduced = pd.Series(feature_scores_reduced).sort_values(ascending=False)
fa_top7 = list(fa_importance_reduced.head(7).index)

print(f"Top 7 features (after removing {fa_top8[0]}): {fa_top7}\n")

results.append({
    'method': 'FA',
    'CV_accuracy': fa_cv_score,
    'top_8_features': ', '.join(fa_top8),
    'top_7_features': ', '.join(fa_top7)
})

# 5. Highly Variable Gene Selection (HVGS) - using variance
print("5. Highly Variable Gene Selection (HVGS)")
hvgs_variance = X.var().sort_values(ascending=False)
hvgs_top8 = list(hvgs_variance.head(8).index)
# Use RF for cross-validation
rf_hvgs = RandomForestClassifier(n_estimators=100, random_state=42)
hvgs_cv_score = cross_val_score(rf_hvgs, X[hvgs_top8], y, cv=5, scoring='accuracy').mean()
print(f"Top 8 features: {hvgs_top8}")
print(f"CV Accuracy (using RF): {hvgs_cv_score:.4f}")

# Remove highest feature and re-select top 7
X_hvgs_reduced = X.drop(hvgs_top8[0], axis=1)
hvgs_variance_reduced = X_hvgs_reduced.var().sort_values(ascending=False)
hvgs_top7 = list(hvgs_variance_reduced.head(7).index)
print(f"Top 7 features (after removing {hvgs_top8[0]}): {hvgs_top7}\n")

results.append({
    'method': 'HVGS',
    'CV_accuracy': hvgs_cv_score,
    'top_8_features': ', '.join(hvgs_top8),
    'top_7_features': ', '.join(hvgs_top7)
})

# 6. Spearman Correlation Feature Selection
print("6. Spearman Correlation Feature Selection")
spearman_corr = {}
for col in X.columns:
    corr, _ = spearmanr(X[col], y)
    spearman_corr[col] = abs(corr)

spearman_importance = pd.Series(spearman_corr).sort_values(ascending=False)
spearman_top8 = list(spearman_importance.head(8).index)
# Use RF for cross-validation
rf_spearman = RandomForestClassifier(n_estimators=100, random_state=42)
spearman_cv_score = cross_val_score(rf_spearman, X[spearman_top8], y, cv=5, scoring='accuracy').mean()
print(f"Top 8 features: {spearman_top8}")
print(f"CV Accuracy (using RF): {spearman_cv_score:.4f}")

# Remove highest feature and re-select top 7
X_spearman_reduced = X.drop(spearman_top8[0], axis=1)
spearman_corr_reduced = {}
for col in X_spearman_reduced.columns:
    corr, _ = spearmanr(X_spearman_reduced[col], y)
    spearman_corr_reduced[col] = abs(corr)

spearman_importance_reduced = pd.Series(spearman_corr_reduced).sort_values(ascending=False)
spearman_top7 = list(spearman_importance_reduced.head(7).index)
print(f"Top 7 features (after removing {spearman_top8[0]}): {spearman_top7}\n")

results.append({
    'method': 'Spearman',
    'CV_accuracy': spearman_cv_score,
    'top_8_features': ', '.join(spearman_top8),
    'top_7_features': ', '.join(spearman_top7)
})

# Create summary table
results_df = pd.DataFrame(results)
results_df['CV_accuracy'] = results_df['CV_accuracy'].round(4)

print("\n=== Summary Table ===")
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('result.csv', index=False)
print("\nResults saved to result.csv")