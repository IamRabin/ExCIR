# ============================================================
# Controlled MI vs ExCIR redundancy experiment
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# -----------------------------
# ExCIR implementation
# -----------------------------
def cir_score_1d(x, y, eps=1e-12):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    x_hat = x.mean()
    y_hat = y.mean()
    m = 0.5 * (x_hat + y_hat)

    num = len(x) * ((x_hat - m) ** 2 + (y_hat - m) ** 2)
    den = np.sum((x - m) ** 2) + np.sum((y - m) ** 2) + eps

    return float(num / den)

def excir_scores(X, y_model):
    return np.array([cir_score_1d(X[:, j], y_model) for j in range(X.shape[1])])

# -----------------------------
# Synthetic redundancy setup
# -----------------------------
n = 5000
sigma_redundant = 0.05
sigma_y = 0.20

X1 = rng.normal(0, 1, n)
X2 = X1 + rng.normal(0, sigma_redundant, n)   # redundant correlated feature
X3 = rng.normal(0, 1, n)                      # irrelevant noise

y = X1 + rng.normal(0, sigma_y, n)

X = np.column_stack([X1, X2, X3])
feature_names = np.array(["X1_predictive", "X2_redundant", "X3_noise"])

# Standardize for stable comparison
X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

# -----------------------------
# Train predictive model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# Model output/logit equivalent for regression
y_model = model.predict(X)

# -----------------------------
# Compute MI and ExCIR
# -----------------------------
mi_scores = mutual_info_regression(X, y, random_state=SEED)
excir = excir_scores(X, y_model)

# -----------------------------
# Ranking
# -----------------------------
mi_rank_order = np.argsort(-mi_scores)
excir_rank_order = np.argsort(-excir)

def ranks_from_scores(scores):
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks

mi_ranks = ranks_from_scores(mi_scores)
excir_ranks = ranks_from_scores(excir)

# -----------------------------
# Summary table
# -----------------------------
df = pd.DataFrame({
    "Feature": feature_names,
    "True Role": ["Predictive", "Redundant", "Noise"],
    "MI Score": mi_scores,
    "MI Rank": mi_ranks,
    "ExCIR Score": excir,
    "ExCIR Rank": excir_ranks,
    "Model Coefficient": model.coef_,
})

print("\nFeature-importance under redundancy:")
print(df.sort_values("ExCIR Rank").to_string(index=False))

# -----------------------------
# Rank agreement
# -----------------------------
spearman = stats.spearmanr(mi_scores, excir).statistic
kendall = stats.kendalltau(mi_rank_order, excir_rank_order).statistic

print("\nRank agreement:")
print(f"Spearman(MI, ExCIR) = {spearman:.4f}")
print(f"Kendall(MI, ExCIR)  = {kendall:.4f}")

# -----------------------------
# Optional LaTeX table
# -----------------------------
latex_table = df[["Feature", "True Role", "MI Rank", "ExCIR Rank"]].copy()
latex_table["Feature"] = [r"$X_1$", r"$X_2$", r"$X_3$"]

print("\nLaTeX table:")
print(latex_table.to_latex(index=False, escape=False))
