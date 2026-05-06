import numpy as np
from .interlinked import stats as rs_stats


#--| Constants |------------------------------------------------------------------------#



#--| Functions |------------------------------------------------------------------------#

# Calculates the Pearson Correlation and its p-value between 2 arrays
def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return rs_stats.pearson_corr(x, y)

# Calculates the Spearman Rank Correlation and its p-value between 2 arrays
def spearman_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return rs_stats.spearman_corr(x, y)

# Calculates the Phi Coefficient and its p-value between 2 binary arrays
def phi_coef(x, y):
    x = np.asarray(x, dtype=bool)
    y = np.asarray(y, dtype=bool)
    return rs_stats.phi_coef(x, y)

# Calculates a quantile-based bin edges for a specified number of bins
def quantile_bins(x, n_bins):
    quantiles = np.linspace(0, 100, n_bins+1)
    edges = np.percentile(x, quantiles)
    return edges





