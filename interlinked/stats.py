import numpy as np
from .interlinked import stats as rs_stats


#--| Constants |------------------------------------------------------------------------#



#--| Functions |------------------------------------------------------------------------#

# Calculates the Pearson Correlation and its p-value between 2 arrays
pearson_corr = rs_stats.pearson_corr

# Calculates the Spearman Rank Correlation and its p-value between 2 arrays
spearman_corr = rs_stats.spearman_corr

# Calculates the Phi Coefficient and its p-value between 2 binary arrays
phi_coef = rs_stats.phi_coef

# Calculates a quantile-based bin edges for a specified number of bins
def quantile_bins(x, n_bins):
    quantiles = np.linspace(0, 100, n_bins+1)
    edges = np.percentile(x, quantiles)
    return edges





