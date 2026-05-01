import scipy as sp
from .interlinked import stats as rs_stats


#--| Constants |------------------------------------------------------------------------#



#--| Reshaping |------------------------------------------------------------------------#

# Calculates the Pearson Correlation and its p-value between 2 arrays
pearson  = rs_stats.pearson

# Calculates the Spearman Rank Correlation and its p-value between 2 arrays
spearman = rs_stats.spearman

# Calculates the Phi Coefficient and its p-value between 2 binary arrays
def phi(x, y):
    assert x.shape == y.shape
    if x.dtype != bool:
        x = x.astype(bool)
    if y.dtype != bool:
        y = y.astype(bool)

    f11 = np.sum( x *  y)
    f00 = np.sum(~x * ~y)
    f10 = np.sum( x * ~y)
    f01 = np.sum(~x *  y)

    f1_ = np.sum( x)
    f0_ = np.sum(~x)
    f_1 = np.sum( y)
    f_0 = np.sum(~y)

    phi = (f11*f00 - f01*f10) / np.sqrt(f1_*f0_ * f_1*f_0)
    pval = sp.stats.chi2.sf(len(x) * phi^2, df=1)
    return phi, pval

# Calculates a quantile-based bin edges for a specified number of bins
def quantile_bins(x, n_bins):
    quantiles = np.linspace(0, 100, n_bins+1)
    edges = np.percentile(x, quantiles)
    return edges





