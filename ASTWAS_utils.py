import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Suppress numpy warnings for numerical computations (normal in SKAT calculations)
np.seterr(divide='ignore', invalid='ignore')

def _import_scipy_stats():
    """Lazy import scipy.stats"""
    from scipy import stats
    return stats

def _import_momentchi2():
    """Lazy import momentchi2 - optional dependency"""
    try:
        from momentchi2 import hbe
        return hbe
    except ImportError:
        return None

def check_class(obj, class_type):
    """Check if object is of the specified class type"""
    if not isinstance(class_type, list):
        class_type = [class_type]
    return any(isinstance(obj, t) for t in class_type)


def is_try_error(obj):
    """Check if object is a try-error"""
    return isinstance(obj, Exception)


def get_satterthwaite(muQ, varQ):
    """Get Satterthwaite approximation parameters"""
    a1 = varQ / muQ / 2
    a2 = muQ / a1
    return {"df": a2, "a": a1}


def get_liu_params_mod(c1):
    """Get Liu parameters for null approximation"""
    muQ = c1[0]
    sigmaQ = np.sqrt(2 * c1[1])
    s1 = c1[2] / (c1[1] ** (3 / 2))
    s2 = c1[3] / (c1[1] ** 2)

    beta1 = np.sqrt(8) * s1
    beta2 = 12 * s2
    type1 = 0

    if s1 ** 2 > s2:
        a = 1 / (s1 - np.sqrt(s1 ** 2 - s2))
        d = s1 * a ** 3 - a ** 2
        l = a ** 2 - 2 * d
    else:
        type1 = 1
        l = 1 / s2
        a = np.sqrt(l)
        d = 0

    muX = l + d
    sigmaX = np.sqrt(2) * a

    return {"l": l, "d": d, "muQ": muQ, "muX": muX, "sigmaQ": sigmaQ, "sigmaX": sigmaX}


# 添加get_liu_params函数(比mod版本略有不同)
def get_liu_params(c1):
    """Get Liu parameters for null approximation (original version)"""
    muQ = c1[0]
    sigmaQ = np.sqrt(2 * c1[1])
    s1 = c1[2] / (c1[1] ** (3 / 2))
    s2 = c1[3] / (c1[1] ** 2)

    beta1 = np.sqrt(8) * s1
    beta2 = 12 * s2
    type1 = 0

    if s1 ** 2 > s2:
        a = 1 / (s1 - np.sqrt(s1 ** 2 - s2))
        d = s1 * a ** 3 - a ** 2
        l = a ** 2 - 2 * d
    else:
        type1 = 1
        a = 1 / s1
        d = 0
        l = 1 / s1 ** 2

    muX = l + d
    sigmaX = np.sqrt(2) * a

    return {"l": l, "d": d, "muQ": muQ, "muX": muX, "sigmaQ": sigmaQ, "sigmaX": sigmaX}

def get_liu_params_mod_lambda(lambda_vals, df1=None):
    """
    Get Liu parameters from lambda values
    
    Parameters:
    ----------
    lambda_vals : array-like
        Eigenvalues
    df1 : array-like, optional
        Degrees of freedom
    
    Returns:
    -------
    dict
        Dictionary with Liu parameters
    """

    if df1 is None:
        df1 = np.ones(len(lambda_vals))

    c1 = np.zeros(4)

    for i in range(4):
        c1[i] = np.sum(lambda_vals ** (i+1) * df1)

    muQ = c1[0]
    sigmaQ = np.sqrt(2 * c1[1])
    s1 = c1[2] / (c1[1] ** (3/2))
    s2 = c1[3] / (c1[1] ** 2)

    beta1 = np.sqrt(8) * s1
    beta2 = 12 * s2
    type1 = 0

    if s1 ** 2 > s2:
        a = 1 / (s1 - np.sqrt(s1 ** 2 - s2))
        d = s1 * a ** 3 - a ** 2
        l = a ** 2 - 2 * d
    else:
        type1 = 1
        l = 1 / s2
        a = np.sqrt(l)
        d = 0

    muX = l + d
    sigmaX = np.sqrt(2) * a

    return {"l": l, "d": d, "muQ": muQ, "muX": muX, "sigmaQ": sigmaQ, "sigmaX": sigmaX}


def get_liu_pval_mod_lambda_zero(Q, muQ, muX, sigmaQ, sigmaX, l, d):
    """
    Handle p-value calculation when p-value is 0
    
    Parameters:
    ----------
    Q : float
        Test statistic
    muQ : float
        Mean of Q
    muX : float
        Mean of X
    sigmaQ : float
        Standard deviation of Q
    sigmaX : float
        Standard deviation of X
    l : float
        Liu parameter l
    d : float
        Liu parameter d
    
    Returns:
    -------
    str
        P-value message
    """
    stats = _import_scipy_stats()

    Q_Norm = (Q - muQ) / sigmaQ
    Q_Norm1 = Q_Norm * sigmaX + muX

    temp = np.array([0.05, 1e-10, 1e-20, 1e-30, 1e-40, 1e-50, 1e-60, 1e-70, 1e-80, 1e-90, 1e-100])

    out = stats.ncx2.isf(temp, df=l, nc=d)

    IDX = np.max(np.where(out < Q_Norm1)[0])

    pval_msg = f"Pvalue < {temp[IDX]:.6e}"

    return pval_msg


def impute(Z, impute_method):
    """
    Impute missing genotypes
    
    Parameters:
    ----------
    Z : array-like
        Genotype matrix
    impute_method : str
        Imputation method ("fixed", "random", "bestguess")
    
    Returns:
    -------
    array-like
        Imputed genotype matrix
    """
    p = Z.shape[1]

    if impute_method == "random":
        for i in range(p):
            idx = np.where(np.isnan(Z[:, i]))[0]
            if len(idx) > 0:
                maf1 = np.mean(Z[~np.isnan(Z[:, i]), i]) / 2
                Z[idx, i] = np.random.binomial(2, maf1, size=len(idx))

    elif impute_method == "fixed":
        for i in range(p):
            idx = np.where(np.isnan(Z[:, i]))[0]
            if len(idx) > 0:
                maf1 = np.mean(Z[~np.isnan(Z[:, i]), i]) / 2
                Z[idx, i] = 2 * maf1

    elif impute_method == "bestguess":
        for i in range(p):
            idx = np.where(np.isnan(Z[:, i]))[0]
            if len(idx) > 0:
                maf1 = np.mean(Z[~np.isnan(Z[:, i]), i]) / 2
                Z[idx, i] = np.round(2 * maf1)

    else:
        raise ValueError("Error: Imputation method should be \"fixed\", \"random\" or \"bestguess\"")

    return Z


def impute_r_version(Z, impute_method):
    """
    Impute missing genotypes using R-compatible logic
    
    Parameters:
    ----------
    Z : array-like
        Genotype matrix
    impute_method : str
        Imputation method
    
    Returns:
    -------
    array-like
        Imputed genotype matrix
    """
    p = Z.shape[1]

    if impute_method == "random":
        for i in range(p):
            idx = np.where(np.isnan(Z[:, i]))[0]
            if len(idx) > 0:
                non_missing = Z[~np.isnan(Z[:, i]), i]
                maf1 = np.mean(non_missing) / 2
                Z[idx, i] = np.random.binomial(2, maf1, size=len(idx))

    elif impute_method == "fixed":
        for i in range(p):
            idx = np.where(np.isnan(Z[:, i]))[0]
            if len(idx) > 0:
                non_missing = Z[~np.isnan(Z[:, i]), i]
                maf1 = np.mean(non_missing) / 2
                Z[idx, i] = 2 * maf1

    elif impute_method == "bestguess":
        for i in range(p):
            idx = np.where(np.isnan(Z[:, i]))[0]
            if len(idx) > 0:
                non_missing = Z[~np.isnan(Z[:, i]), i]
                maf1 = np.mean(non_missing) / 2
                Z[idx, i] = np.round(2 * maf1)

    else:
        raise ValueError("Error: Imputation method should be \"fixed\", \"random\" or \"bestguess\"")

    return Z


def single_snp_info(Z):
    """Get SNP information"""
    snplist = np.sum(Z, axis=0)

    if isinstance(Z, pd.DataFrame):
        snplist = pd.Series(snplist, index=Z.columns)

    return snplist


def beta_weights(MAF, weights_beta):
    """
    Calculate beta weights for MAFs
    
    Parameters:
    ----------
    MAF : array-like
        Minor allele frequencies
    weights_beta : tuple
        Beta distribution parameters
    
    Returns:
    -------
    array-like
        Beta weights
    """
    stats = _import_scipy_stats()

    n = len(MAF)
    weights = np.zeros(n)

    idx_0 = np.where(MAF == 0)[0]

    if len(idx_0) == n:
        raise ValueError("No polymorphic SNPs")
    elif len(idx_0) == 0:
        weights = stats.beta.pdf(MAF, weights_beta[0], weights_beta[1])
    else:
        non_zero_idx = np.setdiff1d(np.arange(n), idx_0)
        weights[non_zero_idx] = stats.beta.pdf(MAF[non_zero_idx], weights_beta[0], weights_beta[1])

    return weights

