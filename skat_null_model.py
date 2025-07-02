# SKAT Null Model
# Functions for creating and processing null models for SKAT analysis

import numpy as np
import pandas as pd

def _import_pandas():
    """Lazy import pandas to avoid startup overhead"""
    import pandas as pd
    return pd

# Cache imported modules to avoid repeated imports
_statsmodels_cache = None
_patsy_cache = None

def _import_statsmodels():
    """Lazy import statsmodels with caching"""
    global _statsmodels_cache
    if _statsmodels_cache is None:
        import statsmodels.api as sm
        _statsmodels_cache = sm
    return _statsmodels_cache

def _import_patsy():
    """Lazy import patsy with caching"""
    global _patsy_cache
    if _patsy_cache is None:
        from patsy.highlevel import dmatrices
        _patsy_cache = dmatrices
    return _patsy_cache

def Get_Resampling_Bin(ncase, prob, n_Resampling):
    """Simulate binomial resampling for binary outcomes"""
    n = len(prob)
    resampled = np.zeros((n, n_Resampling))
    normalized_prob = prob / np.sum(prob)

    for i in range(n_Resampling):
        # Use multinomial distribution approximation to ensure case count
        idx = np.random.choice(n, size=ncase, replace=True, p=normalized_prob)
        resampled[idx, i] = 1

    return resampled


def Get_SKAT_Residuals_Get_X1(X1):
    """Process design matrix X1, using SVD for rank deficiency"""
    q1 = X1.shape[1]
    
    # Compute matrix rank efficiently
    rank = np.linalg.matrix_rank(X1, tol=1e-8)

    if rank < q1:
        # Use economic SVD mode for efficiency
        U, S, Vt = np.linalg.svd(X1, full_matrices=False)
        X1 = U

    return X1


def Get_SKAT_Residuals_linear(formula, data, n_Resampling, type_Resampling, id_include):
    """Generate residuals and resampling for linear models"""
    dmatrices = _import_patsy()
    sm = _import_statsmodels()
    
    # Build model matrices
    y, X = dmatrices(formula, data, return_type='dataframe')
    
    # Use direct numpy linear regression for speed
    X_vals = X.values
    y_vals = y.values.flatten()
    
    # Solve least squares: beta = (X'X)^(-1)X'y
    XTX = X_vals.T @ X_vals
    XTy = X_vals.T @ y_vals
    
    # Use solve instead of inverse for numerical stability
    try:
        beta = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XTX, XTy, rcond=None)[0]
    
    # Calculate residuals and variance
    y_pred = X_vals @ beta
    res = y_vals - y_pred
    
    # Calculate residual variance
    n_obs = len(y_vals)
    n_params = X_vals.shape[1]
    if n_obs > n_params:
        s2 = np.sum(res ** 2) / (n_obs - n_params)
    else:
        s2 = 1.0
    
    X1 = Get_SKAT_Residuals_Get_X1(X_vals)
    res_out = None

    # Handle resampling if requested
    if n_Resampling > 0:
        if type_Resampling == "permutation":
            # Vectorized permutation
            resampled = np.tile(res, (n_Resampling, 1))
            for i in range(n_Resampling):
                np.random.shuffle(resampled[i])
            res_out = resampled.T

        elif type_Resampling == "bootstrap":
            # Pre-allocate arrays for efficiency
            res_out = np.random.normal(0, np.sqrt(s2), (len(res), n_Resampling))
            
            # Pre-compute inverse matrix to avoid repeated calculation
            try:
                XTX_inv = np.linalg.solve(XTX, np.eye(XTX.shape[0]))
            except np.linalg.LinAlgError:
                XTX_inv = np.linalg.pinv(XTX)
            
            # Vectorized adjustment calculation
            adjustment = X1 @ (XTX_inv @ (X1.T @ res_out))
            res_out -= adjustment

        elif type_Resampling == "perturbation":
            raise ValueError("Perturbation not supported for linear models")
        else:
            raise ValueError("Invalid resampling method")

    return {
        'res': res,
        'X1': X1,
        'res_out': res_out,
        'out_type': 'C',
        'n_Resampling': n_Resampling,
        'type_Resampling': type_Resampling,
        'id_include': id_include,
        's2': s2
    }


def Get_SKAT_Residuals_logistic(formula, data, n_Resampling, type_Resampling, id_include):
    """Generate residuals and resampling for logistic models"""
    dmatrices = _import_patsy()
    sm = _import_statsmodels()
    
    y, X = dmatrices(formula, data, return_type='dataframe')
    
    # Fit GLM with optimized settings for speed
    mod_glm = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        maxiter=25,
        tol=1e-6,
        scale=None
    )
    
    X1 = Get_SKAT_Residuals_Get_X1(X.values)
    mu = mod_glm.fittedvalues.values
    
    # Vectorized residual calculation
    res = y.values.flatten() - mu
    n1 = len(res)
    n_case = int(np.sum(y.values))
    pi_1 = mu * (1 - mu)
    res_out = None

    # Handle resampling if requested
    if n_Resampling > 0:
        if type_Resampling == "bootstrap.fast":
            try:
                resampled = Get_Resampling_Bin(n_case, mu, n_Resampling)
                res_out = resampled - mu[:, np.newaxis]
            except Exception:
                type_Resampling = "bootstrap"

        if type_Resampling == "bootstrap":
            # Pre-allocate arrays with float32 for memory efficiency
            res_out = np.zeros((n1, n_Resampling), dtype=np.float32)
            
            # Vectorized bootstrap sampling
            for i in range(n_Resampling):
                res_out1 = np.random.binomial(1, mu, n1)
                res_out2 = np.random.binomial(1, mu, n1)

                id_case1 = np.where(res_out1 == 1)[0]
                id_case2 = np.where(res_out2 == 1)[0]

                id_c1 = np.intersect1d(id_case1, id_case2)
                id_c2 = np.union1d(
                    np.setdiff1d(id_case1, id_case2),
                    np.setdiff1d(id_case2, id_case1)
                )

                if len(id_c1) >= n_case:
                    id_case = np.random.choice(id_c1, n_case, replace=False)
                elif len(id_c1) + len(id_c2) >= n_case:
                    needed = n_case - len(id_c1)
                    id_c3 = np.random.choice(id_c2, needed, replace=False)
                    id_case = np.concatenate([id_c1, id_c3])
                else:
                    id_case3 = np.concatenate([id_c1, id_c2])
                    id_c4 = np.setdiff1d(np.arange(n1), id_case3)
                    needed = n_case - len(id_case3)

                    if len(id_c4) > 0:
                        probs = mu[id_c4]
                        probs = probs / probs.sum()
                        id_c5 = np.random.choice(id_c4, needed, replace=False, p=probs)
                        id_case = np.concatenate([id_case3, id_c5])
                    else:
                        id_case = id_case3

                res_out[id_case, i] = 1

            # Vectorized subtraction
            res_out = res_out - mu[:, np.newaxis]

        elif type_Resampling == "permutation":
            # Vectorized permutation
            resampled = np.tile(res, (n_Resampling, 1))
            for i in range(n_Resampling):
                np.random.shuffle(resampled[i])
            res_out = resampled.T

        else:
            if res_out is None:
                raise ValueError("Invalid resampling method")

    return {
        'res': res,
        'X1': X1,
        'res_out': res_out,
        'out_type': 'D',
        'n_Resampling': n_Resampling,
        'type_Resampling': type_Resampling,
        'id_include': id_include,
        'mu': mu,
        'pi_1': pi_1
    }


def SKAT_Null_Model_Get_Includes(obj_omit, obj_pass):
    """Get indices of non-missing samples"""
    id_include = obj_omit.index.tolist()
    return sorted(id_include)


def SKAT_Null_Model(formula, data=None, out_type="C", n_Resampling=0,
                    type_Resampling="bootstrap", Adjustment=True):
    """
    Create null model for SKAT analysis
    
    Parameters:
    ----------
    formula : str
        R-style formula string
    data : DataFrame
        DataFrame containing variables in formula
    out_type : str, default="C"
        Outcome type - "C" for continuous, "D" for binary
    n_Resampling : int, default=0
        Number of resampling iterations
    type_Resampling : str, default="bootstrap"
        Resampling method ("bootstrap", "permutation")
    Adjustment : bool, default=True
        Whether to apply small sample adjustment
        
    Returns:
    -------
    dict
        Dictionary containing null model results
    """
    if data is None:
        raise ValueError("Data must be provided")

    dmatrices = _import_patsy()
    
    # Parse formula efficiently
    try:
        y, X = dmatrices(formula, data, return_type='dataframe')
    except Exception as e:
        raise ValueError(f"Error parsing formula: {e}")

    # Efficient missing value detection
    y_missing = y.isnull().any().any()
    X_missing = X.isnull().any().any()
    
    if y_missing or X_missing:
        combined = pd.concat([y, X], axis=1).dropna()
        id_include = sorted(combined.index.tolist())
    else:
        id_include = list(range(len(data)))

    # Small sample adjustment logic
    n1 = len(id_include)
    n = len(data)
    
    # Apply small sample adjustment only when necessary
    if n1 < 2000 and out_type == "D" and Adjustment:
        print(f"Sample size = {n1} (<2000). Applying small sample adjustment!")
        n_kurtosis = 10000
        return SKAT_Null_Model_MomentAdjust(
            formula, data, n_Resampling, type_Resampling, True, n_kurtosis
        )

    # Warn about missing values
    if n - n1 > 0:
        print(f"{n - n1} samples excluded due to missing data!")

    # Select appropriate model type
    if out_type == "C":
        result = Get_SKAT_Residuals_linear(
            formula, data, n_Resampling, type_Resampling, id_include
        )
    elif out_type == "D":
        result = Get_SKAT_Residuals_logistic(
            formula, data, n_Resampling, type_Resampling, id_include
        )
    else:
        raise ValueError("Invalid out_type")

    result['n_all'] = n
    return result


def SKAT_Null_Model_MomentAdjust(formula, data=None, n_Resampling=0,
                                 type_Resampling="bootstrap",
                                 is_kurtosis_adj=True,
                                 n_Resampling_kurtosis=10000):
    """Moment-adjusted null model for binary outcomes with small samples"""
    # Main result
    re1 = Get_SKAT_Residuals_logistic(
        formula, data, n_Resampling, type_Resampling, []
    )

    # Additional result for kurtosis adjustment
    re2 = None
    if is_kurtosis_adj:
        re2 = Get_SKAT_Residuals_logistic(
            formula, data, n_Resampling_kurtosis, type_Resampling, []
        )

    return {
        're1': re1,
        're2': re2,
        'is_kurtosis_adj': is_kurtosis_adj,
        'type': 'binary'
    }