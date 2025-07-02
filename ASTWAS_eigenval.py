import numpy as np
import warnings

# Import functions from other modules
from ASTWAS_utils import (
    _import_scipy_stats,
    get_liu_params,
    get_liu_params_mod,
    get_liu_params_mod_lambda,
    get_liu_pval_mod_lambda_zero
)
from davies_method import davies_python, davies_improved


def get_lambda(K, maxK=100):
    """
    Get lambda values from kernel matrix
    
    Parameters:
    ----------
    K : array-like
        Kernel matrix
    maxK : int, default=100
        Maximum number of eigenvalues to consider
    
    Returns:
    -------
    dict
        Dictionary with lambda values and degrees of freedom
    """
    try:
        if K is None or K.size == 0:
            raise ValueError("Kernel matrix K is empty or None")

        if not isinstance(K, np.ndarray):
            K = np.asarray(K)

        if K.shape[0] != K.shape[1]:
            raise ValueError("Kernel matrix K must be square")

        if not np.isfinite(K).all():
            raise ValueError("Kernel matrix K contains non-finite values")

        try:
            eigen_vals = np.linalg.eigvalsh(K)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to compute eigenvalues: {str(e)}")

        lambda_sum = np.trace(K)
        lambda2_sum = np.sum(K * K)

        p1 = K.shape[0]
        k1 = min(int(p1 / 10), maxK)
        k1 = max(k1, 1)

        if p1 <= 100:
            lambda1 = eigen_vals
            lambda1 = np.sort(lambda1)[::-1]
            lambda1 = lambda1[:k1]
        else:
            lambda1 = np.sort(eigen_vals)[::-1][:k1]

        min_positive_threshold = max(1e-12, np.max(np.abs(lambda1)) * 1e-15)
        IDX1 = np.where(lambda1 >= min_positive_threshold)[0]

        if len(IDX1) > 0:
            mean_positive = np.mean(lambda1[IDX1])
            if mean_positive > 0:
                threshold = mean_positive / 100000
            else:
                threshold = min_positive_threshold

            IDX2 = np.where(lambda1 > threshold)[0]

            if len(IDX2) == 0:
                if len(IDX1) > 0:
                    IDX2 = IDX1[:1]
                else:
                    raise ValueError("No positive eigenvalues found!")

            lambda1 = lambda1[IDX2]

            lambda_sum = lambda_sum - np.sum(lambda1)
            lambda2_sum = lambda2_sum - np.sum(lambda1 ** 2)

            if lambda2_sum > 1e-15:
                with np.errstate(divide='ignore', invalid='ignore'):
                    df1_float = lambda_sum ** 2 / lambda2_sum

                if not np.isfinite(df1_float) or df1_float <= 0:
                    lambda_vals = np.round(lambda1, decimals=10)
                    return {"lambda": lambda_vals, "df1": np.ones(len(lambda1))}

                df1 = max(1, int(round(df1_float)))

                if abs(lambda_sum) > 1e-15:
                    lambda_last = lambda_sum / df1

                    if not np.isfinite(lambda_last):
                        lambda_last = 0
                else:
                    lambda_last = 0

                lambda_vals = lambda1
                df1_vals = np.append(np.ones(len(lambda1)), df1)

                lambda_vals = np.round(lambda_vals, decimals=10)

                return {"lambda": lambda_vals, "df1": df1_vals}
            else:
                lambda1 = np.round(lambda1, decimals=10)
                return {"lambda": lambda1, "df1": np.ones(len(lambda1))}
        else:
            raise ValueError("No positive eigenvalues found!")

    except Exception as e:
        error_msg = f"Error in get_lambda: {str(e)}"
        if hasattr(K, 'shape'):
            error_msg += f" (K shape: {K.shape})"
        print(error_msg)
        return None


def get_lambda_improved(K):
    """
    Get lambda values from kernel matrix - improved version with better precision
    
    Parameters:
    ----------
    K : array-like
        Kernel matrix
    
    Returns:
    -------
    array-like
        Significant eigenvalues
    """
    try:
        K = (K + K.T) / 2

        if K.shape[0] <= 100:
            eigenvals, eigenvecs = np.linalg.eigh(K)
        else:
            eigenvals = np.linalg.eigvalsh(K)

        eigenvals = np.sort(eigenvals)[::-1]

        positive_mask = eigenvals > 0
        if not np.any(positive_mask):
            raise ValueError("No positive eigenvalues found!")

        positive_eigenvals = eigenvals[positive_mask]

        max_eigenval = np.max(positive_eigenvals)

        relative_threshold = np.mean(positive_eigenvals) / 100000.0

        absolute_threshold = max_eigenval * 1e-12

        threshold = max(relative_threshold, absolute_threshold)

        significant_mask = eigenvals > threshold
        significant_eigenvals = eigenvals[significant_mask]

        if len(significant_eigenvals) == 0:
            return np.array([max_eigenval])

        return significant_eigenvals

    except Exception as e:
        print(f"Error in get_lambda_improved: {str(e)}")
        print(f"K matrix shape: {K.shape}")
        print(f"K matrix condition number: {np.linalg.cond(K)}")
        print(f"K matrix rank: {np.linalg.matrix_rank(K)}")
        return None


def get_pvalue(K, Q, is_fast=False, fast_cutoff=2000):
    """
    Calculate p-value from kernel matrix and Q statistic
    
    Parameters:
    ----------
    K : array-like
        Kernel matrix
    Q : float or array-like
        Test statistic
    is_fast : bool, default=False
        Whether to use fast approximation
    fast_cutoff : int, default=2000
        Cutoff for fast approximation
    
    Returns:
    -------
    dict
        Dictionary with p-values and convergence information
    """
    lambda_result = get_lambda(K, maxK=fast_cutoff if is_fast else K.shape[0])
    np.set_printoptions(precision=10, suppress=True)

    if lambda_result is None:
        warnings.warn("Eigenvalue calculation failed, using Liu method")
        W_fallback = K * 2
        return {"p.value": [get_liu_pval_mod(Q, W_fallback)["p.value"]],
                "p.val.liu": [get_liu_pval_mod(Q, W_fallback)["p.value"]],
                "is_converge": [0]}

    lambda_vals = lambda_result["lambda"]

    return get_pvalue_lambda(lambda_vals, Q, is_fast=is_fast)


def get_pvalue_improved(K, Q, is_fast=False, fast_cutoff=2000):
    """Calculate p-value from kernel matrix and Q statistic - improved version"""
    lambda_vals = get_lambda_improved(K)
    if lambda_vals is None:
        warnings.warn("Eigenvalue calculation failed, using Liu method")
        W_fallback = K * 2
        return {"p.value": [get_liu_pval_mod(Q, W_fallback)["p.value"]],
                "p.val.liu": [get_liu_pval_mod(Q, W_fallback)["p.value"]],
                "is_converge": [0]}

    return get_pvalue_lambda_improved(lambda_vals, Q, is_fast=is_fast)


def get_pvalue_lambda(lambda_vals, Q, is_fast=False):
    """
    Calculate p-value using lambda values
    
    Parameters:
    ----------
    lambda_vals : array-like
        Eigenvalues
    Q : float or array-like
        Test statistic
    is_fast : bool, default=False
        Whether to use fast approximation
    
    Returns:
    -------
    dict
        Dictionary with p-values and convergence information
    """
    from scipy import stats
    import warnings

    Q = np.atleast_1d(Q)
    n1 = len(Q)

    p_val = np.zeros(n1)
    p_val_liu = np.zeros(n1)
    is_converge = np.zeros(n1)

    df1 = None
    lambda_org = lambda_vals.copy()
    lambda_n = len(lambda_vals)
    nMAX = 100

    if is_fast and lambda_n > nMAX:
        df1 = np.ones(nMAX + 1)
        df1[nMAX + 1] = lambda_n - nMAX
        lambda_vals = lambda_vals[:nMAX + 1].copy()
        lambda_vals[nMAX + 1] = np.mean(lambda_org[nMAX:])

    p_val_liu = get_liu_pval_mod_lambda(Q, lambda_vals, df1)

    for i in range(n1):
        try:
            if df1 is None:
                out = davies_python(Q[i], lambda_vals, acc=1e-6)
            else:
                out = davies_python(Q[i], lambda_vals, h=df1, acc=1e-6)

            p_val[i] = out["Qq"]
            is_converge[i] = 1

            if len(lambda_vals) == 1:
                p_val[i] = p_val_liu[i]
            elif out["ifault"] != 0:
                is_converge[i] = 0

            if p_val[i] > 1 or p_val[i] <= 0:
                is_converge[i] = 0
                p_val[i] = p_val_liu[i]

        except Exception as e:
            warnings.warn(f"Davies method failed for Q[{i}], using Liu method: {str(e)}")
            p_val[i] = p_val_liu[i]
            is_converge[i] = 0

    p_val_msg = None
    p_val_log = None

    if p_val[0] == 0:
        try:
            param = get_liu_params_mod_lambda(lambda_vals, df1)
            p_val_msg = get_liu_pval_mod_lambda_zero(Q[0], param["muQ"], param["muX"],
                                                     param["sigmaQ"], param["sigmaX"],
                                                     param["l"], param["d"])
            p_val_log = get_liu_pval_mod_lambda(Q[0:1], lambda_vals, df1, log_p=True)[0]
        except Exception:
            pass

    return {
        "p.value": p_val,
        "p.val.liu": p_val_liu,
        "is_converge": is_converge,
        "p.val.log": p_val_log,
        "pval.zero.msg": p_val_msg
    }


def get_pvalue_lambda_improved(lambda_vals, Q, is_fast=False):
    """Calculate p-value using lambda values - improved version"""
    from scipy import stats
    import warnings

    Q = np.atleast_1d(Q)
    n1 = len(Q)

    p_val = np.zeros(n1)
    p_val_liu = np.zeros(n1)
    is_converge = np.zeros(n1)

    df1 = None
    lambda_org = lambda_vals.copy()
    lambda_n = len(lambda_vals)
    nMAX = 100

    if is_fast and lambda_n > nMAX:
        df1 = np.ones(nMAX + 1)
        df1[nMAX] = lambda_n - nMAX
        lambda_vals = lambda_vals[:nMAX + 1].copy()
        lambda_vals[nMAX] = np.mean(lambda_org[nMAX:])

    p_val_liu = get_liu_pval_mod_lambda(Q, lambda_vals, df1)

    for i in range(n1):
        try:
            if df1 is None:
                out = davies_improved(Q[i], lambda_vals, acc=1e-8)
            else:
                out = davies_improved(Q[i], lambda_vals, h=df1, acc=1e-8)

            p_val[i] = out["Qq"]
            is_converge[i] = 1

            if len(lambda_vals) == 1:
                p_val[i] = p_val_liu[i]
            elif out["ifault"] != 0:
                is_converge[i] = 0

            if p_val[i] > 1 or p_val[i] < 0 or np.isnan(p_val[i]) or np.isinf(p_val[i]):
                is_converge[i] = 0
                p_val[i] = p_val_liu[i]

        except Exception as e:
            warnings.warn(f"Davies method failed for Q[{i}], using Liu method: {str(e)}")
            p_val[i] = p_val_liu[i]
            is_converge[i] = 0

    p_val_msg = None
    p_val_log = None

    if p_val[0] == 0:
        try:
            param = get_liu_params_mod_lambda(lambda_vals, df1)
            p_val_msg = get_liu_pval_mod_lambda_zero(Q[0], param["muQ"], param["muX"],
                                                     param["sigmaQ"], param["sigmaX"],
                                                     param["l"], param["d"])
            p_val_log = get_liu_pval_mod_lambda(Q[0:1], lambda_vals, df1, log_p=True)[0]
        except Exception:
            pass

    return {
        "p.value": p_val,
        "p.val.liu": p_val_liu,
        "is_converge": is_converge,
        "p.val.log": p_val_log,
        "pval.zero.msg": p_val_msg
    }


def get_liu_pval(Q, W, Q_resampling=None):
    """Get Liu p-value (original version)"""
    Q = np.asmatrix(Q)
    W = np.asmatrix(W)

    if Q_resampling is not None and len(Q_resampling) > 0:
        Q_resampling = np.asmatrix(Q_resampling)

    Q_all = np.concatenate([Q, Q_resampling]) if Q_resampling is not None and len(Q_resampling) > 0 else Q

    A1 = W / 2
    A2 = A1 @ A1

    c1 = np.zeros(4)
    c1[0] = np.sum(np.diag(A1))
    c1[1] = np.sum(np.diag(A2))
    c1[2] = np.sum(A1 * A2.T)
    c1[3] = np.sum(A2 * A2.T)

    param = get_liu_params(c1)

    Q_Norm = (Q_all - param["muQ"]) / param["sigmaQ"]
    Q_Norm1 = Q_Norm * param["sigmaX"] + param["muX"]

    stats = _import_scipy_stats()
    p_value = stats.ncx2.sf(Q_Norm1, df=param["l"], nc=param["d"])

    p_value_resampling = None
    if Q_resampling is not None and len(Q_resampling) > 0:
        p_value_resampling = p_value[1:]

    return {"p.value": p_value[0], "param": param, "p.value.resampling": p_value_resampling}


def get_liu_pval_mod(Q, W, Q_resampling=None):
    """Get Liu p-value (modified version)"""
    Q = np.asmatrix(Q)
    W = np.asmatrix(W)

    if Q_resampling is not None and len(Q_resampling) > 0:
        Q_resampling = np.asmatrix(Q_resampling)

    Q_all = np.concatenate([Q, Q_resampling]) if Q_resampling is not None and len(Q_resampling) > 0 else Q

    A1 = W / 2
    A2 = A1 @ A1

    c1 = np.zeros(4)
    c1[0] = np.sum(np.diag(A1))
    c1[1] = np.sum(np.diag(A2))
    c1[2] = np.sum(A1 * A2.T)
    c1[3] = np.sum(A2 * A2.T)

    param = get_liu_params_mod(c1)

    Q_Norm = (Q_all - param["muQ"]) / param["sigmaQ"]
    Q_Norm1 = Q_Norm * param["sigmaX"] + param["muX"]

    stats = _import_scipy_stats()
    p_value = stats.ncx2.sf(Q_Norm1, df=param["l"], nc=param["d"])

    p_value_resampling = None
    if Q_resampling is not None and len(Q_resampling) > 0:
        p_value_resampling = p_value[1:]

    return {"p.value": p_value[0], "param": param, "p.value.resampling": p_value_resampling}


def get_liu_pval_mod_lambda(Q_all, lambda_vals, df1=None, log_p=False):
    """
    Calculate Liu's p-value with lambda values
    
    Parameters:
    ----------
    Q_all : array-like
        Test statistics
    lambda_vals : array-like
        Eigenvalues
    df1 : array-like, optional
        Degrees of freedom
    log_p : bool, default=False
        Whether to return log p-values
    
    Returns:
    -------
    array-like
        P-values
    """
    from scipy import stats

    param = get_liu_params_mod_lambda(lambda_vals, df1)

    Q_Norm = (Q_all - param["muQ"]) / param["sigmaQ"]
    Q_Norm1 = Q_Norm * param["sigmaX"] + param["muX"]

    if log_p:
        p_value = stats.ncx2.logsf(Q_Norm1, df=param["l"], nc=param["d"])
    else:
        p_value = stats.ncx2.sf(Q_Norm1, df=param["l"], nc=param["d"])

    return p_value


def get_davies_pval(Q, W, Q_resampling=None, is_fast=False, fast_cutoff=2000):
    """
    Get Davies' p-value
    
    Parameters:
    ----------
    Q : array-like
        Test statistic
    W : array-like
        Weight matrix
    Q_resampling : array-like, optional
        Resampling test statistics
    is_fast : bool, default=False
        Whether to use fast approximation
    fast_cutoff : int, default=2000
        Cutoff for fast approximation
    
    Returns:
    -------
    dict
        Dictionary with p-values and parameters
    """
    Q = np.asmatrix(Q)
    W = np.asmatrix(W)

    if Q_resampling is not None and len(Q_resampling) > 0:
        Q_resampling = np.asmatrix(Q_resampling)

    K = W / 2
    Q_all = np.concatenate([Q, Q_resampling]) if Q_resampling is not None and len(Q_resampling) > 0 else Q

    re = get_pvalue(K, Q_all, is_fast=is_fast, fast_cutoff=fast_cutoff)

    param = {}
    param["liu_pval"] = re["p.val.liu"][0]
    param["Is_Converged"] = re["is_converge"][0]

    p_value_resampling = None
    if Q_resampling is not None and len(Q_resampling) > 0:
        p_value_resampling = re["p.value"][1:]
        param["liu_pval.resampling"] = re["p.val.liu"][1:]
        param["Is_Converged.resampling"] = re["is_converge"][1:]

    return {
        "p.value": re["p.value"][0],
        "param": param,
        "p.value.resampling": p_value_resampling,
        "pval.zero.msg": re.get("pval.zero.msg")
    }


