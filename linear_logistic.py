import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Import functions from ASTWAS_eigenval
from ASTWAS_eigenval import (
    get_liu_pval,
    get_liu_pval_mod, 
    get_davies_pval
)


def skat_linear_linear(res, Z, X1, kernel, weights, s2, method, res_out, n_resampling, r_corr, is_meta=False):
    """Run SKAT for linear kernel with linear outcome"""

    # Handle single SNP case for multiple r_corr
    r_corr_array = np.atleast_1d(r_corr)
    if len(r_corr_array) > 1 and Z.shape[1] == 1:
        r_corr = 0
        r_corr_array = np.array([0])

    # If Meta analysis is requested, use Meta SKAT implementation
    if is_meta:
        warnings.warn("Meta SKAT analysis not fully implemented yet, using standard SKAT")

    # Handle multiple r_corr values (SKAT-O)
    if len(r_corr_array) > 1:
        return skat_optimal_linear(res, Z, X1, kernel, weights, s2, method, res_out, n_resampling, r_corr_array)

    # Single r_corr value - use KMTest equivalent
    return kmtest_linear_linear(res, Z, X1, kernel, weights, s2, method, res_out, n_resampling, r_corr)


def kmtest_linear_linear(res, Z, X1, kernel, weights, s2, method, res_out, n_resampling, r_corr):
    """Linear kernel test for linear outcomes"""

    # Convert inputs to proper numpy arrays
    res = np.asarray(res).flatten()
    Z = np.asarray(Z)
    X1 = np.asarray(X1)

    if weights is not None:
        weights = np.asarray(weights).flatten()

    # Apply weights for weighted linear kernel
    if kernel == "linear.weighted" and weights is not None:
        Z = Z * weights[np.newaxis, :]

    # Apply correlation structure if r_corr is not 0
    if r_corr == 1:
        Z = np.sum(Z, axis=1, keepdims=True)
    elif r_corr > 0:
        p_m = Z.shape[1]
        R_M = np.diag(np.ones(p_m) * (1 - r_corr)) + np.ones((p_m, p_m)) * r_corr
        L = np.linalg.cholesky(R_M)
        Z = Z @ L.T

    # Calculate Q statistic
    Q_Temp = res.T @ Z
    Q = Q_Temp @ Q_Temp.T / s2 / 2

    # Handle resampling
    Q_res = None
    if n_resampling > 0 and res_out is not None:
        res_out = np.asarray(res_out)
        Q_Temp_res = res_out.T @ Z
        Q_res = np.sum(Q_Temp_res ** 2, axis=1) / s2 / 2

    # Calculate W_1 with numerical stability
    W1_part1 = Z.T @ Z

    # Compute projection matrix components
    ZT_X1 = Z.T @ X1
    XTX = X1.T @ X1
    XT_Z = X1.T @ Z

    # Use numerically stable solve method
    try:
        XTX_inv_XT_Z = np.linalg.solve(XTX, XT_Z)
    except np.linalg.LinAlgError:
        XTX_inv_XT_Z = np.linalg.lstsq(XTX, XT_Z, rcond=None)[0]

    W1_part2 = ZT_X1 @ XTX_inv_XT_Z
    W_1 = W1_part1 - W1_part2

    # Calculate p-value using appropriate method
    if method == "liu":
        out = get_liu_pval(Q, W_1, Q_res)
        pval_zero_msg = None
    elif method == "liu.mod":
        out = get_liu_pval_mod(Q, W_1, Q_res)
        pval_zero_msg = None
    elif method == "davies":
        out = get_davies_pval(Q, W_1, Q_res)
        pval_zero_msg = out.get("pval.zero.msg")
    else:
        raise ValueError("Invalid Method!")

    return {
        "p.value": out["p.value"],
        "p.value.resampling": out["p.value.resampling"],
        "Test.Type": method,
        "Q": Q,
        "param": out["param"],
        "pval.zero.msg": pval_zero_msg
    }


def skat_optimal_linear(res, Z, X1, kernel, weights, s2, method, res_out, n_resampling, r_all):
    """SKAT-O for linear outcomes"""
    warnings.warn("SKAT-O (multiple r_corr) not fully implemented, using first r_corr value")
    return kmtest_linear_linear(res, Z, X1, kernel, weights, s2, method, res_out, n_resampling, r_all[0])


def skat_linear_other(res, Z, X1, kernel, weights, s2, method, res_out, n_resampling):
    """SKAT for non-linear kernel with linear outcome"""

    # Convert inputs to proper numpy arrays
    res = np.asarray(res).flatten()
    Z = np.asarray(Z)
    X1 = np.asarray(X1)

    n = Z.shape[0]
    m = Z.shape[1]

    # Generate kernel matrix
    if isinstance(kernel, np.ndarray):
        K = kernel
    else:
        K = Z @ Z.T  # Linear kernel as default

    # Calculate Q statistic
    # R: Q = t(res)%*%K%*%res/(2*s2)
    Q = res.T @ K @ res / (2 * s2)

    # Handle resampling
    Q_res = None
    if n_resampling > 0 and res_out is not None:
        # R: Q.res<-rep(0,n.Resampling); for(i in 1:n.Resampling){ Q.res[i] = t(res.out[,i])%*%K%*%res.out[,i]/(2*s2) }
        res_out = np.asarray(res_out)
        Q_res = np.zeros(n_resampling)
        for i in range(n_resampling):
            Q_res[i] = res_out[:, i].T @ K @ res_out[:, i] / (2 * s2)

    # Calculate W
    # R: W = K - X1%*%solve( t(X1)%*%X1)%*%( t(X1) %*% K)
    W = K - X1 @ np.linalg.solve(X1.T @ X1, X1.T @ K)

    if method == "davies":
        # R: W1 = W - (W %*% X1) %*%solve( t(X1)%*%X1)%*% t(X1)
        W1 = W - (W @ X1) @ np.linalg.solve(X1.T @ X1, X1.T)
    else:
        W1 = W  # For liu and liu.mod methods

    # Calculate p-value using appropriate method
    if method == "liu":
        out = get_liu_pval(Q, W, Q_res)
        pval_zero_msg = None
    elif method == "liu.mod":
        out = get_liu_pval_mod(Q, W, Q_res)
        pval_zero_msg = None
    elif method == "davies":
        out = get_davies_pval(Q, W1, Q_res)
        pval_zero_msg = out.get("pval.zero.msg")
    else:
        raise ValueError("Invalid Method!")

    return {
        "p.value": out["p.value"],
        "p.value.resampling": out["p.value.resampling"],
        "Test.Type": method,
        "Q": Q,
        "param": out["param"],
        "pval.zero.msg": pval_zero_msg
    }


def skat_logistic_linear(res, Z, X1, kernel, weights, pi_1, method, res_out, n_resampling, r_corr, is_meta=False):
    """Run SKAT for linear kernel with logistic outcome - Python implementation of R's SKAT.logistic.Linear"""

    # Handle single SNP case for multiple r_corr
    r_corr_array = np.atleast_1d(r_corr)
    if len(r_corr_array) > 1 and Z.shape[1] == 1:
        r_corr = 0
        r_corr_array = np.array([0])

    # If Meta analysis is requested, use Meta SKAT implementation
    if is_meta:
        # Note: This would call SKAT_RunFrom_MetaSKAT in R
        warnings.warn("Meta SKAT analysis not fully implemented yet, using standard SKAT")

    # Handle multiple r_corr values (SKAT-O)
    if len(r_corr_array) > 1:
        return skat_optimal_logistic(res, Z, X1, kernel, weights, pi_1, method, res_out, n_resampling, r_corr_array)

    # Single r_corr value - use KMTest equivalent
    return kmtest_logistic_linear(res, Z, X1, kernel, weights, pi_1, method, res_out, n_resampling, r_corr)


def kmtest_logistic_linear(res, Z, X1, kernel, weights, pi_1, method, res_out, n_resampling, r_corr):
    """Linear kernel test for logistic outcomes"""

    # Convert inputs to proper numpy arrays - BUT preserve exact R behavior
    res = np.asarray(res, dtype=float)
    Z = np.asarray(Z, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    pi_1 = np.asarray(pi_1, dtype=float)

    # Ensure res is 1D (like R vector)
    if res.ndim > 1:
        res = res.flatten()

    # Ensure pi_1 is 1D (like R vector)
    if pi_1.ndim > 1:
        pi_1 = pi_1.flatten()

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim > 1:
            weights = weights.flatten()

    # Apply weights for weighted linear kernel
    # R: Z = t(t(Z) * (weights))
    if kernel == "linear.weighted" and weights is not None:
        # In R: t(t(Z) * weights) means apply weights to each column of Z
        Z = Z * weights[np.newaxis, :]  # Broadcast weights across rows

    # Apply correlation structure if r_corr is not 0
    if r_corr == 1:
        # R: Z<-cbind(rowSums(Z))
        Z = np.sum(Z, axis=1, keepdims=True)
    elif r_corr > 0:
        # R: p.m<-dim(Z)[2]; R.M<-diag(rep(1-r.corr,p.m)) + matrix(rep(r.corr,p.m*p.m),ncol=p.m)
        p_m = Z.shape[1]
        R_M = np.diag(np.ones(p_m) * (1 - r_corr)) + np.ones((p_m, p_m)) * r_corr

        # R: L<-chol(R.M,pivot=TRUE); Z<- Z %*% t(L)
        # 修正：使用R兼容的Cholesky分解方式
        try:
            # 首先尝试标准Cholesky分解
            L = np.linalg.cholesky(R_M)
        except np.linalg.LinAlgError:
            # 如果失败，使用SVD分解来模拟pivoted Cholesky
            U, s, Vt = np.linalg.svd(R_M)
            L = U @ np.diag(np.sqrt(np.maximum(s, 1e-12)))

        Z = Z @ L.T

    # Calculate Q statistic
    # R: Q.Temp = t(res)%*%Z; Q = Q.Temp %*% t(Q.Temp)/2
    if res.ndim == 1:
        Q_Temp = res @ Z  # For 1D res, this gives the correct row vector
    else:
        Q_Temp = res.T @ Z

    # Ensure Q_Temp is treated as row vector for matrix multiplication
    if Q_Temp.ndim == 1:
        Q = np.sum(Q_Temp ** 2) / 2
    else:
        Q = (Q_Temp @ Q_Temp.T).item() / 2  # Extract scalar

    # Handle resampling - 修正：严格按照R的逻辑
    Q_res = None
    if n_resampling > 0 and res_out is not None:
        # R: Q.Temp.res = t(res.out)%*%Z; Q.res = rowSums(rbind(Q.Temp.res^2))/2
        res_out = np.asarray(res_out, dtype=float)

        # t(res.out) %*% Z
        Q_Temp_res = res_out.T @ Z  # This should be (n_resampling, n_snps)

        # rowSums(rbind(Q.Temp.res^2)) - R的rbind在这里是identity operation
        # 所以这等于rowSums(Q.Temp.res^2)
        Q_res = np.sum(Q_Temp_res ** 2, axis=1) / 2  # Sum across SNPs for each resampling

    # Calculate W_1 - 修正：完全按照R版本的逻辑实现，改进数值稳定性
    # R: W.1 = t(Z) %*% (Z * pi_1) - (t(Z * pi_1) %*%X1)%*%solve(t(X1)%*%(X1 * pi_1))%*% (t(X1) %*% (Z * pi_1))

    # Step 1: (Z * pi_1) - 在R中，这表示每行乘以对应的pi_1元素
    Z_times_pi1 = Z * pi_1[:, np.newaxis]  # Broadcasting pi_1 across columns

    # Step 2: t(Z) %*% (Z * pi_1) - 第一部分
    W1_part1 = Z.T @ Z_times_pi1

    # Step 3: 计算第二部分 - 改进数值稳定性
    # R: (t(Z * pi_1) %*%X1)%*%solve(t(X1)%*%(X1 * pi_1))%*% (t(X1) %*% (Z * pi_1))

    # 3.1: X1 * pi_1 (每行乘以对应的pi_1)
    X1_times_pi1 = X1 * pi_1[:, np.newaxis]

    # 3.2: t(X1)%*%(X1 * pi_1) - 计算系数矩阵
    XTX_pi1 = X1.T @ X1_times_pi1

    # 3.3: t(Z * pi_1) %*%X1
    ZT_pi1_X1 = Z_times_pi1.T @ X1

    # 3.4: t(X1) %*% (Z * pi_1)
    XT_Z_pi1 = X1.T @ Z_times_pi1

    # 3.5: 使用数值稳定的solve方法，而不是直接求逆
    # R中的solve(A, B)等价于Python中的np.linalg.solve(A, B)
    # 这里需要计算 solve(XTX_pi1, XT_Z_pi1)
    try:
        # 使用LU分解求解，比直接求逆更稳定
        XTX_pi1_inv_XT_Z_pi1 = np.linalg.solve(XTX_pi1, XT_Z_pi1)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用最小二乘法求解
        XTX_pi1_inv_XT_Z_pi1 = np.linalg.lstsq(XTX_pi1, XT_Z_pi1, rcond=None)[0]

    # 3.6: 计算第二部分
    # (t(Z * pi_1) %*%X1) %*% solve_result
    W1_part2 = ZT_pi1_X1 @ XTX_pi1_inv_XT_Z_pi1

    # Step 4: 最终的W.1
    W_1 = W1_part1 - W1_part2

    # Calculate p-value using appropriate method
    if method == "liu":
        out = get_liu_pval(Q, W_1, Q_res)
        pval_zero_msg = None
    elif method == "liu.mod":
        out = get_liu_pval_mod(Q, W_1, Q_res)
        pval_zero_msg = None
    elif method == "davies":
        out = get_davies_pval(Q, W_1, Q_res)
        pval_zero_msg = out.get("pval.zero.msg")
    else:
        raise ValueError("Invalid Method!")

    return {
        "p.value": out["p.value"],
        "p.value.resampling": out["p.value.resampling"],
        "Test.Type": method,
        "Q": Q,
        "Q.resampling": Q_res,
        "param": out["param"],
        "pval.zero.msg": pval_zero_msg
    }


def skat_optimal_logistic(res, Z, X1, kernel, weights, pi_1, method, res_out, n_resampling, r_all):
    """SKAT-O for logistic outcomes"""
    warnings.warn("SKAT-O (multiple r_corr) not fully implemented, using first r_corr value")
    return kmtest_logistic_linear(res, Z, X1, kernel, weights, pi_1, method, res_out, n_resampling, r_all[0])


def skat_logistic_other(res, Z, X1, kernel, weights, pi_1, method, res_out, n_resampling):
    """Run SKAT for non-linear kernel with logistic outcome - Python implementation of R's SKAT.logistic.Other"""

    # Convert inputs to proper numpy arrays
    res = np.asarray(res).flatten()  # Ensure res is 1D array
    Z = np.asarray(Z)
    X1 = np.asarray(X1)
    pi_1 = np.asarray(pi_1).flatten()  # Ensure pi_1 is 1D array

    n = Z.shape[0]
    m = Z.shape[1]

    # Generate kernel matrix
    if isinstance(kernel, np.ndarray):
        K = kernel
    else:
        # Implement kernel calculation for various kernel types
        # This is a simplified version
        K = Z @ Z.T  # Linear kernel as default

    # Calculate Q statistic
    # R: Q = t(res)%*%K%*%res/2
    Q = res.T @ K @ res / 2

    # Handle resampling
    Q_res = None
    if n_resampling > 0 and res_out is not None:
        # R: Q.res<-rep(0,n.Resampling); for(i in 1:n.Resampling){ Q.res[i] = t(res.out[,i])%*%K%*%res.out[,i]/2 }
        res_out = np.asarray(res_out)
        Q_res = np.zeros(n_resampling)
        for i in range(n_resampling):
            Q_res[i] = res_out[:, i].T @ K @ res_out[:, i] / 2

    # Calculate matrices for test statistic
    # R: D = diag(pi_1)
    # R: gg = X1%*%solve(t(X1)%*%(X1 * pi_1))%*%t(X1 * pi_1)
    # R: P0 = D-(gg * pi_1)

    # In Python, we can compute this more efficiently
    # X1 * pi_1 means multiply each row of X1 by corresponding element of pi_1
    X1_weighted = X1 * pi_1[:, np.newaxis]  # Broadcast pi_1 to match X1 shape
    XTX_pi = X1.T @ X1_weighted
    XTX_pi_inv = np.linalg.solve(XTX_pi, X1.T @ X1_weighted.T)

    if method == "davies":
        # R: P0_half = Get_Matrix_Square.1(P0); W1 = P0_half %*% K %*% t(P0_half)
        # For Davies method, we need the square root of P0
        D = np.diag(pi_1)
        gg = X1 @ XTX_pi_inv
        P0 = D - (gg * pi_1[:, np.newaxis])

        # Get square root of P0 (simplified - in R this uses Get_Matrix_Square.1)
        eigenvals, eigenvecs = np.linalg.eigh(P0)
        eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
        P0_half = eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T

        W = P0_half @ K @ P0_half.T
    else:
        # R: W = K * pi_1 - (X1 *pi_1) %*%solve(t(X1)%*%(X1 * pi_1))%*% ( t(X1 * pi_1) %*% K)
        # First part: K * pi_1 (multiply each row of K by corresponding element of pi_1)
        K_weighted = K * pi_1[:, np.newaxis]

        # Second part: (X1 *pi_1) %*%solve(t(X1)%*%(X1 * pi_1))%*% ( t(X1 * pi_1) %*% K)
        XTK_pi = X1_weighted.T @ K
        second_part = X1_weighted @ np.linalg.solve(XTX_pi, XTK_pi)

        W = K_weighted - second_part

    # Calculate p-value using appropriate method
    if method == "liu":
        out = get_liu_pval(Q, W, Q_res)
        pval_zero_msg = None
    elif method == "liu.mod":
        out = get_liu_pval_mod(Q, W, Q_res)
        pval_zero_msg = None
    elif method == "davies":
        out = get_davies_pval(Q, W, Q_res)
        pval_zero_msg = out.get("pval.zero.msg")
    else:
        raise ValueError("Invalid Method!")

    return {
        "p.value": out["p.value"],
        "p.value.resampling": out["p.value.resampling"],
        "Test.Type": method,
        "Q": Q,
        "param": out["param"],
        "pval.zero.msg": pval_zero_msg
    }

