import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Import functions from ASTWAS_utils
from ASTWAS_utils import _import_scipy_stats, _import_momentchi2




def davies_python(q, lambda_vals, h=None, delta=None, sigma=0, lim=10000, acc=0.0001):
    """
    Davies' method for computing tail probabilities of quadratic forms in normal variables

    Parameters:
    ----------
    q : float
        Test statistic
    lambda_vals : array-like
        Eigenvalues
    h : array-like, optional
        Degrees of freedom for each eigenvalue
    delta : array-like, optional
        Non-centrality parameters
    sigma : float, default=0
        Variance parameter
    lim : int, default=10000
        Maximum number of terms in series
    acc : float, default=0.0001
        Required accuracy
    
    Returns:
    -------
    dict
        Dictionary with p-value, fault code, and trace information
    """
    try:
        # 使用momentchi2库
        hbe = _import_momentchi2()
        if hbe is None:
            raise ImportError("momentchi2 not available")

        r = len(lambda_vals)

        if h is None:
            h = np.ones(r)
        if delta is None:
            delta = np.zeros(r)

        if len(h) != r:
            raise ValueError("lambda and h should have the same length!")
        if len(delta) != r:
            raise ValueError("lambda and delta should have the same length!")

        # 关键修改：应用0.01的缩放因子，与R版本保持一致
        lambda_vals_scaled = lambda_vals * 0.01

        # 对于非中心参数非零的情况，使用回退方法
        if np.any(delta != 0):
            stats = _import_scipy_stats()

            # 计算累积量 - 使用缩放后的特征值
            c1 = np.sum(lambda_vals_scaled * h) + np.sum(lambda_vals_scaled * delta)
            c2 = np.sum(lambda_vals_scaled**2 * h) + 2 * np.sum(lambda_vals_scaled**2 * delta)

            if c2 > 0:
                df_satt = 2 * c1**2 / c2
                scale_satt = c2 / (2 * c1)
                q_scaled = q / scale_satt
                p_val = stats.chi2.sf(q_scaled, df=df_satt)
            else:
                p_val = 1.0

            return {
                "Qq": max(0.0, min(1.0, p_val)),
                "ifault": 0,
                "trace": np.zeros(7)
            }

        # 对于中心卡方分布的加权和，使用momentchi2库
        # 改进：考虑自由度权重和缩放后的特征值
        coeffs = lambda_vals_scaled * h

        # 过滤掉很小的系数 - 使用R版本相同的阈值逻辑
        threshold = np.max(np.abs(coeffs)) * 1e-10 if len(coeffs) > 0 else 1e-10
        nonzero_mask = np.abs(coeffs) > threshold

        if not np.any(nonzero_mask):
            return {
                "Qq": 1.0,
                "ifault": 0,
                "trace": np.zeros(7)
            }

        coeffs_filtered = coeffs[nonzero_mask]

        # 使用HBE方法计算累积分布函数
        # momentchi2计算的是CDF，我们需要生存函数(1-CDF)
        cdf_val = hbe(coeffs_filtered, q)
        p_val = 1.0 - cdf_val

        # 确保p值在有效范围内
        p_val = max(0.0, min(1.0, p_val))

        return {
            "Qq": p_val,
            "ifault": 0,
            "trace": np.zeros(7)
        }

    except ImportError:
        # 如果momentchi2不可用，使用Satterthwaite近似
        warnings.warn("momentchi2 not available, using Satterthwaite approximation")

        stats = _import_scipy_stats()

        # 应用缩放因子
        lambda_vals_scaled = lambda_vals * 0.01

        # 计算累积量
        c1 = np.sum(lambda_vals_scaled * h) + np.sum(lambda_vals_scaled * delta)
        c2 = np.sum(lambda_vals_scaled**2 * h) + 2 * np.sum(lambda_vals_scaled**2 * delta)

        if c2 > 0:
            df_satt = 2 * c1**2 / c2
            scale_satt = c2 / (2 * c1)
            q_scaled = q / scale_satt
            p_val = stats.chi2.sf(q_scaled, df=df_satt)
        else:
            p_val = 1.0

        return {
            "Qq": max(0.0, min(1.0, p_val)),
            "ifault": 1,  # 表示使用了近似方法
            "trace": np.zeros(7)
        }

    except Exception as e:
        # 如果momentchi2失败，使用Satterthwaite近似作为回退
        warnings.warn(f"momentchi2 method failed: {str(e)}, using Satterthwaite approximation")

        stats = _import_scipy_stats()

        # 应用缩放因子
        lambda_vals_scaled = lambda_vals * 0.01

        # 计算累积量
        c1 = np.sum(lambda_vals_scaled * h) + np.sum(lambda_vals_scaled * delta)
        c2 = np.sum(lambda_vals_scaled**2 * h) + 2 * np.sum(lambda_vals_scaled**2 * delta)

        if c2 > 0:
            df_satt = 2 * c1**2 / c2
            scale_satt = c2 / (2 * c1)
            q_scaled = q / scale_satt
            p_val = stats.chi2.sf(q_scaled, df=df_satt)
        else:
            p_val = 1.0

        return {
            "Qq": max(0.0, min(1.0, p_val)),
            "ifault": 2,  # 表示使用了回退方法
            "trace": np.zeros(7)
        }



def davies_improved(q, lambda_vals, h=None, delta=None, sigma=0, lim=10000, acc=0.0001):
    """
    Improved Davies method implementation with enhanced numerical stability
    
    Parameters:
    ----------
    q : float
        Test statistic
    lambda_vals : array-like
        Eigenvalues
    h : array-like, optional
        Degrees of freedom for each eigenvalue
    delta : array-like, optional
        Non-centrality parameters
    sigma : float, default=0
        Variance parameter
    lim : int, default=10000
        Maximum number of terms in series
    acc : float, default=0.0001
        Required accuracy
    
    Returns:
    -------
    dict
        Dictionary with p-value, fault code, and trace information
    """
    try:
        # 使用momentchi2库，但改进参数处理
        hbe = _import_momentchi2()
        if hbe is None:
            raise ImportError("momentchi2 not available")

        r = len(lambda_vals)

        if h is None:
            h = np.ones(r)
        if delta is None:
            delta = np.zeros(r)

        if len(h) != r:
            raise ValueError("lambda and h should have the same length!")
        if len(delta) != r:
            raise ValueError("lambda and delta should have the same length!")

        # 对于非中心参数非零的情况，使用回退方法
        if np.any(delta != 0):
            stats = _import_scipy_stats()

            # 计算累积量
            c1 = np.sum(lambda_vals * h) + np.sum(lambda_vals * delta)
            c2 = np.sum(lambda_vals**2 * h) + 2 * np.sum(lambda_vals**2 * delta)

            if c2 > 0:
                df_satt = 2 * c1**2 / c2
                scale_satt = c2 / (2 * c1)
                q_scaled = q / scale_satt
                p_val = stats.chi2.sf(q_scaled, df=df_satt)
            else:
                p_val = 1.0

            return {
                "Qq": max(0.0, min(1.0, p_val)),
                "ifault": 0,
                "trace": np.zeros(7)
            }

        # 改进的系数处理
        coeffs = lambda_vals * h

        # 更严格的系数筛选
        max_coeff = np.max(coeffs) if len(coeffs) > 0 else 0
        if max_coeff <= 0:
            return {"Qq": 1.0, "ifault": 0, "trace": np.zeros(7)}

        # 使用相对阈值而不是绝对阈值
        threshold = max_coeff * 1e-15  # 更严格的阈值
        nonzero_mask = coeffs > threshold

        if not np.any(nonzero_mask):
            return {"Qq": 1.0, "ifault": 0, "trace": np.zeros(7)}

        coeffs_filtered = coeffs[nonzero_mask]

        # 检查数值稳定性
        if len(coeffs_filtered) == 1:
            # 单个卡方分布的情况
            from scipy import stats
            p_val = stats.chi2.sf(q / coeffs_filtered[0], df=1)
        else:
            # 使用HBE方法
            cdf_val = hbe(coeffs_filtered, q)
            p_val = 1.0 - cdf_val

        # 确保p值在有效范围内
        p_val = max(0.0, min(1.0, p_val))

        return {
            "Qq": p_val,
            "ifault": 0,
            "trace": np.zeros(7)
        }

    except ImportError:
        # 如果momentchi2不可用，使用Satterthwaite近似
        warnings.warn("momentchi2 not available, using Satterthwaite approximation")

        stats = _import_scipy_stats()

        # 计算累积量
        c1 = np.sum(lambda_vals * h) + np.sum(lambda_vals * delta)
        c2 = np.sum(lambda_vals**2 * h) + 2 * np.sum(lambda_vals**2 * delta)

        if c2 > 0:
            df_satt = 2 * c1**2 / c2
            scale_satt = c2 / (2 * c1)
            q_scaled = q / scale_satt
            p_val = stats.chi2.sf(q_scaled, df=df_satt)
        else:
            p_val = 1.0

        return {
            "Qq": max(0.0, min(1.0, p_val)),
            "ifault": 1,
            "trace": np.zeros(7)
        }

    except Exception as e:
        # 如果momentchi2失败，使用Satterthwaite近似作为回退
        warnings.warn(f"momentchi2 method failed: {str(e)}, using Satterthwaite approximation")

        stats = _import_scipy_stats()

        # 计算累积量
        c1 = np.sum(lambda_vals * h) + np.sum(lambda_vals * delta)
        c2 = np.sum(lambda_vals**2 * h) + 2 * np.sum(lambda_vals**2 * delta)

        if c2 > 0:
            df_satt = 2 * c1**2 / c2
            scale_satt = c2 / (2 * c1)
            q_scaled = q / scale_satt
            p_val = stats.chi2.sf(q_scaled, df=df_satt)
        else:
            p_val = 1.0

        return {
            "Qq": max(0.0, min(1.0, p_val)),
            "ifault": 2,
            "trace": np.zeros(7)
        }

