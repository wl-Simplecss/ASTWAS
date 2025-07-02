import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Import functions from ASTWAS_utils
from ASTWAS_utils import (
    beta_weights, 
    impute, 
    impute_r_version
)


def get_snp_maf(Z, id_include=None):
    """Get minor allele frequencies for SNPs"""
    if id_include is None:
        MAF = np.nanmean(Z, axis=0) / 2
    else:
        if isinstance(Z, pd.DataFrame):
            MAF = np.nanmean(Z.iloc[id_include].values, axis=0) / 2
        else:
            MAF = np.nanmean(Z[id_include, :], axis=0) / 2

    return MAF


def skat_get_maf(Z, id_include=None):
    """Calculate Minor Allele Frequencies"""
    if id_include is None:
        MAF = np.nanmean(Z, axis=0) / 2
    else:
        if isinstance(Z, pd.DataFrame):
            MAF = np.nanmean(Z.iloc[id_include].values, axis=0) / 2
        else:
            MAF = np.nanmean(Z[id_include, :], axis=0) / 2

    return MAF


def check_z(Z, n, id_include, set_id=None, weights=None, weights_beta=(1, 25),
            impute_method="fixed", is_check_genotype=True, is_dosage=False,
            missing_cutoff=0.15, max_maf=1, estimate_MAF=1):
    """
    Check Z matrix, perform imputation and weights calculation
    
    Parameters:
    ----------
    Z : array-like
        Genotype matrix
    n : int
        Sample size
    id_include : array-like
        Indices of samples to include
    set_id : str, optional
        Set identifier
    weights : array-like, optional
        Custom weights for SNPs
    weights_beta : tuple, default=(1, 25)
        Beta distribution parameters for weights
    impute_method : str, default="fixed"
        Imputation method
    is_check_genotype : bool, default=True
        Whether to check genotype validity
    is_dosage : bool, default=False
        Whether genotypes are dosage values
    missing_cutoff : float, default=0.15
        Missing rate cutoff for SNP exclusion
    max_maf : float, default=1
        Maximum MAF cutoff
    estimate_MAF : int, default=1
        MAF estimation method
    
    Returns:
    -------
    dict
        Dictionary with processed Z matrix, weights, and other information
    """

    if not isinstance(Z, (np.ndarray, np.matrix, pd.DataFrame)):
        raise ValueError("Z is not a matrix")

    if Z.shape[0] != n:
        raise ValueError("Dimensions of y and Z do not match")

    if is_dosage:
        impute_method = "fixed"

    if isinstance(Z, np.ndarray):
        if Z.shape[1] > 0:
            Z = pd.DataFrame(Z, columns=[f"VAR{i}" for i in range(1, Z.shape[1] + 1)])
    elif hasattr(Z, 'columns') and Z.columns is None:
        Z.columns = [f"VAR{i}" for i in range(1, Z.shape[1] + 1)]

    if not is_check_genotype and not is_dosage:
        if isinstance(Z, pd.DataFrame):
            Z_test = Z.iloc[id_include].values
        else:
            Z_test = Z[id_include, :]

        if not isinstance(Z_test, (np.ndarray, np.matrix)):
            Z_test = np.array(Z_test)

        return {"Z.test": Z_test, "weights": weights, "MAF": np.zeros(Z.shape[1]),
                "id_include.test": id_include, "return": 0}

    if estimate_MAF == 2:
        if isinstance(Z, pd.DataFrame):
            Z = Z.iloc[id_include].copy()
        else:
            Z = Z[id_include, :].copy()
        id_include = np.arange(len(id_include))

    original_is_df = isinstance(Z, pd.DataFrame)
    if original_is_df:
        col_names = Z.columns.tolist()
        Z_values = Z.values.copy()
    else:
        col_names = None
        Z_values = Z.copy()

    idx_miss_na = np.where(np.isnan(Z_values))
    idx_miss_9 = np.where(Z_values == 9)
    idx_miss = (np.concatenate([idx_miss_na[0], idx_miss_9[0]]),
                np.concatenate([idx_miss_na[1], idx_miss_9[1]]))

    if len(idx_miss[0]) > 0:
        Z_values[idx_miss] = np.nan

    m = Z_values.shape[1]
    id_include_snp = []

    MAF_toCutoff = skat_get_maf(Z_values, id_include=None)

    for i in range(m):
        missing_ratio = np.sum(np.isnan(Z_values[:, i])) / n
        sd1 = np.nanstd(Z_values[:, i])

        if missing_ratio < missing_cutoff and sd1 > 0:
            if MAF_toCutoff[i] < max_maf:
                id_include_snp.append(i)

    if len(id_include_snp) == 0:
        if set_id is None:
            msg = "ALL SNPs have either high missing rates or no-variation. P-value=1"
        else:
            msg = f"In {set_id}, ALL SNPs have either high missing rates or no-variation. P-value=1"

        warnings.warn(msg)
        return {
            "p.value": 1,
            "p.value.resampling": np.nan,
            "Test.Type": np.nan,
            "Q": np.nan,
            "param": {"n.marker": 0, "n.marker.test": 0},
            "return": 1
        }

    elif m - len(id_include_snp) > 0:
        if set_id is None:
            msg = f"{m - len(id_include_snp)} SNPs with either high missing rates or no-variation are excluded!"
        else:
            msg = f"In {set_id}, {m - len(id_include_snp)} SNPs with either high missing rates or no-variation are excluded!"

        warnings.warn(msg)
        Z_values = Z_values[:, id_include_snp]
        if col_names is not None:
            col_names = [col_names[i] for i in id_include_snp]

    MAF_Org = skat_get_maf(Z_values, id_include=None)

    if col_names is not None and len(id_include_snp) < len(col_names):
        MAF_Org_col_names = [col_names[i] for i in id_include_snp]
    else:
        MAF_Org_col_names = col_names

    Z_values = skat_main_check_z_impute(Z_values, id_include, impute_method, set_id)

    MAF = skat_get_maf(Z_values, id_include=None)
    MAF1 = skat_get_maf(Z_values, id_include=id_include)

    if np.sum(MAF1 > 0) == 0:
        if set_id is None:
            msg = "No polymorphic SNP. P-value = 1"
        else:
            msg = f"In {set_id}, No polymorphic SNP. P-value = 1"

        warnings.warn(msg)
        return {
            "p.value": 1,
            "p.value.resampling": np.nan,
            "Test.Type": np.nan,
            "Q": np.nan,
            "param": {"n.marker": 0, "n.marker.test": 0},
            "return": 1
        }

    if weights is None:
        weights = beta_weights(MAF, weights_beta)
    else:
        if len(id_include_snp) < len(weights):
            weights = weights[id_include_snp]

    if n - len(id_include) > 0:
        id_Z = np.where(MAF1 > 0)[0]

        if len(id_Z) == 0:
            if set_id is None:
                msg = "No polymorphic SNP. P-value = 1"
            else:
                msg = f"In {set_id}, No polymorphic SNP. P-value = 1"

            warnings.warn(msg)
            return {
                "p.value": 1,
                "p.value.resampling": np.nan,
                "Test.Type": np.nan,
                "Q": np.nan,
                "param": {"n.marker": 0, "n.marker.test": 0},
                "return": 1
            }

        elif len(id_Z) == 1:
            Z_values = Z_values[:, id_Z].reshape(-1, 1)
        else:
            Z_values = Z_values[:, id_Z]

        if weights is not None:
            weights = weights[id_Z]

    if Z_values.shape[1] == 1:
        if set_id is None:
            msg = "Only one SNP in the SNP set!"
        else:
            msg = f"In {set_id}, Only one SNP in the SNP set!"

        Z_test = Z_values[id_include, :].reshape(-1, 1)
    else:
        Z_test = Z_values[id_include, :]

    if MAF_Org_col_names is not None and len(MAF_Org_col_names) == len(MAF_Org):
        MAF_named = pd.Series(MAF_Org, index=MAF_Org_col_names)
    else:
        MAF_named = MAF_Org

    return {
        "Z.test": Z_test,
        "weights": weights,
        "MAF": MAF_named,
        "id_include.test": id_include,
        "return": 0
    }


def check_z_flip(Z, id_include=None):
    """Check and flip genotypes if needed"""
    MAF = get_snp_maf(Z, id_include)
    idx_err = np.where(MAF > 0.5)[0]

    if len(idx_err) > 0:
        warnings.warn("Genotypes of some variants are not the number of minor alleles! These genotypes are flipped!")
        Z[:, idx_err] = 2 - Z[:, idx_err]

    return Z


def check_z_impute(Z, id_include, impute_method, set_id=None):
    """Check Z matrix and perform imputation"""
    MAF = get_snp_maf(Z, id_include=None)
    MAF1 = get_snp_maf(Z, id_include=id_include)
    MAF_Org = MAF.copy()

    idx_miss = np.where(np.isnan(Z) | (Z == 9))
    if len(idx_miss[0]) > 0:
        if set_id is None:
            msg = f"The missing genotype rate is {len(idx_miss[0]) / Z.size}. Imputation is applied."
        else:
            msg = f"In {set_id}, the missing genotype rate is {len(idx_miss[0]) / Z.size}. Imputation is applied."

        warnings.warn(msg)
        Z = impute(Z, impute_method)

    Z = check_z_flip(Z, id_include)

    MAF = get_snp_maf(Z, id_include=None)
    MAF1 = get_snp_maf(Z, id_include=id_include)

    idx_err = np.where(MAF > 0.5)[0]
    if len(idx_err) > 0:
        warnings.warn(f"MAF: {MAF[idx_err]}")
        raise ValueError("ERROR! genotype flipping")

    return Z


def skat_main_check_z_impute(Z, id_include, impute_method, set_id):
    """Impute missing genotypes and check validity"""
    MAF = skat_get_maf(Z, id_include=None)
    MAF1 = skat_get_maf(Z, id_include=id_include)
    MAF_Org = MAF.copy()

    idx_miss_na = np.where(np.isnan(Z))
    idx_miss_9 = np.where(Z == 9)
    idx_miss = (np.concatenate([idx_miss_na[0], idx_miss_9[0]]),
                np.concatenate([idx_miss_na[1], idx_miss_9[1]]))

    if len(idx_miss[0]) > 0:
        if set_id is None:
            msg = f"The missing genotype rate is {len(idx_miss[0]) / Z.size:.6f}. Imputation is applied."
        else:
            msg = f"In {set_id}, the missing genotype rate is {len(idx_miss[0]) / Z.size:.6f}. Imputation is applied."

        warnings.warn(msg)
        Z = impute_r_version(Z, impute_method)

    Z = skat_main_check_z_flip(Z, id_include)

    MAF = skat_get_maf(Z, id_include=None)
    MAF1 = skat_get_maf(Z, id_include=id_include)

    idx_err = np.where(MAF > 0.5)[0]
    if len(idx_err) > 0:
        warnings.warn(f"MAF: {MAF[idx_err]}")
        raise ValueError("ERROR! genotype flipping")

    return Z


def skat_main_check_z_flip(Z, id_include):
    """Check and flip genotypes to minor allele coding"""
    MAF = skat_get_maf(Z, id_include=None)
    idx_err = np.where(MAF > 0.5)[0]

    if len(idx_err) > 0:
        msg = "Genotypes of some variants are not the number of minor alleles! These genotypes are flipped!"
        warnings.warn(msg)
        Z[:, idx_err] = 2 - Z[:, idx_err]

    return Z


def skat_check_method(method, r_corr, n=None, m=None):
    """
    Check and process SKAT method parameters

    Parameters:
    ----------
    method : str
        SKAT method name
    r_corr : float or array-like
        Correlation coefficient(s)
    n : int, optional
        Number of samples
    m : int, optional
        Number of markers

    Returns:
    -------
    dict
        Dictionary with processed method, r_corr, and IsMeta flag
    """
    is_meta = False

    valid_methods = [
        "liu", "davies", "liu.mod", "optimal", "optimal.moment",
        "optimal.mod", "adjust", "optimal.adj", "optimal.moment.adj",
        "SKAT", "SKATO", "Burden", "SKATO.m", "davies.M", "optimal.adj.M"
    ]

    if method not in valid_methods:
        raise ValueError("Invalid method!")

    if method == "davies.M":
        is_meta = True
        method = "davies"
    elif method == "optimal.adj.M":
        is_meta = True
        method = "optimal.adj"

    if method == "SKAT":
        method = "davies"
        r_corr = 0
    elif method == "SKATO":
        method = "optimal.adj"
    elif method == "SKATO.m":
        method = "optimal.moment.adj"
    elif method == "Burden":
        method = "davies"
        r_corr = 1

    r_corr_array = np.atleast_1d(r_corr)
    if (method == "optimal" or method == "optimal.moment") and len(r_corr_array) == 1:
        r_corr = np.arange(0, 11) / 10
    elif (method in ["optimal.mod", "optimal.adj", "optimal.moment.adj"]) and len(r_corr_array) == 1:
        r_corr = np.array([0, 0.1 ** 2, 0.2 ** 2, 0.3 ** 2, 0.5 ** 2, 0.5, 1])

    if method == "optimal":
        method = "davies"
    elif method == "optimal.moment":
        method = "liu.mod"

    if n is not None and m is not None and len(np.atleast_1d(r_corr)) > 1:
        if m / n < 1 and n > 5000:
            is_meta = True

    return {
        "method": method,
        "r_corr": r_corr,
        "is_meta": is_meta
    }


def skat_check_r_corr(kernel, r_corr):
    """
    Check r_corr parameters

    Parameters:
    ----------
    kernel : str
        Kernel type
    r_corr : float or array-like
        Correlation coefficient(s)
    """
    r_corr_array = np.atleast_1d(r_corr)

    if len(r_corr_array) == 1 and r_corr_array[0] == 0:
        return

    if kernel not in ["linear", "linear.weighted"]:
        raise ValueError("Error: non-zero r.corr only can be used with linear or linear.weighted kernels")

    for r in r_corr_array:
        if r < 0 or r > 1:
            raise ValueError("Error: r.corr should be >= 0 and <= 1")


