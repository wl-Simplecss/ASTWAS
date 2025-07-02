import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Suppress numpy warnings for numerical computations (normal in SKAT calculations)
np.seterr(divide='ignore', invalid='ignore')

# Import functions from other modules
from ASTWAS_data_prep import (
    check_z,
    skat_check_method,
    skat_check_r_corr
)
from linear_logistic import (
    skat_linear_linear,
    skat_linear_other,
    skat_logistic_linear, 
    skat_logistic_other
)
from ASTWAS_utils import single_snp_info

# Global environment to track state
SSD_ENV = {
    "SSD_FILE_OPEN.isOpen": 0,
    "SSD_FILE_OPEN.FileName": ""
}



def skat_main_with_nullmodel(Z, obj_res, kernel="linear.weighted", method="davies",
                             weights_beta=(1, 25), weights=None, impute_method="fixed",
                             r_corr=0, is_check_genotype=True, is_dosage=False,
                             missing_cutoff=0.15, max_maf=1, estimate_MAF=1,
                             set_id=None, out_z=None):
    """
    Run SKAT with null model
    
    Parameters:
    ----------
    Z : array-like
        Genotype matrix (n x m)
    obj_res : object
        Null model object with residuals and other model information
    kernel : str, default="linear.weighted"
        Kernel type
    method : str, default="davies"
        Statistical method for p-value calculation
    weights_beta : tuple, default=(1, 25)
        Beta distribution parameters for weights
    weights : array-like, optional
        Custom weights for SNPs
    impute_method : str, default="fixed"
        Imputation method for missing genotypes
    r_corr : float or array-like, default=0
        Correlation coefficient(s) for SKAT-O
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
    set_id : str, optional
        Set identifier for error messages
    out_z : dict, optional
        Pre-computed Z matrix check results
        
    Returns:
    -------
    dict
        SKAT results including p-value and test statistics
    """
    n = Z.shape[0]
    m = Z.shape[1]
    
    # Process method and r_corr parameters
    method_result = skat_check_method(method, r_corr, n=n, m=m)
    method = method_result["method"]
    r_corr = method_result["r_corr"]
    is_meta = method_result["is_meta"]
    
    # Check r_corr parameters
    skat_check_r_corr(kernel, r_corr)
    
    # For old version compatibility
    if not hasattr(obj_res, 'n_all'):
        obj_res.n_all = n
    
    # Check Z matrix
    if out_z is None:
        out_z = check_z(Z, obj_res.n_all, obj_res.id_include, set_id, weights,
                        weights_beta, impute_method, is_check_genotype, is_dosage,
                        missing_cutoff, max_maf, estimate_MAF)
    
    if out_z["return"] == 1:
        out_z["param"]["n.marker"] = m
        return out_z
    
    # Handle r_corr for special cases
    r_corr_array = np.atleast_1d(r_corr)
    if len(r_corr_array) > 1 and out_z["Z.test"].shape[1] <= 1:
        r_corr = 0
        method = "davies"
    elif len(r_corr_array) > 1:
        # Check if all columns are identical (rank = 1)
        Z_test = out_z["Z.test"]
        if Z_test.shape[1] > 1:
            first_col = Z_test[:, [0]]
            if np.sum(np.abs(Z_test - first_col)) == 0:
                r_corr = 0
                method = "davies"
                warnings.warn("Rank of the genotype matrix is one! SKAT is used instead of SKAT-O!")

    # Run appropriate SKAT implementation based on outcome type
    if obj_res.out_type == "C":  # Continuous outcome
        if kernel in ["linear", "linear.weighted"]:
            re = skat_linear_linear(obj_res.res, out_z["Z.test"], obj_res.X1, kernel,
                                    out_z["weights"], obj_res.s2, method,
                                    obj_res.res_out, obj_res.n_resampling, r_corr, 
                                    is_meta=is_meta)
        else:
            re = skat_linear_other(obj_res.res, out_z["Z.test"], obj_res.X1, kernel,
                                   out_z["weights"], obj_res.s2, method,
                                   obj_res.res_out, obj_res.n_resampling)
    elif obj_res.out_type == "D":  # Binary outcome
        if kernel in ["linear", "linear.weighted"]:
            re = skat_logistic_linear(obj_res.res, out_z["Z.test"], obj_res.X1, kernel,
                                      out_z["weights"], obj_res.pi_1, method,
                                      obj_res.res_out, obj_res.n_resampling, r_corr,
                                      is_meta=is_meta)
        else:
            re = skat_logistic_other(obj_res.res, out_z["Z.test"], obj_res.X1, kernel,
                                     out_z["weights"], obj_res.pi_1, method,
                                     obj_res.res_out, obj_res.n_resampling)
    else:
        raise ValueError("Invalid outcome type! Should be 'C' (continuous) or 'D' (binary)")
    
    # Add marker information
    re["param"]["n.marker"] = m
    re["param"]["n.marker.test"] = out_z["Z.test"].shape[1]
    re["test.snp.mac"] = single_snp_info(out_z["Z.test"])
    
    return re


def skat_ssd_one_set_set_index(ssd_info, set_index, obj, **kwargs):
    """Run SKAT on one set in SSD file by set index"""
    # Get SNP weights if provided
    obj_snp_weight = kwargs.pop("obj_snp_weight", None)

    # Get genotypes and weights using our new implementation
    re1 = get_snp_weight(ssd_info, set_index, obj_snp_weight)




    # Run SKAT with or without weights
    if not re1["Is.weights"]:
        re = skat_main_with_nullmodel(re1["Z"], obj, **kwargs)
    else:
        re = skat_main_with_nullmodel(re1["Z"], obj, weights=re1["weights"], **kwargs)


    return re


def skat_ssd_one_set(ssd_info, set_id, obj, **kwargs):
    """Run SKAT on one set in SSD file by set ID"""
    # Find the set index
    id1 = ssd_info['SetInfo'][ssd_info['SetInfo']['SetID'] == set_id]
    if len(id1) == 0:
        raise ValueError(f"Error: cannot find set id [{set_id}] from SSD!")

    set_index = id1['SetIndex'].values[0]

    # Run SKAT on the set
    re = skat_ssd_one_set_set_index(ssd_info, set_index, obj, **kwargs)

    return re


def skat_ssd_all(ssd_info, obj, **kwargs):
    """Run SKAT on all sets in an SSD file

    Parameters:
    ----------
    ssd_info : dict
        SSD file information returned by open_ssd
    obj : object
        SKAT_Null_Model object or equivalent with required attributes
    **kwargs :
        Additional arguments including obj_snp_weight

    Returns:
    -------
    dict
        SKAT results including p-values and other statistics
    """
    # Get SNP weights if provided
    obj_snp_weight = kwargs.pop("obj_snp_weight", None)

    # Get total number of sets
    n_set = ssd_info['nSets']

    # Initialize output arrays
    out_pvalue = np.full(n_set, np.nan)
    out_marker = np.full(n_set, np.nan)
    out_marker_test = np.full(n_set, np.nan)
    out_error = np.full(n_set, -1)
    out_pvalue_resampling = None
    out_snp_mac = {}

    # Adapt the obj to ensure it has the required attributes
    # This makes it compatible with both R-like SKAT_Null_Model objects
    # and Python native objects
    class NullModelAdapter:
        def __init__(self, obj):
            # 创建一个通用的获取属性/键值的函数
            def get_value(obj, keys, default=None):
                """从对象或字典中获取值，支持多个可能的键名"""
                if not isinstance(keys, list):
                    keys = [keys]
                
                for key in keys:
                    if isinstance(obj, dict):
                        if key in obj:
                            return obj[key]
                    else:
                        if hasattr(obj, key):
                            return getattr(obj, key)
                return default
            
            # Mandatory attributes - 修复：正确获取 out_type，不要默认为 "C"
            # 这是关键修复：确保正确检测二元 vs 连续结果
            out_type_detected = get_value(obj, ["out_type"])
            if out_type_detected is not None:
                self.out_type = out_type_detected
            else:
                # 只有在真的找不到时才警告并默认为 "C"
                warnings.warn("Could not detect out_type from null model object, defaulting to 'C' (continuous)")
                self.out_type = "C"
            
            # Handle id_include (could be named differently in R objects)
            self.id_include = get_value(obj, ["id_include", "id.include"], np.arange(ssd_info['nSample']))
            
            # Handle residuals
            self.res = get_value(obj, ["res", "residuals"], np.zeros(ssd_info['nSample']))
            
            # Handle design matrix
            self.X1 = get_value(obj, ["X1", "X"], np.ones((ssd_info['nSample'], 1)))
            
            # Handle resampling
            self.n_resampling = get_value(obj, ["n_resampling", "n.Resampling", "n_Resampling"], 0)
            
            # Handle resampling residuals
            self.res_out = get_value(obj, ["res_out", "res.out", "res_out"])
            
            # For continuous outcomes
            if self.out_type == "C":
                self.s2 = get_value(obj, ["s2", "sigma2"], 1.0)
            
            # For binary outcomes  
            if self.out_type == "D":
                pi_1_value = get_value(obj, ["pi_1", "pi.1"])
                if pi_1_value is not None:
                    self.pi_1 = pi_1_value
                else:
                    # 如果找不到 pi_1，尝试默认值但发出警告
                    warnings.warn("Could not find pi_1 for binary outcome, using default values")
                    self.pi_1 = np.ones(ssd_info['nSample']) * 0.5
            
            # For compatibility with the SKAT_NULL_Model_ADJ
            re1_obj = get_value(obj, ["re1"])
            if re1_obj is not None:
                self.re1 = NullModelAdapter(re1_obj)
            
            # Add n_all attribute as it might be used
            self.n_all = ssd_info['nSample']



    # Adapt the obj
    adapted_obj = NullModelAdapter(obj)


    # Check for resampling
    is_resampling = False
    n_resampling = 0

    if adapted_obj.n_resampling > 0:
        is_resampling = True
        n_resampling = adapted_obj.n_resampling
        out_pvalue_resampling = np.zeros((n_set, n_resampling))
    elif hasattr(adapted_obj, 're1') and hasattr(adapted_obj.re1, 'n_resampling') and adapted_obj.re1.n_resampling > 0:
        is_resampling = True
        n_resampling = adapted_obj.re1.n_resampling
        out_pvalue_resampling = np.zeros((n_set, n_resampling))


    # Process each set with progress bar
    for i in tqdm(range(n_set), desc="Processing sets"):
        is_error = True
        try:
            # Pass the adapted object and explicitly use obj_snp_weight kwarg


            re = skat_ssd_one_set_set_index(ssd_info, i + 1, adapted_obj, obj_snp_weight=obj_snp_weight, **kwargs)

            is_error = False
        except Exception as e:
            err_msg = str(e)
            msg = f"Error to run SKAT for {ssd_info['SetInfo'].iloc[i]['SetID']}: {err_msg}"
            warnings.warn(msg)

        if not is_error:
            out_pvalue[i] = re["p.value"]
            out_marker[i] = re["param"]["n.marker"]
            out_marker_test[i] = re["param"]["n.marker.test"]

            if is_resampling and "p.value.resampling" in re:
                out_pvalue_resampling[i, :] = re["p.value.resampling"]

            set_id = ssd_info['SetInfo'].iloc[i]['SetID']
            out_snp_mac[set_id] = re["test.snp.mac"]

    # Create output data frame
    out_tbl = pd.DataFrame({
        "SetID": ssd_info['SetInfo']['SetID'],
        "P.value": out_pvalue,
        "N.Marker.All": out_marker,
        "N.Marker.Test": out_marker_test
    })

    # Create result object
    result = {
        "results": out_tbl,
        "P.value.Resampling": out_pvalue_resampling,
        "OUT.snp.mac": out_snp_mac
    }

    return result


def get_snp_weight(ssd_info, set_index, obj_snp_weight=None):
    """
    Get SNP weights for a specific set - Python implementation of SKAT.SSD.GetSNP_Weight
    
    This function implements the exact same logic as R's SKAT.SSD.GetSNP_Weight function
    
    Parameters:
    ----------
    ssd_info : dict
        SSD information dictionary with SetInfo containing SetIndex, SetID, etc.
    set_index : int  
        The set index to retrieve (1-based indexing like R)
    obj_snp_weight : object, optional
        SNP weight object with hashset attribute (like R SNPWeight object)
        
    Returns:
    -------
    dict
        Dictionary with keys: 'Z' (genotype matrix), 'Is.weights' (bool), 'weights' (array, optional)
    """
    # Step 1: Find the set index in SSD.INFO$SetInfo
    # R code: id1<-which(SSD.INFO$SetInfo$SetIndex == SetIndex)
    set_info = ssd_info['SetInfo']
    
    # 确保set_index是正确的类型
    if isinstance(set_index, np.ndarray):
        set_index = int(set_index[0])
    elif isinstance(set_index, list):
        set_index = int(set_index[0])
    else:
        set_index = int(set_index)
    
    # 查找匹配的SetIndex
    id1_mask = set_info['SetIndex'] == set_index
    id1_indices = set_info[id1_mask]
    
    if len(id1_indices) == 0:
        raise ValueError(f"Error: cannot find set index [{set_index}] from SSD!")
    
    # R code: SetID<-SSD.INFO$SetInfo$SetID[id1]
    set_id = id1_indices['SetID'].iloc[0]
    
    # Step 2: Determine if weights should be used  
    # R code: is_Weight = FALSE; if(!is.null(obj.SNPWeight)){ is_Weight = TRUE }
    is_weight = False
    if obj_snp_weight is not None:
        is_weight = True
    
    # Step 3: Get genotypes with SNP IDs
    # R code: try1<-try(Get_Genotypes_SSD(SSD.INFO, SetIndex, is_ID=TRUE),silent = TRUE)
    try:
        # 导入并使用正确的get_genotypes_ssd函数
        from open_ssd import get_genotypes_ssd
        Z = get_genotypes_ssd(ssd_info, set_index, is_id=True)
        is_error = False
        
        # 确保Z是正确的格式
        if not isinstance(Z, pd.DataFrame):
            # 如果不是DataFrame，尝试转换
            if hasattr(Z, 'shape') and len(Z.shape) == 2:
                # 生成默认的SNP ID


                n_snps = Z.shape[1]
                snp_ids = [f"SNP_{set_index}_{i+1}" for i in range(n_snps)]
                Z = pd.DataFrame(Z, columns=snp_ids)
            else:
                raise ValueError("Invalid genotype matrix format")



        
    except Exception as e:
        # R code: err.msg<-geterrmessage(); msg<-sprintf("Error to get genotypes of %s: %s",SetID, err.msg); stop(msg)
        err_msg = str(e)
        msg = f"Error to get genotypes of {set_id}: {err_msg}"
        raise ValueError(msg)



    
    # Step 4: Handle case without weights
    # R code: if(!is_Weight){ re=list(Z=Z, Is.weights=FALSE); return(re) }
    if not is_weight:
        return {
            "Z": Z,
            "Is.weights": False
        }
    
    # Step 5: Process SNP weights
    # R code: SNP_ID<-colnames(Z); p<-ncol(Z); weights<-rep(0, p)
    if hasattr(Z, 'columns'):
        snp_ids = Z.columns.tolist()
    elif hasattr(Z, 'dtype') and hasattr(Z, 'shape'):
        # 如果Z是numpy数组但没有列名，生成默认ID
        n_snps = Z.shape[1]
        snp_ids = [f"SNP_{set_index}_{i+1}" for i in range(n_snps)]
        Z = pd.DataFrame(Z, columns=snp_ids)
    else:
        raise ValueError("Genotype matrix Z does not have column names (SNP IDs)")
    
    p = len(snp_ids)
    weights = np.zeros(p)


    
    # Step 6: Extract weights for each SNP
    # R code: for(i in 1:p){ val1<-SNP_ID[i]; val2<-obj.SNPWeight$hashset[[val1]]; ... }
    for i in range(p):
        val1 = snp_ids[i]
        
        # 检查权重对象的结构
        if hasattr(obj_snp_weight, 'hashset'):
            # R风格的权重对象
            try:
                if hasattr(obj_snp_weight.hashset, 'get'):
                    # 如果是dict-like对象
                    val2 = obj_snp_weight.hashset.get(val1)
                elif hasattr(obj_snp_weight.hashset, '__getitem__'):
                    # 如果支持索引访问
                    val2 = obj_snp_weight.hashset[val1]
                else:
                    # 尝试直接访问属性
                    val2 = getattr(obj_snp_weight.hashset, val1, None)
            except (KeyError, AttributeError):
                val2 = None
        elif isinstance(obj_snp_weight, dict) and 'hashset' in obj_snp_weight:
            # Python dict格式的权重对象
            try:
                val2 = obj_snp_weight['hashset'].get(val1)
            except (KeyError, AttributeError):
                val2 = None
        else:
            # 直接在权重对象中查找
            try:
                if hasattr(obj_snp_weight, 'get'):
                    val2 = obj_snp_weight.get(val1)
                elif hasattr(obj_snp_weight, '__getitem__'):
                    val2 = obj_snp_weight[val1]
                else:
                    val2 = getattr(obj_snp_weight, val1, None)
            except (KeyError, AttributeError):
                val2 = None
        
        # R code: if(is.null(val2)){ msg<-sprintf("SNP %s is not found in obj.SNPWeight!", val1); stop(msg) }
        if val2 is None:
            raise ValueError(f"SNP {val1} is not found in obj.SNPWeight!")
        
        # R code: weights[i]<-val2
        weights[i] = float(val2)

    
    # Step 7: Return results with weights
    # R code: re=list(Z=Z, Is.weights=TRUE, weights=weights); return(re)
    return {
        "Z": Z,
        "Is.weights": True,
        "weights": weights
    }

