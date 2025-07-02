# SSD and SetID Data Structure Generation
# Generates in-memory SSD (Sample-SNP-Data) and SetID structures for SKAT analysis

import numpy as np

def _import_pandas():
    """Lazy import pandas to avoid startup overhead"""
    import pandas as pd
    return pd


def Generate_SSD_SetID_From_Memory(G_vector, P_vector, Weight, target_gene="SET", Is_FlipGenotype=True):
    """
    Generate SSD and SetID data structures from in-memory genotype and phenotype data
    
    Parameters:
    ----------
    G_vector : list or np.array
        Genotype matrix (n_samples x n_snps)
    P_vector : np.array
        Phenotype vector
    Weight : dict
        Weight dictionary with 'hashset' containing SNP weights
    target_gene : str, default="SET"
        Gene set identifier
    Is_FlipGenotype : bool, default=True
        Whether to flip genotypes to minor allele coding
        
    Returns:
    -------
    dict
        Dictionary containing SSD info and data
    """
    G_array = np.array(G_vector)
    P_array = np.array(P_vector) if P_vector is not None else None
    
    if G_array.shape[0] != len(P_array):
        if G_array.shape[1] == len(P_array):
            G_array = G_array.T
        else:
            raise ValueError(f"Genotype matrix shape {G_array.shape} doesn't match phenotype length {len(P_array)}")
    
    n_samples, n_snps = G_array.shape
    
    snp_ids = []
    if Weight and 'hashset' in Weight:
        hashset = Weight['hashset']
        snp_ids = list(hashset.keys())[:n_snps]
        
        while len(snp_ids) < n_snps:
            snp_ids.append(f"rs_{len(snp_ids) + 1}")
    else:
        snp_ids = [f"rs_{i+1}" for i in range(n_snps)]
    
    snp_ids = snp_ids[:n_snps]
    
    set_info = []
    snp_data_list = []
    
    for snp_idx, snp_id in enumerate(snp_ids):
        temp_snp_info = G_array[:, snp_idx].astype(int).tolist()
        
        temp_snp_info = [9 if g < 0 or g > 2 else g for g in temp_snp_info]
        
        if Is_FlipGenotype:
            valid_genotypes = [g for g in temp_snp_info if g != 9]
            if valid_genotypes:
                allele0_count = sum(2 if g == 0 else 1 if g == 1 else 0 for g in valid_genotypes)
                allele1_count = sum(2 if g == 2 else 1 if g == 1 else 0 for g in valid_genotypes)
                
                if allele0_count < allele1_count:
                    temp_snp_info = [2 if g == 0 else 0 if g == 2 else g for g in temp_snp_info]
        
        snp_data_list.append({
            'SNP_ID': snp_id,
            'genotypes': temp_snp_info
        })
    
    set_info.append({
        'SetIndex': 1,
        'SetID': target_gene,
        'SetSize': n_snps,
        'SNPs': snp_data_list
    })
    
    ssd_data = {
        'sets': set_info,
        'n_samples': n_samples,
        'n_snps': n_snps,
        'bytes_per_snp': (n_samples + 3) // 4
    }
    
    pd = _import_pandas()
    info_data = {
        'WindowSize': -999,
        'MAFConvert': 1 if Is_FlipGenotype else 0,
        'nSNPs': n_snps,
        'nSample': n_samples,
        'nDecodeSize': (n_samples + 3) // 4 + 1,
        'nSets': 1,
        'SetInfo': pd.DataFrame([{
            'SetIndex': 1,
            'Offset': 0,
            'SetID': target_gene,
            'SetSize': n_snps
        }])
    }
    
    return {
        'success': True,
        'ssd_data': ssd_data,
        'info': info_data,
        'missing_log': []
    }


def Generate_SSD_SetID(Is_FlipGenotype=True, G_vector=None, P_vector=None, Weight=None, target_gene="SET"):
    """
    Generate SSD and SetID data structures from in-memory genotype and phenotype data
    
    Parameters:
    ----------
    Is_FlipGenotype : bool, default=True
        Whether to flip genotypes to minor allele coding
    G_vector : list/np.array, required
        In-memory genotype matrix (n_samples x n_snps)
    P_vector : np.array, optional
        In-memory phenotype vector (used for validation)
    Weight : dict, optional
        Weight dictionary with SNP weights
    target_gene : str, default="SET"
        Gene set identifier
        
    Returns:
    -------
    dict
        In-memory SSD and INFO data structures
    """
    if G_vector is None:
        raise ValueError("G_vector is required. This function only supports in-memory data processing.")
    
    print("Using in-memory genotype data")
    result = Generate_SSD_SetID_From_Memory(G_vector, P_vector, Weight, target_gene, Is_FlipGenotype)
    
    try:
        info = result['info']
        n_sample = info['nSample']
        n_sets = info['nSets']
        n_snps = info['nSNPs']
            
        print(f"{n_sample} Samples, {n_sets} Sets, {n_snps} Total SNPs")
        print("SSD and Info data structures created in memory!")
    except Exception as e:
        print(f"Warning: Could not read Info data: {e}")
        print("SSD and Info data structures created in memory!")
    
    return result

