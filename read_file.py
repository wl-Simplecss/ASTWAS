# Data Loading Functions
# Functions for loading genotype and phenotype data from files

import numpy as np

def load_phenotype(file_path):
    """
    Load phenotype data from TSV file
    
    Parameters:
    ----------
    file_path : str
        Path to TSV file containing phenotype data
        
    Returns:
    -------
    np.array
        Phenotype vector
    """
    P = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            try:
                phen_value = float(parts[1])
                P.append(phen_value)
            except ValueError:
                continue

    return np.array(P, dtype=np.float32)


def load_gene_data(txt_path, csv_path, target_gene):
    """
    Extract target gene information from txt file and construct G and W matrices from csv file
    
    Parameters:
    ----------
    txt_path : str
        Path to weight file
    csv_path : str
        Path to gene CSV file
    target_gene : str
        Target gene ID
        
    Returns:
    -------
    tuple
        (G, W) where G is genotype matrix and W is weight structure
    """
    Chr_value = None
    Gene_start = None
    Gene_end = None
    weight_index = {}
    snp_index = {}

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if not parts:
                continue

            gene_id = parts[0]
            if gene_id == target_gene:
                Chr_value = parts[1]
                Gene_start = int(parts[2])
                Gene_end = int(parts[3])

                for col in parts[4:]:
                    sp = col.split('_')
                    if len(sp) >= 5:
                        try:
                            pos = int(sp[1])
                            weight = float(sp[4])
                            snp_value = str(sp[0])
                            weight_index[pos] = weight
                            snp_index[pos] = snp_value
                        except (ValueError, IndexError):
                            continue
                break

    if Chr_value is None:
        return None, None

    G = []
    g_positions = []
    hashset = {}
    index_number = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if not parts:
                continue

            chr_val = parts[0]
            if chr_val != Chr_value:
                continue

            try:
                pos = int(parts[1])
            except ValueError:
                continue

            if pos < Gene_start or pos > Gene_end:
                continue

            try:
                index_number = index_number + 1
                if pos in weight_index:
                    hashset[snp_index[pos]] = weight_index[pos]
                else:
                    snp_string = f"rs_{index_number}"
                    hashset[snp_string] = 0.0

                g_row = [int(x) for x in parts[2:]]
            except ValueError:
                continue

            G.append(g_row)
            g_positions.append(pos)

    n_snps = len(G)
    W = {"hashset": hashset, "nSNP": n_snps}

    return G, W


