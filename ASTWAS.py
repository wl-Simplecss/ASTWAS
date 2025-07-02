# ASTWAS Analysis Pipeline
# Association Study using Transcriptome-Wide Association Study methods

import os
import time
import argparse
import pandas as pd
import numpy as np

# Local module imports
from skat_ssd_all import skat_ssd_all
from open_ssd import open_ssd
from read_file import load_gene_data, load_phenotype
from generate_ssd_setid import Generate_SSD_SetID
from skat_null_model import SKAT_Null_Model

def run_astwas_analysis(txt_path, csv_path, target_gene, file_path_pheno, output_filepath, input_phenotype_type):
    """
    Execute single ASTWAS analysis task
    
    Parameters:
    ----------
    txt_path : str
        Path to weight file
    csv_path : str
        Path to gene CSV file containing SNP data
    target_gene : str
        Target gene ID
    file_path_pheno : str
        Path to phenotype file
    output_filepath : str
        Output directory path
    input_phenotype_type : str
        Phenotype type ("binary" or "continuous")
    
    Returns:
    -------
    dict
        Analysis results dictionary
    """
    # Convert phenotype type to SKAT format
    if input_phenotype_type == "binary":
        input_phenotype_type = "D"
    else:
        input_phenotype_type = "C"

    # Load gene data and weights
    G_vector, Weight = load_gene_data(txt_path, csv_path, target_gene)

    # Load phenotype data
    P_vector = load_phenotype(file_path_pheno)
    P_vector = P_vector - 1  # Adjust phenotype values
    
    # Convert genotype data to proper format
    G_array = np.array(G_vector, dtype=np.int32).T

    # Prepare data for null model
    data = pd.DataFrame({'y': P_vector})

    # Fit null model
    obj = SKAT_Null_Model('y ~ 1', data=data, out_type=input_phenotype_type)

    # Create output directory
    os.makedirs(output_filepath, exist_ok=True)

    # Generate SSD data structure
    ssd_result = Generate_SSD_SetID(
        Is_FlipGenotype=True,
        G_vector=G_array,
        P_vector=P_vector,
        Weight=Weight,
        target_gene=target_gene
    )

    # Open SSD data and perform association analysis
    info = open_ssd(ssd_result)
    result = skat_ssd_all(info, obj, obj_snp_weight=Weight)

    # Save results
    file_path_result = os.path.join(output_filepath, f"{target_gene}.result") 
    result["results"].to_csv(file_path_result, sep='\t', index=False)

    # Calculate analysis timing
    end_time_all = time.time()
    total_analysis_time = end_time_all - start_time_all
    
    # Print completion summary
    print("\n" + "=" * 80)
    print("                           Analysis Completed Successfully")
    print("=" * 80)
    
    print(f"\n**  Timing Summary:")
    print(f"  Total analysis time: {total_analysis_time:.3f} seconds ({total_analysis_time/60:.2f} minutes)")
    
    print(f"\n** Results Summary:")
    print(f"  Gene:        {target_gene}")
    print(f"  Samples:     {result['results'].iloc[0]['N.Marker.Test'] if len(result['results']) > 0 else 'N/A'}")
    print(f"  SNPs:        {result['results'].iloc[0]['N.Marker.All'] if len(result['results']) > 0 else 'N/A'}")
    pvalue = result['results'].iloc[0]['P.value'] if len(result['results']) > 0 else None
    pvalue_str = f"{pvalue:.2e}" if pvalue is not None else "N/A"
    print(f"  P-value:     {pvalue_str}")
    print(f"  Output:      {file_path_result}")
    print("\n" + "=" * 80)
    
    return result

if __name__ == "__main__":
    # Configure numpy and pandas display options
    np.set_printoptions(threshold=np.inf)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Print banner
    print("=" * 80)
    print("                           ASTWAS Analysis ")
    print("=" * 80)

    # Start timing
    start_time_all = time.time()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run ASTWAS analysis.")
    parser.add_argument("--weight_file", type=str, help="Path to the SNP weight file")
    parser.add_argument("--gene_csv", type=str, help="Path to the gene CSV file containing SNP IDs")
    parser.add_argument("--gene_name", type=str, help="Target gene name or ID")
    parser.add_argument("--pheno_file", type=str, help="Path to the phenotype file")
    parser.add_argument("--output_dir", type=str, default="./out", help="Output directory (default: ./out)")
    parser.add_argument("--input_phenotype_type", type=str, default="continuous", 
                       help="Phenotype type: continuous|binary (default: continuous)")
    parser.add_argument("--task_file", type=str, help="Path to task file for batch processing")

    args = parser.parse_args()

    # Batch processing from task file
    if args.task_file:
        print(f"Reading tasks from file: {args.task_file}")
        with open(args.task_file, 'r') as task_file:
            for line_num, line in enumerate(task_file, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                try:
                    # Parse tab-delimited parameters
                    params = line.split('\t')
                    if len(params) < 6:
                        print(f"Warning: Line {line_num} has insufficient parameters, skipping")
                        continue
                        
                    txt_path, csv_path, target_gene, file_path_pheno, input_phenotype_type, output_filepath = params[:6]
                    
                    print("\n" + "=" * 80)
                    print(f"Processing task {line_num}: Gene {target_gene}")
                    print("=" * 80 + "\n")
                    
                    # Execute ASTWAS analysis
                    run_astwas_analysis(
                        txt_path=txt_path,
                        csv_path=csv_path,
                        target_gene=target_gene,
                        file_path_pheno=file_path_pheno,
                        output_filepath=output_filepath,
                        input_phenotype_type=input_phenotype_type
                    )
                except Exception as e:
                    print(f"Error processing task on line {line_num}: {str(e)}")
    else:
        # Single analysis from command line arguments
        if not args.weight_file or not args.gene_csv or not args.gene_name or not args.pheno_file:
            parser.error("When not using --task_file, the following arguments are required: "
                        "--weight_file, --gene_csv, --gene_name, --pheno_file")
        
        # Execute single ASTWAS analysis
        run_astwas_analysis(
            txt_path=args.weight_file,
            csv_path=args.gene_csv,
            target_gene=args.gene_name,
            file_path_pheno=args.pheno_file,
            output_filepath=args.output_dir,
            input_phenotype_type=args.input_phenotype_type
        )

