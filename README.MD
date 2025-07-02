## Users' Manual of ASTWAS (Version 1.0.0)
### ASTWAS: Modeling alternative polyadenylation and SNP Effects in Kernel-Driven TWAS reveals novel genetic associations for complex traits
### Preamble
![ASTWAS_model_all3.png](./EXAMPLE/ASTWAS_model_all3.png)
The ASTWAS (Association Study using Transcriptome-Wide Association Study) program is a Python-based implementation that integrates transcriptome-wide association studies with kernel-based statistical testing. 
We propose ASTWAS, Modeling alternative polyadenylation and SNP Effects in Kernel-Driven TWAS reveals novel genetic associations for complex traits. By utilizing a kernel function to capture non-linear genetic effects, ASTWAS significantly improves detection power. Extensive simulations demonstrate that our model outperforms the existing 3`aTWAS method under most conditions. In real-data applications, our model successfully identified numerous known risk genes and also uncovered novel genetic associations.


Key features:
- **Kernel-based testing**: Uses kernel machine for improved statistical power
- **Flexible phenotype support**: Handles both continuous and binary traits
- **Batch processing**: Supports analysis of multiple genes simultaneously
- **Memory-efficient**: Optimized data structures for large-scale analysis
- **Cross-platform**: Pure Python implementation with minimal dependencies

### Installation

ASTWAS is implemented in Python and requires the following dependencies:

**Required Python packages:**
- numpy  --  1.23.5
- pandas  --  1.5.3
- scipy  --  1.10.0
- statsmodels  --  0.13.5
- patsy  --  0.5.3
- tqdm  --  4.64.1

Other versions are not tested, although they may work. 

**Optional packages (for enhanced performance):**
- momentchi2  --  0.1.8 (for improved Davies method implementation)

**Installation steps:**
1. Clone the repository:
   ```bash
   git clone https://github.com/wl-Simplecss/ASTWAS.git
   cd ASTWAS
   ```

2. Install required dependencies:
   ```bash
   pip install numpy pandas scipy statsmodels patsy tqdm
   ```

3. Optional: Install momentchi2 for improved Davies method:
   ```bash
   pip install momentchi2
   ```

**System Requirements:**
- Python 3.7 or higher
- At least 4GB RAM (8GB+ recommended for large datasets)
- Compatible with Windows and Linux

### Usage

#### Command Line Arguments

**Single Gene Analysis:**
```bash
python ASTWAS.py --weight_file WEIGHT_FILE --gene_csv GENE_CSV_FILE --gene_name GENE_ID --pheno_file PHENOTYPE_FILE --input_phenotype_type PHENOTYPE_TYPE --output_dir OUTPUT_DIRECTORY
```

**Batch Processing:**
```bash
python ASTWAS.py --task_file TASK_FILE
```

#### Parameters Description

- `--weight_file`: Path to the SNP weight file (ElasticNet weights from GTEx)
- `--gene_csv`: Path to the gene CSV file containing SNP genotype data
- `--gene_name`: Target gene name or Ensembl ID
- `--pheno_file`: Path to the phenotype file (tab-separated format)
- `--input_phenotype_type`: Phenotype type (`continuous` or `binary`)
- `--output_dir`: Output directory path (default: `./out`)
- `--task_file`: Path to task file for batch processing (tab-delimited format)

#### Input File Formats

**Weight File Format (ElasticNet weights):**
```
GENE_ID	CHR	START	END	SNP_INFO1	SNP_INFO2	...
ENSG00000250334.5	1	12345	67890	rs123_12345_A_G_0.123	rs456_23456_C_T_0.456	...
```

**Gene CSV Format:**
```
CHR,POS,SAMPLE1,SAMPLE2,SAMPLE3,...
1,12345,0,1,2,...
1,23456,1,0,1,...
```

**Phenotype File Format:**
```
SAMPLE_ID	PHENOTYPE
SAMPLE1	1.23
SAMPLE2	-0.45
SAMPLE3	2.10
```

**Task File Format (for batch processing):**
```
WEIGHT_FILE	GENE_CSV	GENE_NAME	PHENO_FILE	PHENOTYPE_TYPE	OUTPUT_DIR
path/to/weights.txt	path/to/gene1.csv	GENE1	path/to/pheno.txt	continuous	./out
path/to/weights.txt	path/to/gene2.csv	GENE2	path/to/pheno.txt	binary	./out
```

### Examples

#### Single Gene Analysis Example

**For continuous phenotype:**
```bash
python ASTWAS.py \
    --weight_file "ElasticNet_DB/ElasticNet_Whole_Blood.txt" \
    --gene_csv "EXAMPLE/CSV_FORMAT/example.csv" \
    --gene_name "ENSG00000250334.5" \
    --pheno_file "EXAMPLE/CSV_FORMAT/example.tsv" \
    --input_phenotype_type continuous \
    --output_dir "./out"
```

**For binary phenotype:**
```bash
python ASTWAS.py \
    --weight_file "ElasticNet_DB/ElasticNet_Whole_Blood.txt" \
    --gene_csv "EXAMPLE/CSV_FORMAT/example.csv" \
    --gene_name "ENSG00000250334.5" \
    --pheno_file "EXAMPLE/CSV_FORMAT/example.tsv" \
    --input_phenotype_type binary \
    --output_dir "./out"
```

#### Batch Processing Example

```bash
python ASTWAS.py --task_file input/task_gene.txt
```

Where `task_gene.txt` contains multiple analysis tasks in tab-delimited format.

### Output

ASTWAS generates the following output files in the specified output directory:

- `{GENE_NAME}.result`: Main results file containing:
  - `SetID`: Gene identifier
  - `P.value`: Association p-value
  - `N.Marker.All`: Total number of SNPs in the gene
  - `N.Marker.Test`: Number of SNPs used in testing


### Citations

If you use ASTWAS in your research, please cite:

**For the SKAT methodology:**
- Wu MC, Lee S, Cai T, Li Y, Boehnke M, Lin X. "Rare-variant association testing for sequencing data with the sequence kernel association test." Am J Hum Genet. 2011;89(1):82-93.

**For GTEx elastic net weights:**
- "The GTEx Consortium atlas of genetic regulatory effects across human tissues", Aguet et al, 2019
- "A gene-based association method for mapping traits using reference transcriptome data", Gamazon et al, 2015, Nature Genetics
- "Exploring the phenotypic consequences of tissue specific gene expression variation inferred from GWAS summary statistics", Barbeira et al, 2018, Nature Communications

**For TWAS methodology:**
- Gusev A, Ko A, Shi H, et al. "Integrative approaches for large-scale transcriptome-wide association studies." Nat Genet. 2016;48(3):245-252.

### Contacts

For questions, bug reports, or feature requests, please contact:

- **Chen Cao**: [caochen@njmu.edu.cn]
- **Yan Wang**: [wy6868@jlu.edu.cn]
- **lei Wang**: [wlei24@mails.jlu.edu.cn]



### Copyright License (MIT Open Source)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Version History

**Version 1.0.0**
- Initial release
- Support for continuous and binary phenotypes
- Batch processing capabilities
- Memory-optimized data structures
- Cross-platform compatibility
---

*Last updated: [2025-07-02]* 
