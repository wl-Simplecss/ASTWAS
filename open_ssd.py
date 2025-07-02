# SSD File Operations
# Functions for opening, reading, and managing SSD (Sample-SNP-Data) files

import os
import numpy as np
import pandas as pd

# Global environment to track SSD file state
SSD_ENV = {
    "SSD_FILE_OPEN.isOpen": 0,
    "SSD_FILE_OPEN.FileName": ""
}

# Global variable for file handle
SSD_FILE_HANDLE = None

def check_file_exists(filename):
    """Check if file exists and raise error if not"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")

def print_error_ssd(code):
    """Handle SSD error codes"""
    if code == 0:
        return 0
    elif code == 1:
        raise RuntimeError("Error Can't open BIM file")
    elif code == 2:
        raise RuntimeError("Error Can't open FAM file")
    elif code == 3:
        raise RuntimeError("Error Can't open BED file")
    elif code == 4:
        raise RuntimeError("Error Can't open SETID file")
    elif code == 5:
        raise RuntimeError("Error Can't write SSD file")
    elif code == 6:
        raise RuntimeError("Error Can't read SSD file")
    elif code == 7:
        raise RuntimeError("Error Can't write INFO file")
    elif code == 8:
        raise RuntimeError("Error Can't read INFO file")
    elif code == 9:
        raise RuntimeError("Error Can't write INFO file")
    elif code == 13:
        raise RuntimeError("Error Wrong SNP or Individual sizes")
    elif code == 14:
        raise RuntimeError("Error SetID not found")
    else:
        raise RuntimeError(f"Error [{code}]")


def print_file_info(info):
    """Print SSD file information summary"""
    msg = f"{info['nSample']} Samples, {info['nSets']} Sets, {info['nSNPs']} Total SNPs"
    print(msg)


def read_file_info(file_info):
    """Read INFO file and parse SSD file information"""
    check_file_exists(file_info)

    # Read first 6 lines of basic information
    info1 = pd.read_csv(file_info, sep='\t', header=None, nrows=6)
    # Skip separator line and read set information
    info2 = pd.read_csv(file_info, sep='\t', header=None, skiprows=7, comment="#")
    
    # Create INFO object
    info = {}
    info['WindowSize'] = info1.iloc[0, 0]
    info['MAFConvert'] = info1.iloc[1, 0]
    info['nSNPs'] = info1.iloc[2, 0]
    info['nSample'] = info1.iloc[3, 0]
    info['nDecodeSize'] = info1.iloc[4, 0]
    info['nSets'] = info1.iloc[5, 0]

    # Ensure info2 has at least 4 columns
    if info2.shape[1] < 4:
        raise ValueError(
            f"INFO file format error: expected at least 4 columns in set info section, got {info2.shape[1]}")

    # Create SetInfo DataFrame
    info['SetInfo'] = pd.DataFrame({
        'SetIndex': info2.iloc[:, 0],
        'Offset': info2.iloc[:, 1],
        'SetID': info2.iloc[:, 2],
        'SetSize': info2.iloc[:, 3]
    })

    print_file_info(info)
    return info


def close_ssd():
    """Close SSD file and clean up resources"""
    global SSD_FILE_HANDLE

    if SSD_ENV["SSD_FILE_OPEN.isOpen"] == 1:
        if SSD_FILE_HANDLE is not None:
            # Check if it's a memory-based reader
            if isinstance(SSD_FILE_HANDLE, MemoryBasedSSDReader):
                SSD_FILE_HANDLE = None
            else:
                # For file-based reader, close the file handle
                if hasattr(SSD_FILE_HANDLE, 'm_file') and SSD_FILE_HANDLE.m_file is not None:
                    SSD_FILE_HANDLE.m_file.close()
                SSD_FILE_HANDLE = None

        msg = f"Close the opened SSD data: {SSD_ENV['SSD_FILE_OPEN.FileName']}"
        print(msg)

        SSD_ENV["SSD_FILE_OPEN.isOpen"] = 0
        SSD_ENV["SSD_FILE_OPEN.FileName"] = ""
    else:
        print("No opened SSD data!")

class MwoFileReader:
    """File reader for SSD binary files"""

    def __init__(self, filename, myerror=None, info=None):
        self.m_filename = filename
        self.m_info_file = info

        try:
            self.m_file = open(filename, 'rb')
        except Exception as e:
            if myerror is not None:
                myerror[0] = 6  # Cannot read SSD file
            raise e

        # Read INFO file to get necessary information
        self.read_info_file(info)

    def read_info_file(self, info_path):
        """Read necessary information from INFO file"""
        with open(info_path, 'r') as f:
            lines = f.readlines()

        # Read basic information
        self.m_win_size = int(lines[0].split('\t')[0])
        self.m_ovlp_size = int(lines[1].split('\t')[0])  # MAF conversion status
        self.m_num_of_different_snps = int(lines[2].split('\t')[0])
        self.m_num_of_individuals = int(lines[3].split('\t')[0])
        self.m_num_of_bytes_per_line = int(lines[4].split('\t')[0])
        self.m_total_num_of_sets = int(lines[5].split('\t')[0])

        # Calculate total SNP count
        if self.m_win_size != -999 and self.m_ovlp_size != -999:
            self.m_total_num_of_snps = self.calculate_total_snps()
        else:
            self.m_total_num_of_snps = self.m_num_of_different_snps

        # Read offset table
        self.m_offsetarr = np.zeros(self.m_total_num_of_sets, dtype=np.int64)
        self.m_set_size = np.zeros(self.m_total_num_of_sets, dtype=np.int64)
        self.upload_offsets_table(lines)

    def calculate_total_snps(self):
        """Calculate total SNP count including overlaps"""
        return (self.m_win_size +
                ((self.m_num_of_different_snps - self.m_win_size) /
                 (self.m_win_size - self.m_ovlp_size)) * self.m_win_size +
                ((self.m_num_of_different_snps - self.m_win_size) %
                 (self.m_win_size - self.m_ovlp_size)) + self.m_ovlp_size)

    def upload_offsets_table(self, lines):
        """Read offset table for set positions"""
        # Skip header information and separator
        set_info_start = 7

        for i in range(self.m_total_num_of_sets):
            if set_info_start + i < len(lines):
                parts = lines[set_info_start + i].strip().split('\t')
                self.m_offsetarr[i] = int(parts[1])  # offset

                if self.m_win_size != -999:
                    self.m_set_size[i] = self.m_win_size
                else:
                    self.m_set_size[i] = int(parts[3])  # set size

    def get_TotalNumberofSets(self, total_num_sets):
        """Return total number of sets"""
        total_num_sets[0] = self.m_total_num_of_sets

    def get_TotalNumberofSNPs(self, total_num_snp):
        """Return total number of SNPs"""
        total_num_snp[0] = self.m_num_of_different_snps

    def get_TotalNumberofInd(self, total_num_ind):
        """Return total number of individuals"""
        total_num_ind[0] = self.m_num_of_individuals

    def get_NumberofSnps(self, set_id, myerror=None):
        """Return number of SNPs in specified set"""
        if set_id > 0 and set_id <= self.m_total_num_of_sets:
            return self.m_set_size[set_id - 1]
        else:
            if myerror is not None:
                myerror[0] = 14  # SetID not found
            return -9999

    def __del__(self):
        """Destructor to ensure file is closed"""
        if hasattr(self, 'm_file') and self.m_file is not None:
            self.m_file.close()


def open_mwa(file_ssd, file_info, myerror):
    """Open SSD file using MwoFileReader"""
    global SSD_FILE_HANDLE

    try:
        # Create MwoFileReader object
        reader = MwoFileReader(file_ssd, myerror, file_info)
        SSD_FILE_HANDLE = reader
        return 0
    except Exception as e:
        print(f"Error opening SSD file: {str(e)}")
        return 6  # Cannot read SSD file


def open_ssd(ssd_data_or_file, info_data_or_file=None):
    """
    Open SSD data and read information.
    Works with both file paths and in-memory data structures.
    
    Parameters:
    ----------
    ssd_data_or_file : dict or str
        Either in-memory SSD data structure or path to SSD file
    info_data_or_file : dict or str, optional
        Either in-memory INFO data structure or path to INFO file
        
    Returns:
    -------
    dict
        SSD information dictionary
    """
    global SSD_FILE_HANDLE

    # Close any previously opened SSD
    if SSD_ENV["SSD_FILE_OPEN.isOpen"] == 1:
        close_ssd()

    # Check if we have in-memory data structures
    if isinstance(ssd_data_or_file, dict) and 'ssd_data' in ssd_data_or_file:
        # Working with in-memory data structures
        ssd_data = ssd_data_or_file['ssd_data']
        info_data = ssd_data_or_file['info']
        
        # Create a memory-based SSD handler
        SSD_FILE_HANDLE = MemoryBasedSSDReader(ssd_data, info_data)
        
        print_file_info(info_data)
        print("Open the SSD data from memory")
        
        # Set global state
        SSD_ENV["SSD_FILE_OPEN.isOpen"] = 1
        SSD_ENV["SSD_FILE_OPEN.FileName"] = "memory_ssd"
        
        return info_data
    
    else:
        # Working with file paths (original functionality)
        file_ssd = os.path.abspath(ssd_data_or_file)
        file_info = os.path.abspath(info_data_or_file)
        
        # Check if files exist
        check_file_exists(file_ssd)
        check_file_exists(file_info)
        
        # Read INFO file
        info = read_file_info(file_info)

        # Open SSD file
        err_code = [0]
        result = open_mwa(file_ssd, file_info, err_code)
        print_error_ssd(err_code[0])

        print("Open the SSD file")

        # Set global state
        SSD_ENV["SSD_FILE_OPEN.isOpen"] = 1
        SSD_ENV["SSD_FILE_OPEN.FileName"] = file_ssd

        return info


class MemoryBasedSSDReader:
    """Memory-based SSD reader for in-memory data structures"""
    
    def __init__(self, ssd_data, info_data):
        self.ssd_data = ssd_data
        self.info_data = info_data
        self.m_file = None  # No actual file handle
        
        # Set up data structures similar to MwoFileReader
        self.m_total_num_of_sets = info_data['nSets']
        self.m_num_of_individuals = info_data['nSample']
        self.m_num_of_different_snps = info_data['nSNPs']
        self.m_num_of_bytes_per_line = info_data['nDecodeSize']
        
    def get_TotalNumberofSets(self, total_num_sets):
        """Return total number of sets"""
        total_num_sets[0] = self.m_total_num_of_sets
        
    def get_TotalNumberofSNPs(self, total_num_snp):
        """Return total number of SNPs"""
        total_num_snp[0] = self.m_num_of_different_snps
        
    def get_TotalNumberofInd(self, total_num_ind):
        """Return total number of individuals"""
        total_num_ind[0] = self.m_num_of_individuals
        
    def get_NumberofSnps(self, set_id, myerror=None):
        """Return number of SNPs in specified set"""
        if set_id > 0 and set_id <= self.m_total_num_of_sets:
            set_data = self.ssd_data['sets'][set_id - 1]  # Convert to 0-based index
            return set_data['SetSize']
        else:
            if myerror is not None:
                myerror[0] = 14  # SetID not found
            return -9999


def get_genotypes_ssd(ssd_info, set_index, is_id=True):
    """
    Get genotype data from SSD (memory or file).
    Works with both memory and file-based data.
    
    Parameters:
    ----------
    ssd_info : dict
        SSD information dictionary
    set_index : int
        Set index (1-based)
    is_id : bool
        Whether to return SNP IDs
        
    Returns:
    -------
    pandas.DataFrame or numpy.ndarray
        Genotype matrix
    """
    global SSD_FILE_HANDLE

    # Check if SSD is opened
    if SSD_ENV["SSD_FILE_OPEN.isOpen"] == 0:
        raise RuntimeError("SSD file is not opened. Please open it first!")

    # Find the specified set
    set_info_df = ssd_info['SetInfo']
    id1 = set_info_df[set_info_df['SetIndex'] == set_index]
    if len(id1) == 0:
        raise ValueError(f"Error: cannot find set index [{set_index}] from SSD!")

    set_row = id1.iloc[0]
    n_snp = int(set_row['SetSize'])
    n_sample = int(ssd_info['nSample'])

    # Check if we're using memory-based reader
    if isinstance(SSD_FILE_HANDLE, MemoryBasedSSDReader):
        # Get data from memory
        set_data = SSD_FILE_HANDLE.ssd_data['sets'][set_index - 1]  # Convert to 0-based
        
        genotype_matrix = []
        snp_ids = []
        
        for snp_data in set_data['SNPs']:
            genotype_matrix.append(snp_data['genotypes'])
            snp_ids.append(snp_data['SNP_ID'])
        
        # Convert to numpy array and transpose (SNPs x Samples -> Samples x SNPs)
        Z_matrix = np.array(genotype_matrix).T
        
        if is_id:
            return pd.DataFrame(Z_matrix, columns=snp_ids)
        else:
            return Z_matrix
    
    else:
        # Original file-based implementation
        # Create output array
        size = n_sample * n_snp
        Z = np.full(size, 9, dtype=np.int32)
        snp_ids = []

        # Position file pointer
        if SSD_FILE_HANDLE.m_file.seekable():
            SSD_FILE_HANDLE.m_file.seek(int(set_row['Offset']))
        else:
            raise IOError("SSD file not seekable")

        # Read genotype data
        Zind = 0
        for snp_idx in range(n_snp):
            # Read SNP ID
            snp_id = ""
            while True:
                char = SSD_FILE_HANDLE.m_file.read(1).decode('utf-8', errors='replace')
                if char == " ":
                    break
                if char == "\n":
                    if snp_idx == 0:
                        raise ValueError(f"Unexpected end of line when reading SNP ID")
                    Z_partial = Z[:Zind]
                    Z_out_t = Z_partial.reshape((snp_idx, n_sample), order='C')
                    Z_matrix = Z_out_t.T
                    snp_ids_partial = snp_ids[:snp_idx]
                    return pd.DataFrame(Z_matrix, columns=snp_ids_partial) if is_id else Z_matrix
                snp_id += char

            snp_ids.append(snp_id)

            # Read genotype bytes
            bytes_to_read = (n_sample + 3) // 4
            genotype_bytes = SSD_FILE_HANDLE.m_file.read(bytes_to_read)

            if len(genotype_bytes) < bytes_to_read:
                raise ValueError(f"Unexpected end of file, could not read complete genotype data for SNP {snp_id}")

            # Decode genotype data
            ind_count = 0
            for byte in genotype_bytes:
                for i in range(4):
                    if ind_count >= n_sample:
                        break
                        
                    shift = i * 2
                    bits = (byte >> shift) & 0x3
                    
                    if bits == 0x0:
                        genotype = 0
                    elif bits == 0x1:
                        genotype = 9
                    elif bits == 0x2:
                        genotype = 1
                    elif bits == 0x3:
                        genotype = 2

                    Z[Zind] = genotype
                    Zind += 1
                    ind_count += 1

            # Read newline
            newline = SSD_FILE_HANDLE.m_file.read(1)
            if newline != b'\n':
                raise ValueError(f"Expected newline after genotype data for SNP {snp_id}, got {newline}")

        # Reshape and transpose
        Z_out_t = Z.reshape((n_snp, n_sample), order='C')
        Z_matrix = Z_out_t.T

    if is_id:
        return pd.DataFrame(Z_matrix, columns=snp_ids)
    else:
        return Z_matrix


def print_file_info_memory(info_data):
    """Print information for in-memory INFO data"""
    msg = f"{info_data['nSample']} Samples, {info_data['nSets']} Sets, {info_data['nSNPs']} Total SNPs"
    print(msg)


