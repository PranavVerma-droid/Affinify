import os
import pandas as pd
import requests
from urllib.parse import urljoin
import logging
from typing import Dict, List, Optional, Tuple
import zipfile
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Handles downloading and initial processing of molecular datasets
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and information
        self.datasets = {
            'bindingdb': {
                'url': 'https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202507_tsv.zip',
                'filename': 'BindingDB_All_202507_tsv.zip',
                'description': 'Comprehensive binding database'
            },
            'pdbbind': {
                'url': 'http://www.pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz',
                'filename': 'PDBbind_v2020_refined.tar.gz',
                'description': 'Refined set of protein-ligand complexes'
            }
        }
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """
        Download a specific dataset
        
        Args:
            dataset_name: Name of the dataset to download
            force_download: Whether to re-download if file exists
            
        Returns:
            bool: Success status
        """
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
            
        dataset_info = self.datasets[dataset_name]
        file_path = self.data_dir / dataset_info['filename']
        
        if file_path.exists() and not force_download:
            logger.info(f"Dataset {dataset_name} already exists. Use force_download=True to re-download.")
            return True
            
        try:
            logger.info(f"Downloading {dataset_name}...")
            response = requests.get(dataset_info['url'], stream=True)
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='')
            
            print()  # New line after progress
            logger.info(f"Successfully downloaded {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {str(e)}")
            return False
    
    def extract_dataset(self, dataset_name: str) -> bool:
        """
        Extract compressed dataset files
        
        Args:
            dataset_name: Name of the dataset to extract
            
        Returns:
            bool: Success status
        """
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
            
        dataset_info = self.datasets[dataset_name]
        file_path = self.data_dir / dataset_info['filename']
        
        if not file_path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            return False
            
        try:
            extract_dir = self.data_dir / dataset_name
            extract_dir.mkdir(exist_ok=True)
            
            # Check file type and extract accordingly
            if file_path.suffix == '.zip':
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    logger.info(f"Successfully extracted {dataset_name} to {extract_dir}")
                    return True
                except zipfile.BadZipFile:
                    logger.warning(f"File {file_path} is not a valid zip file, checking if it's already a TSV file...")
                    # Check if the downloaded file is actually a TSV file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            first_line = f.readline().strip()
                            if '\t' in first_line:  # Likely a TSV file
                                # Copy the file to the extract directory with proper extension
                                import shutil
                                tsv_file = extract_dir / f"{dataset_name}.tsv"
                                shutil.copy2(file_path, tsv_file)
                                logger.info(f"File appears to be TSV format, copied to {tsv_file}")
                                return True
                    except Exception as e:
                        logger.error(f"Error checking file format: {e}")
                        return False
                        
            elif file_path.suffix == '.gz':
                import tarfile
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_dir)
                logger.info(f"Successfully extracted {dataset_name} to {extract_dir}")
                return True
            else:
                # Try to handle as a direct file
                import shutil
                target_file = extract_dir / file_path.name
                shutil.copy2(file_path, target_file)
                logger.info(f"Copied {dataset_name} file to {target_file}")
                return True
                
        except Exception as e:
            logger.error(f"Error extracting {dataset_name}: {str(e)}")
            return False
    
    def load_bindingdb_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load and preprocess BindingDB data
        
        Args:
            limit: Maximum number of rows to load
            
        Returns:
            pd.DataFrame: Processed BindingDB data
        """
        bindingdb_dir = self.data_dir / 'bindingdb'
        
        # Find the TSV file
        tsv_files = list(bindingdb_dir.glob('*.tsv'))
        if not tsv_files:
            logger.error("No TSV file found in BindingDB directory")
            return pd.DataFrame()
            
        tsv_file = tsv_files[0]
        
        try:
            logger.info(f"Loading BindingDB data from {tsv_file}")
            
            # Read with specific columns to reduce memory usage
            useful_columns = [
                'Ligand SMILES', 'PDB ID(s)', 'UniProt (SwissProt) Primary ID of Target Chain',
                'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)',
                'Target Name', 'Target Source Organism According to Curator or DataSource',
                'Ligand InChI Key', 'Number of Protein Chains in Target (>1 implies a multichain target)'
            ]
            
            # Read in chunks to handle large files
            chunk_size = 10000
            chunks = []
            
            for chunk in pd.read_csv(tsv_file, sep='\t', chunksize=chunk_size, 
                                   usecols=useful_columns, low_memory=False):
                chunks.append(chunk)
                if limit and len(chunks) * chunk_size >= limit:
                    break
            
            df = pd.concat(chunks, ignore_index=True)
            
            if limit:
                df = df.head(limit)
            
            logger.info(f"Loaded {len(df)} rows from BindingDB")
            return df
            
        except Exception as e:
            logger.error(f"Error loading BindingDB data: {str(e)}")
            return pd.DataFrame()
    
    def create_sample_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create a sample dataset for testing purposes
        
        Args:
            n_samples: Number of samples to create
            
        Returns:
            pd.DataFrame: Sample dataset
        """
        import random
        import string
        
        # Generate sample SMILES strings (simplified)
        sample_smiles = [
            'CCO',  # Ethanol
            'CC(C)O',  # Isopropanol
            'c1ccccc1',  # Benzene
            'CCN(CC)CC',  # Triethylamine
            'CC(=O)O',  # Acetic acid
            'c1ccc2c(c1)ccccc2',  # Naphthalene
            'CC(C)(C)O',  # tert-Butanol
            'CCCCCCCCCCCCCCCC(=O)O',  # Palmitic acid
        ]
        
        # Generate sample protein IDs
        sample_proteins = ['1ABC', '2DEF', '3GHI', '4JKL', '5MNO', '6PQR']
        
        data = []
        for _ in range(n_samples):
            data.append({
                'Ligand SMILES': random.choice(sample_smiles),
                'PDB ID(s)': random.choice(sample_proteins),
                'Ki (nM)': round(random.uniform(0.1, 10000), 2),
                'IC50 (nM)': round(random.uniform(0.1, 10000), 2),
                'Kd (nM)': round(random.uniform(0.1, 10000), 2),
                'Target Name': f"Target_{random.randint(1, 100)}",
                'UniProt (SwissProt) Primary ID of Target Chain': f"P{random.randint(10000, 99999)}"
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Created sample dataset with {len(df)} rows")
        return df
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about available datasets
        
        Returns:
            Dict: Dataset information
        """
        info = {}
        for name, dataset in self.datasets.items():
            file_path = self.data_dir / dataset['filename']
            info[name] = {
                'description': dataset['description'],
                'downloaded': file_path.exists(),
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'url': dataset['url']
            }
        return info

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Get dataset information
    info = collector.get_dataset_info()
    print("Available datasets:")
    for name, details in info.items():
        print(f"  {name}: {details['description']}")
        print(f"    Downloaded: {details['downloaded']}")
        if details['downloaded']:
            print(f"    Size: {details['file_size'] / (1024*1024):.1f} MB")
    
    # Create sample dataset for testing
    sample_df = collector.create_sample_dataset(100)
    print(f"\nSample dataset shape: {sample_df.shape}")
    print(sample_df.head())
