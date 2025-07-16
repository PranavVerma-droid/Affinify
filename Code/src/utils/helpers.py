"""
Utility functions for the Affinify project
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd #type: ignore
import numpy as np

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger('affinify')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def create_directory_structure(base_path: str) -> None:
    """
    Create the standard directory structure for the project
    
    Args:
        base_path: Base directory path
    """
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'models',
        'results',
        'logs',
        'notebooks',
        'config',
        'tests',
        'docs'
    ]
    
    base = Path(base_path)
    
    for directory in directories:
        dir_path = base / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep for empty directories
        gitkeep_file = dir_path / '.gitkeep'
        if not any(dir_path.iterdir()):
            gitkeep_file.touch()

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_file: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        return json.load(f)

def save_config(config: Dict[str, Any], config_file: str) -> None:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_file: Path to configuration file
    """
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        True if valid, False otherwise
    """
    try:
        from rdkit import Chem #type: ignore
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        # Simple validation without RDKit
        if not smiles or not isinstance(smiles, str):
            return False
        
        # Basic checks for valid SMILES characters
        valid_chars = set('CNOSPcnospFBrClI[]()=#+-.0123456789')
        return all(c in valid_chars for c in smiles)

def convert_binding_affinity(value: float, unit: str, target_unit: str = 'nM') -> float:
    """
    Convert binding affinity between different units
    
    Args:
        value: Binding affinity value
        unit: Original unit (nM, uM, mM, M)
        target_unit: Target unit (default: nM)
    
    Returns:
        Converted value
    """
    # Conversion factors to nM
    to_nM = {
        'M': 1e9,
        'mM': 1e6,
        'uM': 1e3,
        'μM': 1e3,
        'nM': 1,
        'pM': 1e-3
    }
    
    # Convert to nM first
    nM_value = value * to_nM.get(unit, 1)
    
    # Convert to target unit
    from_nM = {
        'M': 1e-9,
        'mM': 1e-6,
        'uM': 1e-3,
        'μM': 1e-3,
        'nM': 1,
        'pM': 1e3
    }
    
    return nM_value * from_nM.get(target_unit, 1)

def calculate_molecular_properties(smiles: str) -> Dict[str, float]:
    """
    Calculate basic molecular properties from SMILES
    
    Args:
        smiles: SMILES string
    
    Returns:
        Dictionary of molecular properties
    """
    try:
        from rdkit import Chem #type: ignore
        from rdkit.Chem import Descriptors #type: ignore
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        properties = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'num_hbd': Descriptors.NumHDonors(mol),
            'num_hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_rings': Descriptors.RingCount(mol)
        }
        
        return properties
        
    except ImportError:
        # Fallback without RDKit
        return {
            'smiles_length': len(smiles),
            'num_carbons': smiles.count('C'),
            'num_nitrogens': smiles.count('N'),
            'num_oxygens': smiles.count('O'),
            'num_rings': smiles.count('c')
        }

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess dataset
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove rows with missing SMILES
    if 'Ligand SMILES' in df.columns:
        df = df.dropna(subset=['Ligand SMILES'])
        
        # Validate SMILES
        valid_smiles = df['Ligand SMILES'].apply(validate_smiles)
        df = df[valid_smiles]
    
    # Clean binding affinity values
    binding_columns = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
    for col in binding_columns:
        if col in df.columns:
            # Remove negative values
            df[col] = df[col].apply(lambda x: x if pd.isna(x) or x > 0 else np.nan)
    
    return df

def get_project_root() -> Path:
    """
    Get the project root directory
    
    Returns:
        Path to project root
    """
    current_path = Path(__file__).parent
    
    # Look for markers of project root
    markers = ['README.md', 'requirements.txt', 'setup.py', '.git']
    
    for path in [current_path] + list(current_path.parents):
        if any((path / marker).exists() for marker in markers):
            return path
    
    # Fallback to current directory
    return current_path

def format_number(value: float, decimal_places: int = 3) -> str:
    """
    Format number for display
    
    Args:
        value: Number to format
        decimal_places: Number of decimal places
    
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"
    
    if abs(value) >= 1000:
        return f"{value:.{decimal_places}e}"
    else:
        return f"{value:.{decimal_places}f}"

def create_sample_prediction_data() -> pd.DataFrame:
    """
    Create sample data for testing predictions
    
    Returns:
        Sample DataFrame
    """
    sample_data = {
        'Ligand SMILES': [
            'CCO',  # Ethanol
            'CC(=O)O',  # Acetic acid
            'c1ccccc1',  # Benzene
            'CCN(CC)CC',  # Triethylamine
            'CC(C)O',  # Isopropanol
            'c1ccc2c(c1)ccccc2',  # Naphthalene
            'CCCCCCCCCCCCCCCC(=O)O',  # Palmitic acid
            'CC(C)(C)O'  # tert-Butanol
        ],
        'PDB ID(s)': [
            '1ABC', '2DEF', '3GHI', '4JKL', 
            '5MNO', '6PQR', '7STU', '8VWX'
        ],
        'Target Name': [
            'Protein kinase A',
            'GABA receptor',
            'Acetylcholine esterase',
            'Ion channel',
            'Cytochrome P450',
            'DNA polymerase',
            'Ribosomal protein',
            'Membrane transporter'
        ]
    }
    
    return pd.DataFrame(sample_data)

def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are installed
    
    Returns:
        Dictionary of dependency status
    """
    dependencies = {
        'pandas': False,
        'numpy': False,
        'scikit-learn': False,
        'matplotlib': False,
        'seaborn': False,
        'rdkit': False,
        'tensorflow': False,
        'torch': False,
        'xgboost': False,
        'streamlit': False,
        'plotly': False,
        'py3dmol': False
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies

def print_system_info():
    """Print system information and dependency status"""
    import sys
    import platform
    
    print("=== Affinify System Information ===")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor: {platform.processor()}")
    
    print("\n=== Dependency Status ===")
    deps = check_dependencies()
    
    for dep, status in deps.items():
        status_str = "✅ Installed" if status else "❌ Missing"
        print(f"{dep}: {status_str}")
    
    missing_deps = [dep for dep, status in deps.items() if not status]
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install -r requirements.txt")
    else:
        print("\n✅ All dependencies are installed!")

if __name__ == "__main__":
    # Example usage
    print_system_info()
    
    # Create sample data
    sample_df = create_sample_prediction_data()
    print(f"\nSample data shape: {sample_df.shape}")
    print(sample_df.head())
