#!/usr/bin/env python3
"""
Affinify CLI - Unified Data Processing and Model Training Script
A single script to handle all data operations and model training for the Affinify project.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import argparse
import subprocess
import shutil
from typing import Optional, Tuple, Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import unified configuration
try:
    from config.manager import get_config, get_config_manager
    config = get_config()
    config_manager = get_config_manager()
    LOGGING_CONFIG = config.logging
    DATA_CONFIG = config.data
    MODELS_CONFIG = config.models
    PATHS_CONFIG = config.paths
except ImportError as e:
    print(f"Warning: Could not import unified config: {e}")
    # Fallback configuration
    class FallbackConfig:
        level = "INFO"
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file = "logs/affinity_cli.log"
    LOGGING_CONFIG = FallbackConfig()
    config_manager = None

try:
    from data_processing.data_collector import DataCollector
    from data_processing.feature_extractor import MolecularFeatureExtractor
    from models.ml_models import ModelTrainer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the src directory is properly set up.")
    sys.exit(1)

# Configure logging using unified config
def setup_logging(log_level=None):
    """Setup logging configuration using unified config"""
    if log_level is None:
        log_level = getattr(LOGGING_CONFIG, 'level', 'INFO')
    
    log_file = getattr(LOGGING_CONFIG, 'file', 'logs/affinity_cli.log')
    log_format = getattr(LOGGING_CONFIG, 'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    os.makedirs('logs', exist_ok=True)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

class AffinifyCLI:
    """Main CLI class for Affinify operations"""
    
    def __init__(self, logger):
        self.logger = logger
        self.data_collector = DataCollector()
        self.feature_extractor = MolecularFeatureExtractor()
        self.model_trainer = ModelTrainer()
        
    def create_directories(self):
        """Create necessary directories"""
        self.logger.info("Creating data directories...")
        
        directories = [
            "data/raw/bindingdb",
            "data/processed",
            "data/external",
            "models",
            "results",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def guide_manual_download(self):
        """Guide user through manual download process"""
        print("\n" + "="*70)
        print("📥 MANUAL DOWNLOAD REQUIRED")
        print("="*70)
        print()
        print("BindingDB dataset needs to be downloaded manually.")
        print("Please follow these steps:")
        print()
        print("1. 🌐 Go to: https://www.bindingdb.org/rwd/bind/downloads/")
        print("2. 📁 Find: 'BindingDB_All_202507_tsv.zip' (488 MB)")
        print("3. 💾 Download the file")
        print("4. 📂 Place it in: data/raw/bindingdb/")
        print("5. 📦 Extract the zip file in the same directory")
        print()
        print("Expected file structure after extraction:")
        print("  data/raw/bindingdb/")
        print("  ├── BindingDB_All_202507_tsv.zip")
        print("  └── BindingDB_All_202507.tsv")
        print()
        print("⚠️  The TSV file should be about 1.2GB after extraction.")
        print()
        
        # Wait for user to complete download
        input("Press ENTER after you've downloaded and extracted the file...")
        print()
    
    def check_bindingdb_file(self) -> Optional[Path]:
        """Check if BindingDB file exists and return path"""
        self.logger.info("Checking for BindingDB file...")
        
        bindingdb_dir = Path("data/raw/bindingdb")
        
        # Check for various possible file names
        possible_files = [
            "BindingDB_All_202507.tsv",
            "BindingDB_All_202507_tsv.tsv",
            "BindingDB_All.tsv"
        ]
        
        for filename in possible_files:
            file_path = bindingdb_dir / filename
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024*1024)  # Size in MB
                self.logger.info(f"Found BindingDB file: {filename} ({file_size:.1f} MB)")
                return file_path
        
        # If not found, look for any TSV file
        tsv_files = list(bindingdb_dir.glob("*.tsv"))
        if tsv_files:
            file_path = tsv_files[0]
            file_size = file_path.stat().st_size / (1024*1024)
            self.logger.info(f"Found TSV file: {file_path.name} ({file_size:.1f} MB)")
            return file_path
        
        return None
    
    def process_bindingdb_data(self, max_rows: int = 50000) -> Optional[pd.DataFrame]:
        """Process BindingDB data if available"""
        self.logger.info("Processing BindingDB data...")
        
        # Check if already processed
        processed_file = Path("data/processed/bindingdb_subset.csv")
        if processed_file.exists():
            self.logger.info("Loading existing BindingDB subset...")
            return pd.read_csv(processed_file)
        
        # Check for raw file
        bindingdb_file = self.check_bindingdb_file()
        if not bindingdb_file:
            self.logger.warning("No BindingDB file found")
            return None
        
        try:
            self.logger.info("Reading BindingDB file in chunks...")
            chunk_size = 10000
            chunks = []
            max_chunks = max_rows // chunk_size
            
            for i, chunk in enumerate(pd.read_csv(bindingdb_file, sep='\t', chunksize=chunk_size, 
                                               low_memory=False, encoding='utf-8')):
                
                # Filter for useful columns that exist
                useful_columns = [
                    'Ligand SMILES', 'PDB ID(s)', 'Ki (nM)', 'IC50 (nM)', 
                    'Kd (nM)', 'EC50 (nM)', 'Target Name', 'UniProt (SwissProt) Primary ID of Target Chain'
                ]
                
                available_columns = [col for col in useful_columns if col in chunk.columns]
                if available_columns:
                    chunk = chunk[available_columns]
                    
                    # Basic filtering - remove rows with missing SMILES
                    if 'Ligand SMILES' in chunk.columns:
                        initial_len = len(chunk)
                        chunk = chunk.dropna(subset=['Ligand SMILES'])
                        if i == 0:  # Log for first chunk only
                            self.logger.info(f"Removed {initial_len - len(chunk)} rows with missing SMILES from first chunk")
                        
                        # Remove very short SMILES (likely invalid)
                        chunk = chunk[chunk['Ligand SMILES'].str.len() >= 3]
                    
                    # Less aggressive filtering - just check if we have any useful data
                    if len(chunk) > 0:
                        chunks.append(chunk)
                
                if i >= max_chunks:
                    break
                    
                if i % 10 == 0:
                    self.logger.info(f"Processed {i * chunk_size:,} rows...")
            
            if not chunks:
                self.logger.warning("No valid data found in BindingDB file")
                return None
            
            # Combine chunks
            self.logger.info(f"Found {len(chunks)} valid chunks to combine")
            df = pd.concat(chunks, ignore_index=True)
            self.logger.info(f"Combined {len(df)} rows from BindingDB")
            
            # Clean data
            if 'Ligand SMILES' in df.columns:
                df = df.dropna(subset=['Ligand SMILES'])
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Clean binding affinity values
            affinity_cols = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
            for col in affinity_cols:
                if col in df.columns:
                    # Convert to numeric, handling non-numeric values
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Remove rows where this column has invalid values (but keep NaN)
                    df = df[(df[col].isna()) | ((df[col] > 0) & (df[col] < 1000000))]
            
            # Save subset
            subset_size = min(max_rows, len(df))
            if len(df) > 0:
                df_subset = df.sample(n=subset_size, random_state=42)
                
                # Save original data for prediction system
                df_subset.to_csv(processed_file, index=False)
                self.logger.info(f"Saved BindingDB subset with {len(df_subset)} rows")
                
                # Create data summary
                summary = {
                    'total_proteins': df_subset['Target Name'].nunique(),
                    'total_ligands': len(df_subset),
                    'affinity_range': {
                        col: {
                            'min': float(df_subset[col].min()),
                            'max': float(df_subset[col].max()),
                            'mean': float(df_subset[col].mean()),
                            'median': float(df_subset[col].median())
                        } for col in affinity_cols if col in df_subset.columns
                    }
                }
                
                # Save summary
                summary_file = Path("data/processed/data_summary.json")
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                return df_subset
            else:
                self.logger.warning("No valid data remaining after cleaning")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing BindingDB: {e}")
            return None
    
    def create_sample_data(self, size: int = 5000) -> pd.DataFrame:
        """Create sample dataset"""
        self.logger.info(f"Creating sample dataset with {size} records...")
        
        sample_df = self.data_collector.create_sample_dataset(size)
        
        # Save sample dataset
        processed_dir = Path("data/processed")
        processed_dir.mkdir(exist_ok=True)
        sample_file = processed_dir / "sample_dataset.csv"
        sample_df.to_csv(sample_file, index=False)
        
        self.logger.info(f"Sample dataset saved to {sample_file}")
        return sample_df
    
    def extract_features(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Extract features from dataset"""
        self.logger.info(f"Extracting features from {dataset_name}...")
        self.logger.info(f"Input dataframe shape: {df.shape}")
        self.logger.info(f"Input dataframe columns: {df.columns.tolist()}")
        
        # Add debug logging
        self.logger.info("Calling prepare_features...")
        try:
            result = self.feature_extractor.prepare_features(df)
            self.logger.info(f"prepare_features returned: {type(result)}")
        except Exception as e:
            self.logger.error(f"prepare_features threw an exception: {e}")
            result = None
        
        if result is None:
            self.logger.error("Feature extraction failed - prepare_features returned None")
            self.logger.error("This might be due to missing required columns or invalid data")
            
            # Try to debug what's in the dataframe
            self.logger.info("Attempting to debug dataframe content...")
            if 'Ligand SMILES' in df.columns:
                self.logger.info(f"SMILES column found with {df['Ligand SMILES'].notna().sum()} non-null values")
            
            # Check for binding affinity columns
            affinity_cols = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
            found_affinity_cols = [col for col in affinity_cols if col in df.columns]
            self.logger.info(f"Found affinity columns: {found_affinity_cols}")
            
            for col in found_affinity_cols:
                non_null_count = df[col].notna().sum()
                self.logger.info(f"{col}: {non_null_count} non-null values")
            
            # Try fallback feature extraction
            self.logger.info("Attempting fallback feature extraction...")
            features, target = self.create_simple_features(df)
            
            if features is None or target is None:
                self.logger.error("Even fallback feature extraction failed!")
                return None, None
            
            self.logger.info("Fallback feature extraction successful!")
        else:
            features, target = result
        
        # Save features
        processed_dir = Path("data/processed")
        features_file = processed_dir / f"{dataset_name}_features.csv"
        target_file = processed_dir / f"{dataset_name}_target.csv"
        
        features.to_csv(features_file, index=False)
        target.to_csv(target_file, index=False)
        
        # Also save as main processed files
        features.to_csv(processed_dir / "processed_features.csv", index=False)
        target.to_csv(processed_dir / "target_values.csv", index=False)
        
        self.logger.info(f"Extracted {len(features)} samples with {len(features.columns)} features")
        return features, target
    
    def create_simple_features(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Create enhanced features for better performance"""
        self.logger.info("Creating enhanced molecular features...")
        
        features_data = []
        target_data = []
        
        for idx, row in df.iterrows():
            smiles = row.get('Ligand SMILES', '')
            if not smiles or pd.isna(smiles) or len(str(smiles)) < 3:
                continue
                
            smiles = str(smiles)
            
            # Enhanced molecular descriptors (25+ features)
            features = {
                # Basic counts
                'mol_weight': len(smiles) * 10,
                'smiles_length': len(smiles),
                'carbon_count': smiles.count('C') + smiles.count('c'),
                'nitrogen_count': smiles.count('N') + smiles.count('n'),
                'oxygen_count': smiles.count('O') + smiles.count('o'),
                'sulfur_count': smiles.count('S') + smiles.count('s'),
                'halogen_count': smiles.count('F') + smiles.count('Cl') + smiles.count('Br') + smiles.count('I'),
                
                # Bond counts
                'single_bonds': smiles.count('-'),
                'double_bonds': smiles.count('='),
                'triple_bonds': smiles.count('#'),
                'aromatic_bonds': smiles.count(':'),
                
                # Ring features
                'num_rings': smiles.count('c') + smiles.count('C1') + smiles.count('C2'),
                'aromatic_atoms': smiles.count('c') + smiles.count('n') + smiles.count('o') + smiles.count('s'),
                'aliphatic_rings': smiles.count('C1') + smiles.count('C2') + smiles.count('C3'),
                
                # Functional groups
                'hydroxyl_groups': smiles.count('OH'),
                'carbonyl_groups': smiles.count('C=O'),
                'carboxyl_groups': smiles.count('COOH'),
                'amino_groups': smiles.count('NH2'),
                'methyl_groups': smiles.count('CH3'),
                'phenyl_groups': smiles.count('c1ccccc1'),
                
                # Complexity measures
                'branch_points': smiles.count('(') + smiles.count('['),
                'complexity_score': len(set(smiles)),
                'hetero_ratio': (smiles.count('N') + smiles.count('O') + smiles.count('S')) / max(len(smiles), 1),
                
                # Charge-related
                'positive_charges': smiles.count('+'),
                'negative_charges': smiles.count('-'),
                
                # Stereochemistry
                'chiral_centers': smiles.count('@'),
                'cis_trans_bonds': smiles.count('/') + smiles.count('\\'),
                
                # Derived features
                'mw_per_atom': (len(smiles) * 10) / max(smiles.count('C') + smiles.count('N') + smiles.count('O') + smiles.count('S'), 1),
                'heavy_atom_ratio': (smiles.count('N') + smiles.count('O') + smiles.count('S')) / max(smiles.count('C'), 1),
                'ring_density': (smiles.count('c') + smiles.count('C1')) / max(len(smiles), 1),
                'saturation_ratio': smiles.count('=') / max(len(smiles), 1),
                'flexibility_score': smiles.count('-') / max(len(smiles), 1),
            }
            
            # Get target value - try each column in order
            target_val = None
            for col in ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']:
                if col in df.columns and col in row:
                    val = row[col]
                    if pd.notna(val):
                        try:
                            val = float(val)
                            if val > 0 and val < 1000000:  # Reasonable range
                                target_val = val
                                break
                        except (ValueError, TypeError):
                            continue
            
            if target_val is not None:
                # Convert to p-scale: p = 9 - log10(value in nM)
                target_val = 9 - np.log10(target_val)
                features_data.append(features)
                target_data.append(target_val)
        
        self.logger.info(f"Processed {len(df)} rows, created {len(features_data)} valid feature sets")
        
        if not features_data:
            self.logger.error("No valid features could be created")
            return None, None
        
        features_df = pd.DataFrame(features_data)
        target_series = pd.Series(target_data)
        
        self.logger.info(f"Created {len(features_df)} samples with {len(features_df.columns)} enhanced features")
        self.logger.info(f"Target range: {target_series.min():.2f} to {target_series.max():.2f}")
        
        return features_df, target_series
    
    def train_models(self, features: pd.DataFrame, target: pd.Series, 
                    models: list = None, test_size: float = 0.2) -> Dict[str, Any]:
        """Train selected models with enhanced hyperparameters"""
        if models is None:
            models = ['RandomForest', 'XGBoost']  # Skip Neural Network for faster training
        
        self.logger.info(f"Training models: {models}")
        
        # Validate data quality for better model performance
        self.logger.info(f"Target statistics: mean={target.mean():.3f}, std={target.std():.3f}")
        self.logger.info(f"Target range: {target.min():.3f} to {target.max():.3f}")
        self.logger.info(f"Enhanced features: {len(features.columns)} total features")
        
        # Check for sufficient variance in target
        if target.std() < 0.01:
            self.logger.warning("Target has very low variance, model performance may be poor")
        
        # Enhanced model training with better hyperparameters
        results = {}
        for model_name in models:
            self.logger.info(f"Training enhanced {model_name}...")
            try:
                if model_name == 'RandomForest':
                    # Enhanced RandomForest training
                    model_result = self.train_enhanced_random_forest(features, target)
                elif model_name == 'XGBoost':
                    model_result = self.model_trainer.train_xgboost(features, target)
                elif model_name == 'NeuralNetwork':
                    model_result = self.model_trainer.train_neural_network(features, target)
                else:
                    self.logger.warning(f"Unknown model: {model_name}")
                    continue
                    
                results[model_name] = model_result
                self.logger.info(f"{model_name} trained successfully")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Extract only serializable metrics for JSON
        metrics_summary = {}
        for model_name, result in results.items():
            metrics_summary[model_name] = {
                'test_metrics': result['test_metrics'],
                'model_file': result['model_file']
            }
        
        metrics_file = results_dir / "model_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        self.logger.info(f"Training results saved to {metrics_file}")
        return results
    
    def train_enhanced_random_forest(self, features: pd.DataFrame, target: pd.Series):
        """Train RandomForest with parameters from unified config"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        from sklearn.preprocessing import RobustScaler
        import joblib
        
        # Get test size from config
        test_size = getattr(DATA_CONFIG, 'test_size', 0.2) if config_manager else 0.2
        random_state = getattr(DATA_CONFIG, 'random_state', 42) if config_manager else 42
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        # Scale features for better performance
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Get RandomForest parameters from config
        if config_manager:
            rf_params = config_manager.get_model_params('RandomForest')
        else:
            # Fallback parameters
            rf_params = {
                'n_estimators': 300,
                'max_depth': 25,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Enhanced RandomForest with parameters from config
        rf = RandomForestRegressor(**rf_params)
        
        # Train the model
        self.logger.info(f"Training enhanced RandomForest with {rf_params['n_estimators']} estimators...")
        rf.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Feature importance logging
        feature_importance = sorted(
            zip(features.columns, rf.feature_importances_), 
            key=lambda x: x[1], reverse=True
        )
        
        self.logger.info("Top 10 important features:")
        for feat, importance in feature_importance[:10]:
            self.logger.info(f"  {feat}: {importance:.4f}")
        
        # Save model using config path
        models_dir = getattr(PATHS_CONFIG, 'models_dir', 'models') if config_manager else 'models'
        model_file = Path(models_dir) / (getattr(MODELS_CONFIG, 'rf_file', 'enhanced_randomforest_model.pkl') if config_manager else 'enhanced_randomforest_model.pkl')
        
        model_data = {
            'model': rf,
            'scaler': scaler,
            'features': features.columns.tolist()
        }
        joblib.dump(model_data, model_file)
        
        result = {
            'model': rf,
            'scaler': scaler,
            'test_metrics': {
                'r2_score': r2,
                'rmse': rmse
            },
            'model_file': str(model_file),
            'feature_importance': dict(feature_importance)
        }
        
        self.logger.info(f"Enhanced RandomForest - R²: {r2:.4f}, RMSE: {rmse:.4f}")
        return result
    
    def create_data_summary(self, datasets_processed: list):
        """Create data summary"""
        self.logger.info("Creating data summary...")
        
        summary = {
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets_processed": datasets_processed,
            "files_created": {}
        }
        
        # Check created files
        processed_dir = Path("data/processed")
        for file_path in processed_dir.glob("*.csv"):
            file_info = {
                "size_mb": file_path.stat().st_size / (1024*1024),
                "records": None
            }
            
            try:
                df = pd.read_csv(file_path, nrows=1)
                total_lines = sum(1 for _ in open(file_path, 'r'))
                file_info["records"] = total_lines - 1
                file_info["columns"] = len(df.columns)
            except:
                pass
            
            summary["files_created"][file_path.name] = file_info
        
        # Save summary
        summary_file = processed_dir / "data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Data summary saved to {summary_file}")
        return summary

def main():
    """Main CLI function"""
    # Get defaults from config if available
    default_sample_size = getattr(DATA_CONFIG, 'sample_size', 5000) if config_manager else 5000
    default_max_rows = getattr(DATA_CONFIG, 'max_rows', 50000) if config_manager else 50000
    default_test_size = getattr(DATA_CONFIG, 'test_size', 0.2) if config_manager else 0.2
    default_data_source = getattr(config.cli, 'default_data_source', 'auto') if config_manager else 'auto'
    default_models = getattr(config.cli, 'default_models', ['RandomForest', 'XGBoost']) if config_manager else ['RandomForest', 'XGBoost']
    default_log_level = getattr(LOGGING_CONFIG, 'level', 'INFO') if config_manager else 'INFO'
    
    parser = argparse.ArgumentParser(
        description="Affinify CLI - Unified data processing and model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with BindingDB
  python affinity_cli.py --download --process --train --models RandomForest XGBoost
  
  # Process BindingDB and train models
  python affinity_cli.py --process --train --data-source bindingdb
  
  # Create sample data and train models
  python affinity_cli.py --process --train --data-source sample --sample-size 5000
  
  # Just train models on existing data
  python affinity_cli.py --train --models RandomForest XGBoost NeuralNetwork
  
  # Process data only
  python affinity_cli.py --process --data-source bindingdb --max-rows 10000
        """
    )
    
    # Main actions
    parser.add_argument('--download', action='store_true', 
                       help='Guide through manual BindingDB download')
    parser.add_argument('--process', action='store_true',
                       help='Process data and extract features')
    parser.add_argument('--train', action='store_true',
                       help='Train machine learning models')
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (download + process + train)')
    
    # Data options
    parser.add_argument('--data-source', choices=['bindingdb', 'sample', 'auto'], 
                       default=default_data_source, help='Data source to use')
    parser.add_argument('--sample-size', type=int, default=default_sample_size,
                       help=f'Size of sample dataset (default: {default_sample_size})')
    parser.add_argument('--max-rows', type=int, default=default_max_rows,
                       help=f'Maximum rows to process from BindingDB (default: {default_max_rows})')
    
    # Model options
    parser.add_argument('--models', nargs='+', 
                       choices=['RandomForest', 'XGBoost', 'NeuralNetwork'],
                       default=default_models,
                       help=f'Models to train (default: {default_models})')
    parser.add_argument('--test-size', type=float, default=default_test_size,
                       help=f'Test set size (0.1-0.5) (default: {default_test_size})')
    
    # Other options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default=default_log_level, help=f'Logging level (default: {default_log_level})')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing even if data exists')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    # Handle --all flag
    if args.all:
        args.download = True
        args.process = True
        args.train = True
    
    # Validate arguments
    if not any([args.download, args.process, args.train]):
        parser.error("Must specify at least one action: --download, --process, --train, or --all")
    
    # Initialize CLI
    cli = AffinifyCLI(logger)
    
    print("=" * 70)
    print("🧬 AFFINIFY CLI - Unified Data Processing & Model Training")
    if config_manager:
        print(f"📁 Configuration loaded from .env")
        print(f"📊 Data directory: {getattr(PATHS_CONFIG, 'data_dir', 'data')}")
        print(f"🤖 Models directory: {getattr(PATHS_CONFIG, 'models_dir', 'models')}")
    else:
        print("⚠️  Using default configuration (no .env found)")
    print("=" * 70)
    
    try:
        # Create directories
        cli.create_directories()
        
        datasets_processed = []
        
        # Step 1: Download guide
        if args.download:
            cli.guide_manual_download()
        
        # Step 2: Process data
        if args.process:
            logger.info("Starting data processing...")
            
            df = None
            dataset_name = "unknown"
            
            # Determine data source
            if args.data_source == 'bindingdb' or args.data_source == 'auto':
                df = cli.process_bindingdb_data(args.max_rows)
                if df is not None:
                    dataset_name = "bindingdb"
                    datasets_processed.append("bindingdb")
                elif args.data_source == 'bindingdb':
                    logger.error("BindingDB data not available. Use --download first or change data source.")
                    return
            
            # Fall back to sample data if needed
            if df is None:
                logger.info("Using sample data...")
                df = cli.create_sample_data(args.sample_size)
                dataset_name = "sample"
                datasets_processed.append("sample")
            
            # Extract features
            features, target = cli.extract_features(df, dataset_name)
            
            if features is None or target is None:
                logger.error("Feature extraction failed!")
                return
            
            logger.info(f"Successfully processed {len(features)} samples")
            
            # Create summary
            summary = cli.create_data_summary(datasets_processed)
        
        # Step 3: Train models
        if args.train:
            logger.info("Starting model training...")
            
            # Load processed data
            processed_dir = Path("data/processed")
            features_file = processed_dir / "processed_features.csv"
            target_file = processed_dir / "target_values.csv"
            
            if not features_file.exists() or not target_file.exists():
                logger.error("Processed data not found. Run --process first.")
                return
            
            features = pd.read_csv(features_file)
            target = pd.read_csv(target_file).iloc[:, 0]  # Convert to Series
            
            # Train models
            results = cli.train_models(features, target, args.models, args.test_size)
            
            # Report results
            logger.info("Training completed!")
            for model_name, result in results.items():
                if 'test_metrics' in result:
                    r2 = result['test_metrics'].get('r2_score', 'N/A')
                    rmse = result['test_metrics'].get('rmse', 'N/A')
                    logger.info(f"{model_name}: R² = {r2}, RMSE = {rmse}")
        
        # Final summary
        print("\n" + "="*70)
        print("🎉 AFFINIFY CLI COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        if args.process:
            print(f"📊 Datasets processed: {', '.join(datasets_processed)}")
            if 'features' in locals():
                print(f"🔬 Features: {len(features)} samples, {len(features.columns)} features")
        
        if args.train:
            print(f"🤖 Models trained: {', '.join(args.models)}")
            if 'results' in locals():
                print(f"📈 Models completed: {list(results.keys())}")
        
        print("\n🚀 Next step: streamlit run app/main.py")
        print("="*70)
        
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        raise

if __name__ == "__main__":
    main()