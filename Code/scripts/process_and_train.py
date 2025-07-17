#!/usr/bin/env python3
"""
Combined BindingDB Processing and Model Training Script
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.data_collector import DataCollector
from data_processing.feature_extractor import MolecularFeatureExtractor
from models.ml_models import ModelTrainer

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/process_and_train.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def process_bindingdb_data():
    """Process BindingDB data if available"""
    logger.info("Checking for BindingDB data...")
    
    bindingdb_dir = Path("data/raw/bindingdb")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    bindingdb_subset_file = processed_dir / "bindingdb_subset.csv"
    if bindingdb_subset_file.exists():
        logger.info("BindingDB subset already exists, loading...")
        return pd.read_csv(bindingdb_subset_file)
    
    # Look for TSV files
    tsv_files = list(bindingdb_dir.glob("*.tsv"))
    if not tsv_files:
        logger.warning("No BindingDB TSV files found. Run download_data.py first.")
        return None
    
    tsv_file = tsv_files[0]
    logger.info(f"Processing BindingDB file: {tsv_file}")
    
    try:
        # Process in chunks
        chunk_size = 10000
        chunks = []
        max_chunks = 100  # Limit for faster processing
        
        logger.info("Reading BindingDB file in chunks...")
        
        for i, chunk in enumerate(pd.read_csv(tsv_file, sep='\t', chunksize=chunk_size, 
                                           low_memory=False, encoding='utf-8')):
            
            # Filter for useful columns that exist
            useful_columns = [
                'Ligand SMILES', 'PDB ID(s)', 'Ki (nM)', 'IC50 (nM)', 
                'Kd (nM)', 'EC50 (nM)', 'Target Name'
            ]
            
            available_columns = [col for col in useful_columns if col in chunk.columns]
            if available_columns:
                chunk = chunk[available_columns]
                
                # Basic filtering - remove rows with missing SMILES
                if 'Ligand SMILES' in chunk.columns:
                    initial_len = len(chunk)
                    chunk = chunk.dropna(subset=['Ligand SMILES'])
                    if i == 0:  # Log for first chunk only
                        logger.info(f"Removed {initial_len - len(chunk)} rows with missing SMILES from first chunk")
                
                # Less aggressive filtering - just check if we have any useful data
                if len(chunk) > 0:
                    chunks.append(chunk)
                    
            if i >= max_chunks:
                break
                
            if i % 10 == 0:
                logger.info(f"Processed {i * chunk_size:,} rows...")
        
        # Combine chunks
        if not chunks:
            logger.warning("No valid chunks found after filtering")
            return None
            
        logger.info(f"Found {len(chunks)} valid chunks to combine")
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Combined {len(df)} rows from BindingDB")
        
        # Basic cleaning
        if 'Ligand SMILES' in df.columns:
            df = df.dropna(subset=['Ligand SMILES'])
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Additional cleaning for better model performance
        # Remove very short SMILES (likely invalid)
        if 'Ligand SMILES' in df.columns:
            df = df[df['Ligand SMILES'].str.len() >= 3]
        
        # Remove rows with invalid binding affinity values
        affinity_cols = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
        for col in affinity_cols:
            if col in df.columns:
                # Convert to numeric, handling non-numeric values
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Remove rows where this column has invalid values (but keep NaN)
                df = df[(df[col].isna()) | ((df[col] > 0) & (df[col] < 1000000))]
        
        # Save subset
        subset_size = min(10000, len(df))
        if len(df) > 0:
            df_subset = df.sample(n=subset_size, random_state=42)
        else:
            logger.warning("No valid data remaining after cleaning")
            return None
            
        df_subset.to_csv(bindingdb_subset_file, index=False)
        
        logger.info(f"Saved BindingDB subset with {len(df_subset)} rows")
        return df_subset
        
    except Exception as e:
        logger.error(f"Error processing BindingDB: {e}")
        return None

def extract_features(df, dataset_name="bindingdb"):
    """Extract features from dataset"""
    logger.info(f"Extracting features from {dataset_name}...")
    logger.info(f"Input dataframe shape: {df.shape}")
    logger.info(f"Input dataframe columns: {df.columns.tolist()}")
    
    feature_extractor = MolecularFeatureExtractor()
    
    # Add debug logging
    logger.info("Calling prepare_features...")
    try:
        result = feature_extractor.prepare_features(df)
        logger.info(f"prepare_features returned: {type(result)}")
    except Exception as e:
        logger.error(f"prepare_features threw an exception: {e}")
        result = None
    
    if result is None:
        logger.error("Feature extraction failed - prepare_features returned None")
        logger.error("This might be due to missing required columns or invalid data")
        
        # Try to debug what's in the dataframe
        logger.info("Attempting to debug dataframe content...")
        if 'Ligand SMILES' in df.columns:
            logger.info(f"SMILES column found with {df['Ligand SMILES'].notna().sum()} non-null values")
        
        # Check for binding affinity columns
        affinity_cols = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
        found_affinity_cols = [col for col in affinity_cols if col in df.columns]
        logger.info(f"Found affinity columns: {found_affinity_cols}")
        
        for col in found_affinity_cols:
            non_null_count = df[col].notna().sum()
            logger.info(f"{col}: {non_null_count} non-null values")
        
        # Try fallback feature extraction
        logger.info("Attempting fallback feature extraction...")
        features, target = create_simple_features(df)
        
        if features is None or target is None:
            logger.error("Even fallback feature extraction failed!")
            return None, None
        
        logger.info("Fallback feature extraction successful!")
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
    
    logger.info(f"Extracted {len(features)} samples with {len(features.columns)} features")
    return features, target

def create_simple_features(df):
    """Create simple features as a fallback"""
    logger.info("Creating simple features as fallback...")
    
    # Basic features from SMILES
    features_data = []
    target_data = []
    
    for idx, row in df.iterrows():
        smiles = row.get('Ligand SMILES', '')
        if not smiles or pd.isna(smiles) or len(str(smiles)) < 3:
            continue
            
        smiles = str(smiles)
        
        # Simple molecular descriptors
        features = {
            'mol_weight': len(smiles) * 10,  # Rough approximation
            'num_atoms': smiles.count('C') + smiles.count('N') + smiles.count('O'),
            'num_bonds': smiles.count('=') + smiles.count('#'),
            'num_rings': smiles.count('c'),
            'hetero_atoms': smiles.count('N') + smiles.count('O') + smiles.count('S'),
            'aromatic_atoms': smiles.count('c') + smiles.count('n'),
            'smiles_length': len(smiles),
            'carbon_count': smiles.count('C') + smiles.count('c'),
            'nitrogen_count': smiles.count('N') + smiles.count('n'),
            'oxygen_count': smiles.count('O') + smiles.count('o'),
            'sulfur_count': smiles.count('S') + smiles.count('s'),
            'halogen_count': smiles.count('F') + smiles.count('Cl') + smiles.count('Br') + smiles.count('I'),
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
            # Convert to pKd/pKi (negative log of molar concentration)
            target_val = -np.log10(target_val * 1e-9)
            features_data.append(features)
            target_data.append(target_val)
    
    logger.info(f"Processed {len(df)} rows, created {len(features_data)} valid feature sets")
    
    if not features_data:
        logger.error("No valid features could be created")
        return None, None
    
    features_df = pd.DataFrame(features_data)
    target_series = pd.Series(target_data)
    
    logger.info(f"Created {len(features_df)} samples with {len(features_df.columns)} features")
    logger.info(f"Target range: {target_series.min():.2f} to {target_series.max():.2f}")
    
    return features_df, target_series

def train_selected_models(features, target, models_to_train=None):
    """Train only selected models"""
    if models_to_train is None:
        models_to_train = ['RandomForest', 'XGBoost']  # Skip Neural Network for faster training
    
    logger.info(f"Training models: {models_to_train}")
    
    trainer = ModelTrainer()
    
    # Train only selected models
    results = {}
    for model_name in models_to_train:
        logger.info(f"Training {model_name}...")
        try:
            if model_name == 'RandomForest':
                model_result = trainer.train_random_forest(features, target)
            elif model_name == 'XGBoost':
                model_result = trainer.train_xgboost(features, target)
            elif model_name == 'NeuralNetwork':
                model_result = trainer.train_neural_network(features, target)
            else:
                logger.warning(f"Unknown model: {model_name}")
                continue
                
            results[model_name] = model_result
            logger.info(f"{model_name} trained successfully")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
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
    
    logger.info(f"Training results saved to {metrics_file}")
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process BindingDB and train models")
    parser.add_argument("--models", nargs="+", default=["RandomForest", "XGBoost"],
                       help="Models to train (RandomForest, XGBoost, NeuralNetwork)")
    parser.add_argument("--use-sample", action="store_true",
                       help="Use sample data instead of BindingDB")
    args = parser.parse_args()
    
    logger.info("Starting BindingDB processing and model training...")
    
    try:
        # Step 1: Get data
        if args.use_sample:
            logger.info("Creating sample dataset...")
            data_collector = DataCollector()
            df = data_collector.create_sample_dataset(5000)
            dataset_name = "sample"
        else:
            df = process_bindingdb_data()
            dataset_name = "bindingdb"
            
            if df is None:
                logger.info("BindingDB not available, using sample data...")
                data_collector = DataCollector()
                df = data_collector.create_sample_dataset(5000)
                dataset_name = "sample"
        
        # Step 2: Extract features
        features, target = extract_features(df, dataset_name)
        
        if features is None or target is None:
            logger.error("Feature extraction failed!")
            return
        
        if len(features) == 0:
            logger.error("No features extracted!")
            return
        
        # Validate data quality for better model performance
        logger.info(f"Target statistics: mean={target.mean():.3f}, std={target.std():.3f}")
        logger.info(f"Target range: {target.min():.3f} to {target.max():.3f}")
        
        # Check for sufficient variance in target
        if target.std() < 0.01:
            logger.warning("Target has very low variance, model performance may be poor")
        
        # Step 3: Train models
        results = train_selected_models(features, target, args.models)
        
        # Step 4: Report results
        logger.info("Training completed!")
        for model_name, result in results.items():
            if 'test_metrics' in result:
                r2 = result['test_metrics'].get('r2_score', 'N/A')
                rmse = result['test_metrics'].get('rmse', 'N/A')
                logger.info(f"{model_name}: R² = {r2}, RMSE = {rmse}")
        
        print("\n" + "="*50)
        print("🎉 PROCESSING AND TRAINING COMPLETE!")
        print("="*50)
        print(f"📊 Dataset: {dataset_name}")
        print(f"🔬 Features: {len(features)} samples, {len(features.columns)} features")
        print(f"🤖 Models trained: {list(results.keys())}")
        print("\n🚀 Next step: streamlit run app/main.py")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()