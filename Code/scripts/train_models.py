#!/usr/bin/env python3
"""
Model Training Script for Affinify Project
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
import json

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
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_data():
    """Load and prepare training data"""
    logger.info("Loading training data...")
    
    data_dir = Path("data/processed")
    
    # Try to load processed data first
    features_file = data_dir / "processed_features.csv"
    target_file = data_dir / "target_values.csv"
    
    if features_file.exists() and target_file.exists():
        logger.info("Loading preprocessed features and targets...")
        features = pd.read_csv(features_file)
        target = pd.read_csv(target_file)
        
        # Convert target to Series if it's a DataFrame
        if isinstance(target, pd.DataFrame):
            target = target.iloc[:, 0]
        
        logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
        return features, target
    
    # Otherwise, create sample data
    logger.info("Creating sample dataset...")
    data_collector = DataCollector()
    df = data_collector.create_sample_dataset(5000)
    
    # Extract features
    logger.info("Extracting molecular features...")
    feature_extractor = MolecularFeatureExtractor()
    features, target = feature_extractor.prepare_features(df)
    
    # Save processed data
    data_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_file, index=False)
    target.to_csv(target_file, index=False)
    
    logger.info(f"Processed {len(features)} samples with {len(features.columns)} features")
    return features, target

def train_models(features, target):
    """Train multiple models and compare performance"""
    logger.info("Starting model training...")
    
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Train models
    results = trainer.train_models(features, target, test_size=0.2, random_state=42)
    
    # Get best model
    best_name, best_model = trainer.get_best_model()
    
    logger.info(f"Best model: {best_name}")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save performance metrics
    metrics_summary = {}
    for model_name, result in results.items():
        metrics_summary[model_name] = {
            'train_metrics': result.get('train_metrics', {}),
            'test_metrics': result['test_metrics']
        }
    
    metrics_file = results_dir / "model_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    logger.info(f"Model metrics saved to {metrics_file}")
    
    return results, best_model

def main():
    """Main training function"""
    logger.info("Starting Affinify model training pipeline...")
    
    try:
        # Load data
        features, target = load_data()
        
        # Train models
        results, best_model = train_models(features, target)
        
        logger.info("Model training completed successfully!")
        
        # Display results
        for model_name, result in results.items():
            test_r2 = result['test_metrics']['r2_score']
            test_rmse = result['test_metrics']['rmse']
            logger.info(f"{model_name}: R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
