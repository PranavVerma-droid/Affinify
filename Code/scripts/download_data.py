#!/usr/bin/env python3
"""
Data Download and Processing Script for Affinify Project
Interactive script that guides user through manual download and then processes data automatically
"""

import os
import sys
import logging
import pandas as pd #type: ignore
from pathlib import Path
import json
import time
import zipfile

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.data_collector import DataCollector #type: ignore
from data_processing.feature_extractor import MolecularFeatureExtractor #type: ignore
 
# Configure logging
os.makedirs('logs', exist_ok=True)

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_download.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Test logging
logger.info("Data download script started")

def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("🧬 AFFINIFY DATA DOWNLOAD & PROCESSING SCRIPT")
    print("=" * 70)
    print()

def create_directories():
    """Create necessary directories"""
    logger.info("Creating data directories...")
    
    directories = [
        "data/raw/bindingdb",
        "data/processed",
        "data/external"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    print("✅ Data directories created successfully!")

def guide_manual_download():
    """Guide user through manual download process"""
    print("\n" + "="*50)
    print("📥 MANUAL DOWNLOAD REQUIRED")
    print("="*50)
    print()
    print("We need to download the BindingDB dataset manually.")
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

def check_downloaded_file():
    """Check if the required file exists"""
    logger.info("Checking for downloaded BindingDB file...")
    
    # Look for the TSV file
    bindingdb_dir = Path("data/raw/bindingdb")
    
    # Check for various possible file names
    possible_files = [
        "BindingDB_All_202507.tsv",
        "BindingDB_All_202507_tsv.tsv",
        "BindingDB_All.tsv"
    ]
    
    found_file = None
    for filename in possible_files:
        file_path = bindingdb_dir / filename
        if file_path.exists():
            found_file = file_path
            break
    
    # If not found, look for any TSV file
    if not found_file:
        tsv_files = list(bindingdb_dir.glob("*.tsv"))
        if tsv_files:
            found_file = tsv_files[0]
    
    if found_file:
        file_size = found_file.stat().st_size / (1024*1024)  # Size in MB
        logger.info(f"Found BindingDB file: {found_file.name} ({file_size:.1f} MB)")
        
        if file_size < 100:  # Less than 100MB is suspicious
            print(f"⚠️  Warning: File size is {file_size:.1f}MB, which seems small.")
            print("   Expected size is around 1000-1500MB for the full dataset.")
            response = input("Continue anyway? (y/n): ").lower()
            if response != 'y':
                return None
        
        return found_file
    else:
        print("❌ BindingDB TSV file not found!")
        print("Please ensure you have:")
        print("1. Downloaded BindingDB_All_202507_tsv.zip")
        print("2. Extracted it to data/raw/bindingdb/")
        print("3. The TSV file is present in the directory")
        return None

def process_bindingdb_file(file_path):
    """Process the BindingDB file"""
    logger.info(f"Processing BindingDB file: {file_path}")
    
    try:
        # Read the file in chunks to handle large size
        logger.info("Reading BindingDB file (this may take a few minutes)...")
        
        # First, peek at the file to understand its structure
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline().strip()
            sample_line = f.readline().strip()
        
        num_columns = len(header.split('\t'))
        print(f"📄 File contains {num_columns} columns")
        columns_sample = header.split('\t')[:5]
        print(f"📝 Sample columns: {columns_sample}...")
        
        # Define columns we need
        useful_columns = [
            'Ligand SMILES', 'PDB ID(s)', 'UniProt (SwissProt) Primary ID of Target Chain',
            'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)',
            'Target Name', 'Target Source Organism According to Curator or DataSource',
            'Ligand InChI Key'
        ]
        
        # Read file in chunks
        chunk_size = 10000
        chunks = []
        chunk_count = 0
        max_chunks = 500  # Limit to ~5M rows for memory management
        
        logger.info("Reading file in chunks...")
        
        for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size, 
                                low_memory=False, encoding='utf-8', errors='ignore'):
            
            # Filter for useful columns that exist
            available_columns = [col for col in useful_columns if col in chunk.columns]
            if available_columns:
                chunk = chunk[available_columns]
            
            chunks.append(chunk)
            chunk_count += 1
            
            if chunk_count % 50 == 0:
                print(f"  Processed {chunk_count * chunk_size:,} rows...")
            
            if chunk_count >= max_chunks:
                logger.warning(f"Limiting to {max_chunks * chunk_size:,} rows for memory management")
                break
        
        # Combine chunks
        logger.info("Combining chunks...")
        df = pd.concat(chunks, ignore_index=True)
        
        logger.info(f"Loaded {len(df):,} rows from BindingDB")
        
        # Basic cleaning
        logger.info("Cleaning data...")
        
        # Remove rows with missing SMILES
        if 'Ligand SMILES' in df.columns:
            initial_rows = len(df)
            df = df.dropna(subset=['Ligand SMILES'])
            logger.info(f"Removed {initial_rows - len(df):,} rows with missing SMILES")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df):,} duplicate rows")
        
        # Save subset for processing
        processed_dir = Path("data/processed")
        processed_dir.mkdir(exist_ok=True)
        
        # Save a manageable subset
        subset_size = min(50000, len(df))
        df_subset = df.sample(n=subset_size, random_state=42)
        
        bindingdb_file = processed_dir / "bindingdb_subset.csv"
        df_subset.to_csv(bindingdb_file, index=False)
        
        logger.info(f"Saved BindingDB subset ({subset_size:,} rows) to {bindingdb_file}")
        
        return df_subset
        
    except Exception as e:
        logger.error(f"Error processing BindingDB file: {str(e)}")
        return None

def create_sample_dataset():
    """Create sample dataset for testing"""
    logger.info("Creating sample dataset for testing...")
    
    data_collector = DataCollector()
    sample_df = data_collector.create_sample_dataset(5000)
    
    # Save sample dataset
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    sample_file = processed_dir / "sample_dataset.csv"
    sample_df.to_csv(sample_file, index=False)
    
    logger.info(f"Sample dataset saved to {sample_file}")
    return sample_df

def process_features(df, dataset_name):
    """Process molecular features from dataset"""
    logger.info(f"Processing molecular features for {dataset_name}...")
    
    feature_extractor = MolecularFeatureExtractor()
    
    try:
        features, target = feature_extractor.prepare_features(df)
        
        if len(features) == 0:
            logger.warning(f"No valid features extracted from {dataset_name}")
            return None, None
        
        # Save processed features
        processed_dir = Path("data/processed")
        
        features_file = processed_dir / f"{dataset_name}_features.csv"
        target_file = processed_dir / f"{dataset_name}_target.csv"
        
        features.to_csv(features_file, index=False)
        target.to_csv(target_file, index=False)
        
        logger.info(f"Processed features saved to {features_file}")
        logger.info(f"Target values saved to {target_file}")
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Target shape: {target.shape}")
        
        return features, target
        
    except Exception as e:
        logger.error(f"Error processing features for {dataset_name}: {str(e)}")
        return None, None

def create_data_summary(datasets_processed):
    """Create data summary"""
    logger.info("Creating data summary...")
    
    summary = {
        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets_processed": datasets_processed,
        "files_created": {}
    }
    
    # Check what files were created
    processed_dir = Path("data/processed")
    for file_path in processed_dir.glob("*.csv"):
        file_info = {
            "size_mb": file_path.stat().st_size / (1024*1024),
            "records": None
        }
        
        try:
            # Try to count records
            df = pd.read_csv(file_path, nrows=1)
            total_lines = sum(1 for _ in open(file_path, 'r'))
            file_info["records"] = total_lines - 1  # Subtract header
            file_info["columns"] = len(df.columns)
        except:
            pass
        
        summary["files_created"][file_path.name] = file_info
    
    # Save summary
    summary_file = processed_dir / "data_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Data summary saved to {summary_file}")
    return summary

def main():
    """Main data processing function"""
    print_banner()
    
    try:
        # Create directories
        create_directories()
        
        # Guide user through manual download
        guide_manual_download()
        
        # Check if file was downloaded
        bindingdb_file = check_downloaded_file()
        
        datasets_processed = []
        
        if bindingdb_file:
            logger.info("Processing BindingDB dataset...")
            bindingdb_df = process_bindingdb_file(bindingdb_file)
            
            if bindingdb_df is not None:
                # Process features
                features, target = process_features(bindingdb_df, "bindingdb")
                if features is not None:
                    datasets_processed.append("bindingdb")
                    print("✅ BindingDB dataset processed successfully!")
                else:
                    print("⚠️  BindingDB feature extraction failed")
            else:
                print("⚠️  BindingDB processing failed")
        
        # Always create sample dataset
        logger.info("Creating sample dataset...")
        sample_df = create_sample_dataset()
        features, target = process_features(sample_df, "sample")
        
        if features is not None:
            datasets_processed.append("sample")
            print("✅ Sample dataset created successfully!")
        
        # Create overall processed dataset (use BindingDB if available, otherwise sample)
        if "bindingdb" in datasets_processed:
            logger.info("Creating main processed dataset from BindingDB...")
            # Copy BindingDB features as main dataset
            import shutil
            shutil.copy2("data/processed/bindingdb_features.csv", "data/processed/processed_features.csv")
            shutil.copy2("data/processed/bindingdb_target.csv", "data/processed/target_values.csv")
        else:
            logger.info("Creating main processed dataset from sample...")
            # Copy sample features as main dataset
            import shutil
            shutil.copy2("data/processed/sample_features.csv", "data/processed/processed_features.csv")
            shutil.copy2("data/processed/sample_target.csv", "data/processed/target_values.csv")
        
        # Create summary
        summary = create_data_summary(datasets_processed)
        
        # Final report
        print("\n" + "="*50)
        print("🎉 DATA PROCESSING COMPLETE!")
        print("="*50)
        print(f"📊 Datasets processed: {', '.join(datasets_processed)}")
        print(f"📁 Files created in data/processed/:")
        
        for filename, info in summary["files_created"].items():
            size_mb = info["size_mb"]
            records = info.get("records", "Unknown")
            print(f"  - {filename}: {size_mb:.1f}MB, {records} records")
        
        print("\n🚀 Next steps:")
        print("1. Run: python scripts/train_models.py")
        print("2. Run: streamlit run app/main.py")
        print("\n✅ Data download and processing completed successfully!")
        
        # Flush all handlers to ensure logs are written
        for handler in logging.root.handlers:
            handler.flush()
        
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        print("Check logs/data_download.log for details")
        
        # Flush all handlers to ensure logs are written
        for handler in logging.root.handlers:
            handler.flush()
        raise

if __name__ == "__main__":
    main()