# Affinify CLI - Unified Interface

The Affinify project has been simplified to use a single CLI script that handles all operations:

## Quick Start

### Basic Usage
```bash
# Full pipeline (download guide + process + train)
python scripts/affinity_cli.py --all

# Process BindingDB data and train models
python scripts/affinity_cli.py --process --train --data-source bindingdb

# Create sample data and train models (fastest option)
python scripts/affinity_cli.py --process --train --data-source sample --sample-size 5000

# Just train models on existing processed data
python scripts/affinity_cli.py --train --models RandomForest XGBoost
```

### Step-by-Step Process

#### 1. Download Data (Optional)
```bash
# Guide through manual BindingDB download
python scripts/affinity_cli.py --download
```

#### 2. Process Data
```bash
# Process BindingDB data (if available)
python scripts/affinity_cli.py --process --data-source bindingdb

# Or create sample data for testing
python scripts/affinity_cli.py --process --data-source sample --sample-size 5000
```

#### 3. Train Models
```bash
# Train all models
python scripts/affinity_cli.py --train --models RandomForest XGBoost NeuralNetwork

# Train specific models only
python scripts/affinity_cli.py --train --models RandomForest
```

### Advanced Options

#### Data Processing
- `--data-source`: Choose between 'bindingdb', 'sample', or 'auto'
- `--sample-size`: Size of sample dataset (default: 5000)
- `--max-rows`: Maximum rows to process from BindingDB (default: 50000)
- `--force-reprocess`: Force reprocessing even if data exists

#### Model Training
- `--models`: Select specific models (RandomForest, XGBoost, NeuralNetwork)
- `--test-size`: Test set proportion (default: 0.2)

#### Other Options
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### Examples

```bash
# Quick demo with sample data
python scripts/affinity_cli.py --process --train --data-source sample --sample-size 1000

# Production run with BindingDB
python scripts/affinity_cli.py --all --data-source bindingdb --max-rows 100000

# Fast training with only Random Forest
python scripts/affinity_cli.py --train --models RandomForest

# Debug mode
python scripts/affinity_cli.py --process --train --log-level DEBUG
```

## Output Files

After running the CLI, you'll find:
- `data/processed/`: Processed datasets and features
- `models/`: Trained model files
- `results/`: Performance metrics and results
- `logs/`: Execution logs

## Run the Web App

After processing data and training models:
```bash
streamlit run app/main.py
```

## Migration from Old Scripts

The old scripts have been replaced:
- `download_data.py` → `affinity_cli.py --download --process`
- `process_and_train.py` → `affinity_cli.py --process --train`
- `train_models.py` → `affinity_cli.py --train`

The new CLI provides all functionality in a single, well-organized interface.