# Affinify - AI-Powered Protein-Ligand Binding Affinity Predictor

A unified CLI-based system for predicting protein-ligand binding affinity using machine learning.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
```bash
# Create virtual environment
python -m venv affinity_env
source affinity_env/bin/activate  # On Windows: affinity_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

The project now uses a **single unified CLI** that handles all operations:

```bash
# Full pipeline (recommended for first run)
python scripts/affinity_cli.py --all

# Or step by step:
python scripts/affinity_cli.py --download    # Guide through data download
python scripts/affinity_cli.py --process     # Process data
python scripts/affinity_cli.py --train       # Train models

# Quick demo with sample data
python scripts/affinity_cli.py --process --train --data-source sample --sample-size 5000

# Process BindingDB and train models
python scripts/affinity_cli.py --process --train --data-source bindingdb --max-rows 50000
```

### Run Web Application
```bash
streamlit run app/main.py
```

## 📊 Project Structure

```
Code/
├── scripts/
│   └── affinity_cli.py          # 🆕 Unified CLI (replaces 3 old scripts)
├── src/
│   ├── data_processing/         # Data collection and feature extraction
│   ├── models/                  # ML model implementations
│   └── visualization/           # Plotting and visualization
├── app/
│   └── main.py                  # Streamlit web interface
├── data/
│   ├── raw/bindingdb/          # Raw BindingDB data
│   └── processed/              # Processed features and datasets
├── models/                      # Trained model files
└── results/                     # Training results and metrics
```

## 🧬 Features

- **Unified CLI Interface**: Single script handles all operations
- **BindingDB Integration**: Process real molecular binding data
- **Multiple ML Models**: RandomForest, XGBoost, Neural Networks
- **Interactive Web App**: Streamlit-based interface
- **3D Visualization**: Molecular structure visualization
- **Feature Engineering**: Automated molecular feature extraction

## 📈 Performance

Recent training results on BindingDB data:
- **Dataset**: 42,395 protein-ligand pairs
- **RandomForest**: R² = 0.38, RMSE = 1.13
- **XGBoost**: R² = 0.32, RMSE = 1.18

## 🔧 CLI Options

```bash
# Main actions
--download          # Guide through BindingDB download
--process           # Process data and extract features  
--train             # Train machine learning models
--all               # Run complete pipeline

# Data options
--data-source       # Choose: bindingdb, sample, or auto
--sample-size       # Size of sample dataset (default: 5000)
--max-rows          # Max rows from BindingDB (default: 50000)

# Model options
--models            # Select models: RandomForest, XGBoost, NeuralNetwork
--test-size         # Test set proportion (default: 0.2)

# Other options
--log-level         # Logging level: DEBUG, INFO, WARNING, ERROR
--force-reprocess   # Force reprocessing of existing data
```

## 🎯 Examples

```bash
# Quick demo
python scripts/affinity_cli.py --process --train --data-source sample --sample-size 1000

# Production run
python scripts/affinity_cli.py --all --data-source bindingdb --max-rows 100000

# Train specific models
python scripts/affinity_cli.py --train --models RandomForest XGBoost

# Debug mode
python scripts/affinity_cli.py --process --train --log-level DEBUG
```

## 🧪 Educational Focus

This project demonstrates:
- **Data Science Pipeline**: From raw data to trained models
- **Machine Learning**: Feature engineering, model training, evaluation
- **Software Engineering**: Clean code, CLI design, error handling
- **Computational Biology**: Molecular descriptors, binding affinity
- **Web Development**: Interactive applications with Streamlit

## 🛠️ Development

### Migration from Old Scripts
The project has been simplified:
- ✅ `affinity_cli.py` - New unified interface
- ❌ `download_data.py` - Replaced by `--download` flag
- ❌ `process_and_train.py` - Replaced by `--process --train`
- ❌ `train_models.py` - Replaced by `--train` flag

### Next Steps
1. **Enhanced Features**: Add more sophisticated molecular descriptors
2. **Model Improvement**: Implement graph neural networks
3. **Cloud Deployment**: Deploy to cloud platforms
4. **Database Integration**: Connect to live molecular databases

## 📝 License

This project is for educational purposes. See individual data source licenses for BindingDB and other datasets.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

**Lead Developer**: Pranav Verma  
**School**: Lotus Valley International School  
**Class**: XII Aryabhatta  
**Subject**: Computer Science Project
