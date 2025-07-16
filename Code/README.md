# Affinify: AI-Powered Protein-Ligand Binding Affinity Predictor

## Quick Start

### Installation

**Method 1: Manual Installation**
```bash
cd Code
conda create -n affinity_env python=3.9
conda activate affinity_env
pip install -r requirements.txt
# Note: Some packages may fail - that's okay, fallbacks are available
```

**Method 2: Core Dependencies Only**
```bash
pip install scikit-learn pandas numpy matplotlib streamlit plotly
```

**Method 3: Step-by-Step Installation**
```bash
# Install core dependencies
pip install scikit-learn pandas numpy scipy matplotlib seaborn plotly streamlit requests beautifulsoup4 pytest jupyter ipywidgets joblib

# Install optional dependencies (if needed)
pip install xgboost tensorflow torch

# Install molecular libraries (recommended via conda)
conda install -c conda-forge rdkit biopython py3dmol
```

### Running the Application

1. **Download Data**:
   ```bash
   python scripts/download_data.py
   ```

2. **Train Models**:
   ```bash
   python scripts/train_models.py
   ```

3. **Launch Web App**:
   ```bash
   streamlit run app/main.py
   ```

### Troubleshooting

#### Common Issues

1. **RDKit Installation Issues**
   - Use conda: `conda install -c conda-forge rdkit`
   - The project will work without RDKit but with limited molecular features

2. **TensorFlow/PyTorch Issues**
   - These are optional for advanced models
   - Random Forest and XGBoost work without them

3. **Compilation Errors (like hdbscan)**
   - These packages are not essential for the core functionality
   - Skip them and proceed with the installation

4. **General Installation Issues**
   - Use the core dependencies only
   - The project works without RDKit but with limited molecular features

#### System Requirements
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- 2GB free disk space

#### Platform-Specific Notes

**Linux/Ubuntu**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev gcc g++
```

**macOS**
```bash
# Install Xcode command line tools
xcode-select --install
```

**Windows**
- Use Anaconda or Miniconda for easier package management
- Visual Studio Build Tools may be required for some packages

#### Verification

After installation, verify everything works:

```bash
python -c "import pandas, numpy, sklearn, matplotlib, streamlit; print('Core packages OK')"
python scripts/download_data.py
python scripts/train_models.py
streamlit run app/main.py
```

#### Getting Help

If you encounter issues:
1. Check the logs in the `logs/` directory
2. Run the test suite: `python -m pytest tests/`
3. Use the simplified requirements.txt for essential packages only

## Installation Status

The project structure has been successfully created! Here's what we've built:

### ✅ Project Structure Created
```
Code/
├── src/                          # Source code modules
│   ├── data_processing/         # Data collection and feature extraction
│   ├── models/                  # Machine learning models  
│   ├── visualization/           # Molecular visualization tools
│   └── utils/                   # Utility functions
├── app/                         # Streamlit web application
├── scripts/                     # Training and data scripts
├── tests/                       # Unit tests
├── config/                      # Configuration files
├── requirements.txt             # Python dependencies (fixed)
├── setup.py                     # Installation script
├── demo.py                      # Demo script
└── README.md                    # Documentation
```

### 🔧 What to Do Next

1. **Install Core Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Setup Script**:
   ```bash
   python setup.py
   ```

3. **Test the Installation**:
   ```bash
   python demo.py
   ```

4. **Run the Full Pipeline**:
   ```bash
   python scripts/download_data.py
   python scripts/train_models.py
   streamlit run app/main.py
   ```

### 📦 Key Components Built

- **Data Processing**: Handles molecular datasets and feature extraction
- **ML Models**: Random Forest, XGBoost, Neural Networks with ensemble
- **Web App**: Interactive Streamlit interface for predictions
- **Visualization**: 3D molecular visualization and analysis plots
- **Configuration**: Centralized settings management
- **Tests**: Unit tests for all components

### 🎯 Features Implemented

- ✅ Multiple ML algorithms (Random Forest, XGBoost, Neural Networks)
- ✅ Molecular feature extraction (works with/without RDKit)
- ✅ Interactive web interface
- ✅ 3D molecular visualization
- ✅ Batch prediction capabilities
- ✅ Model performance analysis
- ✅ Educational content and documentation

The installation error you encountered was due to the `hdbscan` package compilation issue. I've fixed the requirements.txt to avoid such problems and created a robust setup script that handles optional dependencies gracefully.

Try running the installation now!

## Project Structure

```
Code/
├── data/
│   ├── raw/                 # Original downloaded datasets
│   ├── processed/           # Processed and cleaned data
│   └── external/           # External reference data
├── src/
│   ├── data_processing/    # Data collection and processing
│   ├── models/            # ML model implementations
│   ├── visualization/     # Molecular visualization tools
│   └── utils/            # Utility functions
├── app/                   # Streamlit web application
├── scripts/              # Training and data scripts
├── notebooks/           # Jupyter notebooks for analysis
├── tests/              # Unit tests
└── config/            # Configuration files
```

## Features

- Multiple ML models (Random Forest, XGBoost, Neural Networks)
- 3D molecular visualization
- Interactive web interface
- Batch prediction capabilities
- Real-time binding affinity prediction

## Usage Examples

See `notebooks/` for detailed examples and tutorials.
