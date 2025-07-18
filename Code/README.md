# Affinify - AI-Powered Protein-Ligand Binding Affinity Predictor


## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Installation
```bash
cd Code
conda create -n affinity_env python=3.9
conda activate affinity_env
pip install -r requirements.txt
# Note: Some packages may fail - that's okay, fallbacks are available
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

## 🤖 Ollama AI Integration

Affinify includes an integrated AI assistant powered by Ollama for enhanced user support and molecular modeling guidance.

### Prerequisites for Ollama
- **Ollama installed**: Download from [ollama.com](https://ollama.com)
- **Python 3.9+**: Same as main project requirements
- **At least 4GB RAM**: For model inference

### Installation & Setup

1. **Install Ollama**:
   ```bash
   # Linux/macOS
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows: Download from ollama.com
   ```

2. **Start Ollama Service**:
   ```bash
   ollama serve
   ```

3. **Pull the Fine-tuned Affinify Model**:
   ```bash
   ollama pull pranavverma/Affinify-AI:latest
   ```

### Fine-tuned Model Details

The **Affinify-AI** model is a custom fine-tuned version specifically trained for this project.

**Model Repository**: [ollama.com/pranavverma/Affinify-AI](https://ollama.com/pranavverma/Affinify-AI)

**Modelfile**: [Modelfile](ollama-model-finetuning/Modelfile)



### Configuration

Enable/disable Ollama integration in `config/config.json`:

```json
{
  "ollama": {
    "enabled": true,                              // Set to false to disable
    "host": "http://localhost:11434",             // Ollama server URL
    "model": "pranavverma/Affinify-AI:latest",   // Model to use
    "temperature": 0.7,                          // Response creativity (0.0-1.0)
    "max_tokens": 1000,                          // Max response length
    "timeout": 30                                // Request timeout (seconds)
  }
}
```

### Using Different Models

To use a different Ollama model:

1. **Pull the model**:
   ```bash
   ollama pull llama2:7b
   # or
   ollama pull mistral:7b
   ```

2. **Update config**:
   ```json
   {
     "ollama": {
       "model": "llama2:7b"  // Change to your preferred model
     }
   }
   ```

### Usage in Web Application

The AI assistant is integrated into the Streamlit web interface:

1. **Start the application**:
   ```bash
   streamlit run app/main.py
   ```

2. **Access the AI Assistant**:
   - Look for the "🤖 Affinify Assistant" section
   - Ask questions about:
     - Protein-ligand binding concepts
     - Machine learning model interpretations
     - Data processing workflows
     - Troubleshooting guidance

### Troubleshooting

**Common Issues**:

1. **"AI model not available"**:
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if needed
   ollama serve
   ```

2. **Model not found**:
   ```bash
   # Pull the model
   ollama pull pranavverma/Affinify-AI:latest
   ```

3. **Connection timeout**:
   - Increase timeout in config.json
   - Check firewall settings
   - Verify Ollama host URL

**Disable Ollama**:
Set `"enabled": false` in config.json if you prefer to use the application without AI assistance.

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

## 📝 License

This project is for educational purposes. See individual data source licenses for BindingDB and other datasets. [LICENSE](../LICENSE)


## 📧 Contact

**Lead Developer**: Pranav Verma  
**School**: Lotus Valley International School  
