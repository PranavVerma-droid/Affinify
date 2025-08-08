# Database configuration
DATABASE_PATH = "data/affinity_database.db"

# Model configuration
MODEL_SAVE_PATH = "models/"
TRAINED_MODELS = {
    'random_forest': 'rf_model.joblib',
    'xgboost': 'xgb_model.joblib',
    'neural_network': 'nn_model.h5'
}

# Data sources
DATA_SOURCES = {
    'chembl_base_url': 'https://www.ebi.ac.uk/chembl/api/data/',
    'bindingdb_url': 'https://www.bindingdb.org/bind/downloads/BindingDB_All_202X.tsv',
    'pdb_base_url': 'https://files.rcsb.org/download/',
}

# Feature engineering
MOLECULAR_DESCRIPTORS = [
    'MolWt', 'LogP', 'HBD', 'HBA', 'TPSA', 'NumRotatableBonds',
    'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
    'FractionCsp3', 'HeavyAtomCount', 'NumHeteroatoms'
]

# Model hyperparameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'random_state': 42
}

XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42
}

NN_PARAMS = {
    'hidden_layers': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}

# Visualization settings
PLOT_STYLE = 'plotly_white'
MOLECULAR_COLORS = {
    'protein': '#2E86AB',
    'ligand': '#A23B72',
    'binding_site': '#F18F01'
}
