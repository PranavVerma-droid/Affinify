#!/usr/bin/env python3
"""
Unified Configuration Manager for Affinify
Loads all settings from a single .env file and provides easy access to configuration values.
This replaces all other config files and serves as the single source of truth.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

# Try to import python-dotenv, create fallback if not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not found. Using basic env loading.")

@dataclass
class ProjectConfig:
    """Project information configuration"""
    name: str = "Affinify"
    version: str = "1.0.0"
    description: str = "AI-Powered Protein-Ligand Binding Affinity Predictor"
    authors: List[str] = field(default_factory=lambda: ["Pranav Verma", "Ekaksh Goyal"])

@dataclass
class PathsConfig:
    """File paths and directories configuration"""
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"
    config_dir: str = "config"
    videos_dir: str = "videos"
    database_path: str = "data/affinity_database.db"

@dataclass
class DataConfig:
    """Data processing configuration"""
    sample_size: int = 5000
    test_size: float = 0.2
    random_state: int = 42
    min_binding_affinity: float = 0.1
    max_binding_affinity: float = 100000.0
    max_rows: int = 50000
    
    # Data sources
    bindingdb_url: str = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202507_tsv.zip"
    chembl_base_url: str = "https://www.ebi.ac.uk/chembl/api/data/"
    pdb_base_url: str = "https://files.rcsb.org/download/"
    
    bindingdb_enabled: bool = True
    sample_enabled: bool = True

@dataclass
class RandomForestConfig:
    """Random Forest model configuration"""
    enabled: bool = True
    n_estimators: int = 300
    max_depth: Optional[int] = 25
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    bootstrap: bool = True
    random_state: int = 42
    n_jobs: int = -1

@dataclass
class XGBoostConfig:
    """XGBoost model configuration"""
    enabled: bool = True
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42

@dataclass
class NeuralNetworkConfig:
    """Neural Network model configuration"""
    enabled: bool = True
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    random_state: int = 42

@dataclass
class ModelsConfig:
    """Machine learning models configuration"""
    rf: RandomForestConfig = field(default_factory=RandomForestConfig)
    xgb: XGBoostConfig = field(default_factory=XGBoostConfig)
    nn: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)
    
    # Model files
    rf_file: str = "enhanced_randomforest_model.pkl"
    xgb_file: str = "xgboost_model.pkl"
    nn_file: str = "nn_model.h5"

@dataclass
class FeaturesConfig:
    """Feature extraction configuration"""
    # Molecular descriptors
    molecular_basic: bool = True
    molecular_topological: bool = True
    molecular_electronic: bool = True
    molecular_geometric: bool = True
    
    # Protein features
    protein_sequence: bool = True
    protein_structural: bool = True
    protein_binding_site: bool = True
    
    # Interaction features
    interaction_molecular_protein: bool = True
    interaction_binding_mode: bool = True
    
    # Standard molecular descriptors
    molecular_descriptors: List[str] = field(default_factory=lambda: [
        'MolWt', 'LogP', 'HBD', 'HBA', 'TPSA', 'NumRotatableBonds',
        'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
        'FractionCsp3', 'HeavyAtomCount', 'NumHeteroatoms'
    ])

@dataclass
class VisualizationConfig:
    """Visualization and animation configuration"""
    plot_style: str = "default"
    figure_width: int = 10
    figure_height: int = 8
    dpi: int = 300
    save_format: str = "png"
    
    # Molecular viewer
    molecular_width: int = 400
    molecular_height: int = 400
    molecular_style: str = "stick"
    
    # Animation settings
    fps: int = 15
    duration: int = 8
    output_dir: str = "videos"
    
    # Colors
    color_protein: str = "#2E86AB"
    color_ligand: str = "#A23B72"
    color_binding_site: str = "#F18F01"
    
    # Atom colors
    atom_colors: Dict[str, str] = field(default_factory=lambda: {
        'C': '#606060',
        'O': '#FF0D0D',
        'N': '#0000FF',
        'H': '#CCCCCC',
        'S': '#FFFF30',
        'P': '#FF8000',
        'F': '#90E050',
        'Cl': '#1FF01F',
        'Br': '#A62929',
        'I': '#940094',
        'DEFAULT': '#FFC0CB'
    })

@dataclass
class PerformanceConfig:
    """Performance targets configuration"""
    target_r2: float = 0.7
    target_rmse: float = 1.0
    max_prediction_time: float = 5.0
    cv_folds: int = 5

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/affinify.log"

@dataclass
class StreamlitConfig:
    """Streamlit application configuration"""
    page_title: str = "Affinify - AI-Powered Protein-Ligand Binding Affinity Predictor"
    page_icon: str = "🧬"
    layout: str = "wide"
    sidebar_state: str = "expanded"

@dataclass
class OllamaConfig:
    """Ollama AI integration configuration"""
    enabled: bool = True
    host: str = "http://localhost:11434"
    model: str = "pranavverma/Affinify-AI:3b"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    chat_title: str = "🤖 Affinify Assistant"
    welcome_message: str = "Hello! I'm your Affinify Assistant. I can help you with questions about protein-ligand binding, machine learning models, data processing, and using this platform. What would you like to know?"
    error_message: str = "I'm having trouble connecting to the AI model. Please check if Ollama is running and try again."
    system_prompt: str = "You are Affinify Assistant, an AI helper for the Affinify protein-ligand binding affinity prediction platform. You help users with questions about molecular modeling, machine learning, data processing, and using the Affinify system. Be helpful, concise, and scientifically accurate."

@dataclass
class CLIConfig:
    """Command line interface defaults"""
    default_models: List[str] = field(default_factory=lambda: ["RandomForest", "XGBoost"])
    default_data_source: str = "auto"
    force_reprocess: bool = False

@dataclass
class AffinifyConfig:
    """Main configuration class containing all settings"""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)

class ConfigManager:
    """Manages loading and accessing configuration from .env file"""
    
    def __init__(self, env_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            env_path: Path to .env file. If None, looks for .env in current directory.
        """
        self.env_path = Path(env_path) if env_path else Path(".env")
        self.config = AffinifyConfig()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from .env file"""
        # Create .env from .env.example if it doesn't exist
        if not self.env_path.exists():
            example_path = self.env_path.with_suffix('.env.example')
            if example_path.exists():
                print(f"⚠️  Creating {self.env_path} from {example_path}")
                print("   Please restart the application after reviewing the configuration.")
                import shutil
                try:
                    shutil.copy2(example_path, self.env_path)
                    print(f"✅ Created {self.env_path} successfully!")
                except Exception as e:
                    print(f"❌ Error creating .env file: {e}")
                    print("   Please copy .env.example to .env manually")
                    raise RuntimeError(f"Could not create .env file: {e}")
            else:
                error_msg = f"""
Configuration Error: Missing .env file!

Expected location: {self.env_path.absolute()}
Template file: {example_path.absolute()} (not found)

Please ensure you have the complete project files including .env.example
"""
                print(error_msg)
                raise FileNotFoundError(error_msg)
        
        # Load environment variables
        if DOTENV_AVAILABLE:
            load_dotenv(self.env_path)
        else:
            self._load_env_manual()
        
        # Load all configuration sections
        self._load_project_config()
        self._load_paths_config()
        self._load_data_config()
        self._load_models_config()
        self._load_features_config()
        self._load_visualization_config()
        self._load_performance_config()
        self._load_logging_config()
        self._load_streamlit_config()
        self._load_ollama_config()
        self._load_cli_config()
    
    def _load_env_manual(self):
        """Manual .env loading when python-dotenv is not available"""
        try:
            with open(self.env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('"').strip("'")
                        os.environ[key] = value
        except Exception as e:
            print(f"Error loading .env manually: {e}")
    
    def _get_env_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _get_env_int(self, key: str, default: int = 0) -> int:
        """Get integer value from environment variable"""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def _get_env_float(self, key: str, default: float = 0.0) -> float:
        """Get float value from environment variable"""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def _get_env_list(self, key: str, default: List[str] = None, separator: str = ',') -> List[str]:
        """Get list value from environment variable"""
        if default is None:
            default = []
        value = os.getenv(key, '')
        if not value:
            return default
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def _get_env_int_list(self, key: str, default: List[int] = None, separator: str = ',') -> List[int]:
        """Get list of integers from environment variable"""
        if default is None:
            default = []
        str_list = self._get_env_list(key, [], separator)
        try:
            return [int(item) for item in str_list]
        except ValueError:
            return default
    
    def _load_project_config(self):
        """Load project configuration"""
        self.config.project.name = os.getenv('PROJECT_NAME', self.config.project.name)
        self.config.project.version = os.getenv('PROJECT_VERSION', self.config.project.version)
        self.config.project.description = os.getenv('PROJECT_DESCRIPTION', self.config.project.description)
        self.config.project.authors = self._get_env_list('PROJECT_AUTHORS', self.config.project.authors)
    
    def _load_paths_config(self):
        """Load paths configuration"""
        self.config.paths.data_dir = os.getenv('DATA_DIR', self.config.paths.data_dir)
        self.config.paths.models_dir = os.getenv('MODELS_DIR', self.config.paths.models_dir)
        self.config.paths.results_dir = os.getenv('RESULTS_DIR', self.config.paths.results_dir)
        self.config.paths.logs_dir = os.getenv('LOGS_DIR', self.config.paths.logs_dir)
        self.config.paths.config_dir = os.getenv('CONFIG_DIR', self.config.paths.config_dir)
        self.config.paths.videos_dir = os.getenv('VIDEOS_DIR', self.config.paths.videos_dir)
        self.config.paths.database_path = os.getenv('DATABASE_PATH', self.config.paths.database_path)
    
    def _load_data_config(self):
        """Load data configuration"""
        self.config.data.sample_size = self._get_env_int('DATA_SAMPLE_SIZE', self.config.data.sample_size)
        self.config.data.test_size = self._get_env_float('DATA_TEST_SIZE', self.config.data.test_size)
        self.config.data.random_state = self._get_env_int('DATA_RANDOM_STATE', self.config.data.random_state)
        self.config.data.min_binding_affinity = self._get_env_float('DATA_MIN_BINDING_AFFINITY', self.config.data.min_binding_affinity)
        self.config.data.max_binding_affinity = self._get_env_float('DATA_MAX_BINDING_AFFINITY', self.config.data.max_binding_affinity)
        self.config.data.max_rows = self._get_env_int('DATA_MAX_ROWS', self.config.data.max_rows)
        
        self.config.data.bindingdb_url = os.getenv('DATA_BINDINGDB_URL', self.config.data.bindingdb_url)
        self.config.data.chembl_base_url = os.getenv('DATA_CHEMBL_BASE_URL', self.config.data.chembl_base_url)
        self.config.data.pdb_base_url = os.getenv('DATA_PDB_BASE_URL', self.config.data.pdb_base_url)
        
        self.config.data.bindingdb_enabled = self._get_env_bool('DATA_BINDINGDB_ENABLED', self.config.data.bindingdb_enabled)
        self.config.data.sample_enabled = self._get_env_bool('DATA_SAMPLE_ENABLED', self.config.data.sample_enabled)
    
    def _load_models_config(self):
        """Load models configuration"""
        # Random Forest
        self.config.models.rf.enabled = self._get_env_bool('MODEL_RF_ENABLED', self.config.models.rf.enabled)
        self.config.models.rf.n_estimators = self._get_env_int('MODEL_RF_N_ESTIMATORS', self.config.models.rf.n_estimators)
        max_depth = os.getenv('MODEL_RF_MAX_DEPTH', str(self.config.models.rf.max_depth))
        self.config.models.rf.max_depth = None if max_depth.lower() == 'none' else int(max_depth)
        self.config.models.rf.min_samples_split = self._get_env_int('MODEL_RF_MIN_SAMPLES_SPLIT', self.config.models.rf.min_samples_split)
        self.config.models.rf.min_samples_leaf = self._get_env_int('MODEL_RF_MIN_SAMPLES_LEAF', self.config.models.rf.min_samples_leaf)
        self.config.models.rf.max_features = os.getenv('MODEL_RF_MAX_FEATURES', self.config.models.rf.max_features)
        self.config.models.rf.bootstrap = self._get_env_bool('MODEL_RF_BOOTSTRAP', self.config.models.rf.bootstrap)
        self.config.models.rf.random_state = self._get_env_int('MODEL_RF_RANDOM_STATE', self.config.models.rf.random_state)
        self.config.models.rf.n_jobs = self._get_env_int('MODEL_RF_N_JOBS', self.config.models.rf.n_jobs)
        
        # XGBoost
        self.config.models.xgb.enabled = self._get_env_bool('MODEL_XGB_ENABLED', self.config.models.xgb.enabled)
        self.config.models.xgb.n_estimators = self._get_env_int('MODEL_XGB_N_ESTIMATORS', self.config.models.xgb.n_estimators)
        self.config.models.xgb.max_depth = self._get_env_int('MODEL_XGB_MAX_DEPTH', self.config.models.xgb.max_depth)
        self.config.models.xgb.learning_rate = self._get_env_float('MODEL_XGB_LEARNING_RATE', self.config.models.xgb.learning_rate)
        self.config.models.xgb.subsample = self._get_env_float('MODEL_XGB_SUBSAMPLE', self.config.models.xgb.subsample)
        self.config.models.xgb.colsample_bytree = self._get_env_float('MODEL_XGB_COLSAMPLE_BYTREE', self.config.models.xgb.colsample_bytree)
        self.config.models.xgb.random_state = self._get_env_int('MODEL_XGB_RANDOM_STATE', self.config.models.xgb.random_state)
        
        # Neural Network
        self.config.models.nn.enabled = self._get_env_bool('MODEL_NN_ENABLED', self.config.models.nn.enabled)
        self.config.models.nn.hidden_layers = self._get_env_int_list('MODEL_NN_HIDDEN_LAYERS', self.config.models.nn.hidden_layers)
        self.config.models.nn.dropout_rate = self._get_env_float('MODEL_NN_DROPOUT_RATE', self.config.models.nn.dropout_rate)
        self.config.models.nn.learning_rate = self._get_env_float('MODEL_NN_LEARNING_RATE', self.config.models.nn.learning_rate)
        self.config.models.nn.batch_size = self._get_env_int('MODEL_NN_BATCH_SIZE', self.config.models.nn.batch_size)
        self.config.models.nn.epochs = self._get_env_int('MODEL_NN_EPOCHS', self.config.models.nn.epochs)
        self.config.models.nn.random_state = self._get_env_int('MODEL_NN_RANDOM_STATE', self.config.models.nn.random_state)
        
        # Model files
        self.config.models.rf_file = os.getenv('MODEL_RF_FILE', self.config.models.rf_file)
        self.config.models.xgb_file = os.getenv('MODEL_XGB_FILE', self.config.models.xgb_file)
        self.config.models.nn_file = os.getenv('MODEL_NN_FILE', self.config.models.nn_file)
    
    def _load_features_config(self):
        """Load features configuration"""
        self.config.features.molecular_basic = self._get_env_bool('FEATURES_MOLECULAR_BASIC', self.config.features.molecular_basic)
        self.config.features.molecular_topological = self._get_env_bool('FEATURES_MOLECULAR_TOPOLOGICAL', self.config.features.molecular_topological)
        self.config.features.molecular_electronic = self._get_env_bool('FEATURES_MOLECULAR_ELECTRONIC', self.config.features.molecular_electronic)
        self.config.features.molecular_geometric = self._get_env_bool('FEATURES_MOLECULAR_GEOMETRIC', self.config.features.molecular_geometric)
        
        self.config.features.protein_sequence = self._get_env_bool('FEATURES_PROTEIN_SEQUENCE', self.config.features.protein_sequence)
        self.config.features.protein_structural = self._get_env_bool('FEATURES_PROTEIN_STRUCTURAL', self.config.features.protein_structural)
        self.config.features.protein_binding_site = self._get_env_bool('FEATURES_PROTEIN_BINDING_SITE', self.config.features.protein_binding_site)
        
        self.config.features.interaction_molecular_protein = self._get_env_bool('FEATURES_INTERACTION_MOLECULAR_PROTEIN', self.config.features.interaction_molecular_protein)
        self.config.features.interaction_binding_mode = self._get_env_bool('FEATURES_INTERACTION_BINDING_MODE', self.config.features.interaction_binding_mode)
        
        self.config.features.molecular_descriptors = self._get_env_list('MOLECULAR_DESCRIPTORS', self.config.features.molecular_descriptors)
    
    def _load_visualization_config(self):
        """Load visualization configuration"""
        self.config.visualization.plot_style = os.getenv('VIZ_PLOT_STYLE', self.config.visualization.plot_style)
        self.config.visualization.figure_width = self._get_env_int('VIZ_FIGURE_WIDTH', self.config.visualization.figure_width)
        self.config.visualization.figure_height = self._get_env_int('VIZ_FIGURE_HEIGHT', self.config.visualization.figure_height)
        self.config.visualization.dpi = self._get_env_int('VIZ_DPI', self.config.visualization.dpi)
        self.config.visualization.save_format = os.getenv('VIZ_SAVE_FORMAT', self.config.visualization.save_format)
        
        self.config.visualization.molecular_width = self._get_env_int('VIZ_MOLECULAR_WIDTH', self.config.visualization.molecular_width)
        self.config.visualization.molecular_height = self._get_env_int('VIZ_MOLECULAR_HEIGHT', self.config.visualization.molecular_height)
        self.config.visualization.molecular_style = os.getenv('VIZ_MOLECULAR_STYLE', self.config.visualization.molecular_style)
        
        self.config.visualization.fps = self._get_env_int('ANIMATION_FPS', self.config.visualization.fps)
        self.config.visualization.duration = self._get_env_int('ANIMATION_DURATION', self.config.visualization.duration)
        self.config.visualization.output_dir = os.getenv('ANIMATION_OUTPUT_DIR', self.config.visualization.output_dir)
        
        self.config.visualization.color_protein = os.getenv('VIZ_COLOR_PROTEIN', self.config.visualization.color_protein)
        self.config.visualization.color_ligand = os.getenv('VIZ_COLOR_LIGAND', self.config.visualization.color_ligand)
        self.config.visualization.color_binding_site = os.getenv('VIZ_COLOR_BINDING_SITE', self.config.visualization.color_binding_site)
        
        # Load atom colors
        for atom in ['C', 'O', 'N', 'H', 'S', 'P', 'F', 'CL', 'BR', 'I', 'DEFAULT']:
            env_key = f'ATOM_COLOR_{atom}'
            if env_key in os.environ:
                self.config.visualization.atom_colors[atom if atom != 'CL' else 'Cl'] = os.environ[env_key]
    
    def _load_performance_config(self):
        """Load performance configuration"""
        self.config.performance.target_r2 = self._get_env_float('PERFORMANCE_TARGET_R2', self.config.performance.target_r2)
        self.config.performance.target_rmse = self._get_env_float('PERFORMANCE_TARGET_RMSE', self.config.performance.target_rmse)
        self.config.performance.max_prediction_time = self._get_env_float('PERFORMANCE_MAX_PREDICTION_TIME', self.config.performance.max_prediction_time)
        self.config.performance.cv_folds = self._get_env_int('PERFORMANCE_CV_FOLDS', self.config.performance.cv_folds)
    
    def _load_logging_config(self):
        """Load logging configuration"""
        self.config.logging.level = os.getenv('LOG_LEVEL', self.config.logging.level)
        self.config.logging.format = os.getenv('LOG_FORMAT', self.config.logging.format)
        self.config.logging.file = os.getenv('LOG_FILE', self.config.logging.file)
    
    def _load_streamlit_config(self):
        """Load Streamlit configuration"""
        self.config.streamlit.page_title = os.getenv('STREAMLIT_PAGE_TITLE', self.config.streamlit.page_title)
        self.config.streamlit.page_icon = os.getenv('STREAMLIT_PAGE_ICON', self.config.streamlit.page_icon)
        self.config.streamlit.layout = os.getenv('STREAMLIT_LAYOUT', self.config.streamlit.layout)
        self.config.streamlit.sidebar_state = os.getenv('STREAMLIT_SIDEBAR_STATE', self.config.streamlit.sidebar_state)
    
    def _load_ollama_config(self):
        """Load Ollama configuration"""
        self.config.ollama.enabled = self._get_env_bool('OLLAMA_ENABLED', self.config.ollama.enabled)
        self.config.ollama.host = os.getenv('OLLAMA_HOST', self.config.ollama.host)
        self.config.ollama.model = os.getenv('OLLAMA_MODEL', self.config.ollama.model)
        self.config.ollama.temperature = self._get_env_float('OLLAMA_TEMPERATURE', self.config.ollama.temperature)
        self.config.ollama.max_tokens = self._get_env_int('OLLAMA_MAX_TOKENS', self.config.ollama.max_tokens)
        self.config.ollama.timeout = self._get_env_int('OLLAMA_TIMEOUT', self.config.ollama.timeout)
        self.config.ollama.chat_title = os.getenv('OLLAMA_CHAT_TITLE', self.config.ollama.chat_title)
        self.config.ollama.welcome_message = os.getenv('OLLAMA_WELCOME_MESSAGE', self.config.ollama.welcome_message)
        self.config.ollama.error_message = os.getenv('OLLAMA_ERROR_MESSAGE', self.config.ollama.error_message)
        self.config.ollama.system_prompt = os.getenv('OLLAMA_SYSTEM_PROMPT', self.config.ollama.system_prompt)
    
    def _load_cli_config(self):
        """Load CLI configuration"""
        self.config.cli.default_models = self._get_env_list('CLI_DEFAULT_MODELS', self.config.cli.default_models)
        self.config.cli.default_data_source = os.getenv('CLI_DEFAULT_DATA_SOURCE', self.config.cli.default_data_source)
        self.config.cli.force_reprocess = self._get_env_bool('CLI_FORCE_REPROCESS', self.config.cli.force_reprocess)
    
    def get_config(self) -> AffinifyConfig:
        """Get the complete configuration object"""
        return self.config
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get model parameters for a specific model type"""
        if model_type.lower() == 'randomforest' or model_type.lower() == 'rf':
            return {
                'n_estimators': self.config.models.rf.n_estimators,
                'max_depth': self.config.models.rf.max_depth,
                'min_samples_split': self.config.models.rf.min_samples_split,
                'min_samples_leaf': self.config.models.rf.min_samples_leaf,
                'max_features': self.config.models.rf.max_features,
                'bootstrap': self.config.models.rf.bootstrap,
                'random_state': self.config.models.rf.random_state,
                'n_jobs': self.config.models.rf.n_jobs
            }
        elif model_type.lower() == 'xgboost' or model_type.lower() == 'xgb':
            return {
                'n_estimators': self.config.models.xgb.n_estimators,
                'max_depth': self.config.models.xgb.max_depth,
                'learning_rate': self.config.models.xgb.learning_rate,
                'subsample': self.config.models.xgb.subsample,
                'colsample_bytree': self.config.models.xgb.colsample_bytree,
                'random_state': self.config.models.xgb.random_state
            }
        elif model_type.lower() == 'neuralnetwork' or model_type.lower() == 'nn':
            return {
                'hidden_layers': self.config.models.nn.hidden_layers,
                'dropout_rate': self.config.models.nn.dropout_rate,
                'learning_rate': self.config.models.nn.learning_rate,
                'batch_size': self.config.models.nn.batch_size,
                'epochs': self.config.models.nn.epochs,
                'random_state': self.config.models.nn.random_state
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_enabled_models(self) -> List[str]:
        """Get list of enabled models"""
        enabled = []
        if self.config.models.rf.enabled:
            enabled.append('RandomForest')
        if self.config.models.xgb.enabled:
            enabled.append('XGBoost')
        if self.config.models.nn.enabled:
            enabled.append('NeuralNetwork')
        return enabled
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration"""
        # Create logs directory
        logs_dir = Path(self.config.paths.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level.upper()),
            format=self.config.logging.format,
            handlers=[
                logging.FileHandler(self.config.logging.file, mode='w'),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger('affinify')

# Global configuration instance
_config_manager = None

def get_config(env_path: Optional[str] = None) -> AffinifyConfig:
    """
    Get the global configuration instance.
    
    Args:
        env_path: Path to .env file. Only used on first call.
    
    Returns:
        AffinifyConfig: The configuration object
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(env_path)
    return _config_manager.get_config()

def get_config_manager(env_path: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        env_path: Path to .env file. Only used on first call.
    
    Returns:
        ConfigManager: The configuration manager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(env_path)
    return _config_manager

if __name__ == "__main__":
    # Test the configuration system
    config = get_config()
    print("Affinify Configuration Loaded Successfully!")
    print(f"Project: {config.project.name} v{config.project.version}")
    print(f"Data directory: {config.paths.data_dir}")
    print(f"Enabled models: {get_config_manager().get_enabled_models()}")
    print(f"Animation FPS: {config.visualization.fps}")
    print(f"Ollama enabled: {config.ollama.enabled}")
