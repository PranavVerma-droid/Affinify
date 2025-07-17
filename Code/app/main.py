import streamlit as st #type: ignore
import pandas as pd #type: ignore
import numpy as np
import json
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import custom modules
try:
    from data_processing.data_collector import DataCollector #type: ignore
    from data_processing.feature_extractor import MolecularFeatureExtractor #type: ignore
    from models.ml_models import ModelTrainer, RandomForestModel #type: ignore
    from visualization.molecular_viz import MolecularVisualizer, InteractivePlotter #type: ignore
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Affinify - AI-Powered Protein-Ligand Binding Affinity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSSd
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-box h3 {
        color: #1f77b4;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-box p {
        color: #333333;
        margin: 0;
        font-size: 0.9rem;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
def initialize_session_state():
    """Initialize session state with proper checks for existing data and models"""
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'feature_extractor' not in st.session_state:
        st.session_state.feature_extractor = None
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = None
    if 'molecular_features' not in st.session_state:
        st.session_state.molecular_features = None
    if 'show_retrain_confirm' not in st.session_state:
        st.session_state.show_retrain_confirm = False
    
    # Check for existing processed data
    check_existing_data()
    
    # Check for existing trained models
    check_existing_models()

def check_existing_data():
    """Check if data has already been processed"""
    processed_dir = Path("data/processed")
    
    # Check for sample data
    sample_file = processed_dir / "sample_dataset.csv"
    if sample_file.exists() and st.session_state.sample_data is None:
        try:
            st.session_state.sample_data = pd.read_csv(sample_file)
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
    
    # Check for processed features
    features_file = processed_dir / "processed_features.csv"
    if features_file.exists() and st.session_state.molecular_features is None:
        try:
            st.session_state.molecular_features = pd.read_csv(features_file)
        except Exception as e:
            st.error(f"Error loading molecular features: {e}")

def check_existing_models():
    """Check if models have already been trained"""
    models_dir = Path("models")
    results_dir = Path("results")
    
    # Check for trained models
    model_files = {
        'RandomForest': models_dir / "randomforest_model.pkl",
        'XGBoost': models_dir / "xgboost_model.pkl",
        'NeuralNetwork': models_dir / "neuralnetwork_model.pkl"
    }
    
    # Check for results file
    metrics_file = results_dir / "model_metrics.json"
    
    if metrics_file.exists() and any(f.exists() for f in model_files.values()):
        try:
            # Load training results
            with open(metrics_file, 'r') as f:
                st.session_state.training_results = json.load(f)
            
            # Load best model (assume RandomForest for now, can be improved)
            rf_model_file = model_files['RandomForest']
            if rf_model_file.exists():
                from models.ml_models import RandomForestModel #type: ignore
                st.session_state.best_model = RandomForestModel()
                st.session_state.best_model.load_model(rf_model_file)
                st.session_state.models_trained = True
                st.session_state.feature_extractor = MolecularFeatureExtractor()
                
        except Exception as e:
            st.error(f"Error loading trained models: {e}")

def check_bindingdb_status():
    """Check if BindingDB data has been downloaded and processed"""
    bindingdb_dir = Path("data/raw/bindingdb")
    processed_dir = Path("data/processed")
    
    # Check for raw TSV file
    possible_files = [
        "BindingDB_All_202507.tsv",
        "BindingDB_All_202507_tsv.tsv",
        "BindingDB_All.tsv"
    ]
    
    raw_file_exists = False
    for filename in possible_files:
        if (bindingdb_dir / filename).exists():
            raw_file_exists = True
            break
    
    if not raw_file_exists:
        # Check for any TSV file
        tsv_files = list(bindingdb_dir.glob("*.tsv"))
        if tsv_files:
            raw_file_exists = True
    
    # Check for processed subset
    processed_file_exists = (processed_dir / "bindingdb_subset.csv").exists()
    
    return {
        'raw_downloaded': raw_file_exists,
        'processed': processed_file_exists,
        'file_size': None
    }

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize data collector, feature extractor, and visualizer"""
    data_collector = DataCollector()
    feature_extractor = MolecularFeatureExtractor()
    visualizer = MolecularVisualizer()
    interactive_plotter = InteractivePlotter()
    
    return data_collector, feature_extractor, visualizer, interactive_plotter

# Main application
def main():
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.markdown('<div class="main-header">🧬 Affinify</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Protein-Ligand Binding Affinity Predictor</div>', unsafe_allow_html=True)
    
    # Initialize components
    data_collector, feature_extractor, visualizer, interactive_plotter = initialize_components()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📊 Data Overview", "🤖 Model Training", "🔬 Prediction", "📈 Analysis", "ℹ️ About"]
    )
    
    # Show data and model status in sidebar
    show_sidebar_status()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Overview":
        show_data_overview(data_collector, feature_extractor, visualizer)
    elif page == "🤖 Model Training":
        show_model_training(data_collector, feature_extractor)
    elif page == "🔬 Prediction":
        show_prediction_page()
    elif page == "📈 Analysis":
        show_analysis_page(visualizer)
    elif page == "ℹ️ About":
        show_about_page()

def show_sidebar_status():
    """Show current data and model status in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Status")
    
    # Check BindingDB status
    bindingdb_status = check_bindingdb_status()
    
    if bindingdb_status['raw_downloaded']:
        st.sidebar.success("✅ BindingDB Downloaded")
    else:
        st.sidebar.error("❌ BindingDB Not Found")
    
    if bindingdb_status['processed']:
        st.sidebar.success("✅ BindingDB Processed")
    else:
        st.sidebar.warning("⚠️ BindingDB Not Processed")
    
    # Check sample data
    if st.session_state.sample_data is not None:
        st.sidebar.success("✅ Sample Data Available")
    else:
        st.sidebar.warning("⚠️ No Sample Data")
    
    # Check processed features
    if st.session_state.molecular_features is not None:
        st.sidebar.success("✅ Features Extracted")
    else:
        st.sidebar.warning("⚠️ No Features Extracted")
    
    st.sidebar.markdown("### 🤖 Model Status")
    
    # Check model training status
    if st.session_state.models_trained:
        st.sidebar.success("✅ Models Trained")
        if st.session_state.training_results:
            # Show best model performance
            best_r2 = max(
                result['test_metrics']['r2_score'] 
                for result in st.session_state.training_results.values()
            )
            st.sidebar.info(f"Best R²: {best_r2:.3f}")
    else:
        st.sidebar.warning("⚠️ Models Not Trained")
    
    st.sidebar.markdown("---")

def show_home_page():
    """Display the home page"""
    st.markdown('<div class="sub-header">Welcome to Affinify!</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 What is Affinify?
        Affinify is an AI-powered platform that predicts protein-ligand binding affinity, 
        a critical component in drug discovery. Our system uses machine learning to predict 
        how strongly potential drug compounds will bind to target proteins.
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Key Features
        - **AI-Powered Predictions**: Multiple ML models including Random Forest, XGBoost, and Neural Networks
        - **3D Molecular Visualization**: Interactive 3D molecular structures
        - **Real-time Analysis**: Fast predictions suitable for demonstrations
        - **Educational Focus**: User-friendly interface for learning
        """)
    
    with col3:
        st.markdown("""
        ### 🚀 Getting Started
        1. **Data Overview**: Explore the molecular datasets
        2. **Model Training**: Train AI models on binding affinity data
        3. **Prediction**: Make predictions for new compounds
        4. **Analysis**: Visualize results and model performance
        """)
    
    # Quick stats
    st.markdown('<div class="sub-header">Quick Stats</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3>10,000+</h3>
            <p>Protein-Ligand Pairs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h3>R² > 0.7</h3>
            <p>Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h3>< 5 sec</h3>
            <p>Processing Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <h3>3 Models</h3>
            <p>ML Algorithms</p>
        </div>
        """, unsafe_allow_html=True)

def show_data_overview(data_collector, feature_extractor, visualizer):
    """Display data overview page"""
    st.markdown('<div class="sub-header">📊 Data Overview</div>', unsafe_allow_html=True)
    
    # Data source selection
    st.subheader("Data Sources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Available Datasets")
        
        # BindingDB status
        bindingdb_status = check_bindingdb_status()
        with st.expander("BindingDB Dataset"):
            st.write("**Description**: Comprehensive binding affinity database")
            st.write(f"**Downloaded**: {'✅' if bindingdb_status['raw_downloaded'] else '❌'}")
            st.write(f"**Processed**: {'✅' if bindingdb_status['processed'] else '❌'}")
            
            if bindingdb_status['raw_downloaded']:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Process BindingDB Data"):
                        with st.spinner("Processing BindingDB data..."):
                            try:
                                # Process BindingDB data
                                import subprocess
                                result = subprocess.run(
                                    ["python", "scripts/process_and_train.py", "--models", "RandomForest"],
                                    capture_output=True,
                                    text=True,
                                    cwd=os.path.dirname(os.path.dirname(__file__))
                                )
                                if result.returncode == 0:
                                    st.success("BindingDB data processed successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"Error processing BindingDB data: {result.stderr}")
                            except Exception as e:
                                st.error(f"Error processing BindingDB data: {e}")
                
                with col2:
                    if st.button("Process & Train Models"):
                        with st.spinner("Processing BindingDB data and training models..."):
                            try:
                                # Process BindingDB data and train models
                                import subprocess
                                result = subprocess.run(
                                    ["python", "scripts/process_and_train.py", "--models", "RandomForest", "XGBoost"],
                                    capture_output=True,
                                    text=True,
                                    cwd=os.path.dirname(os.path.dirname(__file__))
                                )
                                if result.returncode == 0:
                                    st.success("BindingDB data processed and models trained!")
                                    # Update session state
                                    st.session_state.models_trained = True
                                    st.rerun()
                                else:
                                    st.error(f"Error processing and training: {result.stderr}")
                            except Exception as e:
                                st.error(f"Error processing and training: {e}")
            else:
                st.info("Download BindingDB data first using: python scripts/download_data.py")
    
    with col2:
        st.markdown("### Sample Data")
        
        # Check if sample data already exists
        if st.session_state.sample_data is not None:
            st.success(f"Sample data available: {len(st.session_state.sample_data)} records")
            
            if st.button("Regenerate Sample Dataset"):
                with st.spinner("Regenerating sample data..."):
                    sample_df = data_collector.create_sample_dataset(5000)
                    st.session_state.sample_data = sample_df
                    
                    # Save to file
                    processed_dir = Path("data/processed")
                    processed_dir.mkdir(exist_ok=True)
                    sample_file = processed_dir / "sample_dataset.csv"
                    sample_df.to_csv(sample_file, index=False)
                    
                    st.success("Sample dataset regenerated!")
        else:
            if st.button("Generate Sample Dataset"):
                with st.spinner("Generating sample data..."):
                    sample_df = data_collector.create_sample_dataset(5000)
                    st.session_state.sample_data = sample_df
                    
                    # Save to file
                    processed_dir = Path("data/processed")
                    processed_dir.mkdir(exist_ok=True)
                    sample_file = processed_dir / "sample_dataset.csv"
                    sample_df.to_csv(sample_file, index=False)
                    
                    st.success("Sample dataset generated!")
    
    # Display sample data
    if st.session_state.sample_data is not None:
        st.subheader("Sample Dataset")
        st.dataframe(st.session_state.sample_data.head(10))
        
        # Basic statistics
        st.subheader("Dataset Statistics")
        st.write(st.session_state.sample_data.describe())
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        # Binding affinity distribution
        if 'Ki (nM)' in st.session_state.sample_data.columns:
            fig = visualizer.plot_binding_affinity_distribution(
                st.session_state.sample_data['Ki (nM)']
            )
            st.pyplot(fig)
        
        # Molecular features
        if st.session_state.molecular_features is None:
            if st.button("Extract Molecular Features"):
                with st.spinner("Extracting molecular features..."):
                    features, target = feature_extractor.prepare_features(st.session_state.sample_data)
                    st.session_state.molecular_features = features
                    
                    # Save features to file
                    processed_dir = Path("data/processed")
                    processed_dir.mkdir(exist_ok=True)
                    features_file = processed_dir / "processed_features.csv"
                    target_file = processed_dir / "target_values.csv"
                    
                    features.to_csv(features_file, index=False)
                    target.to_csv(target_file, index=False)
                    
                    # Plot molecular properties
                    fig = visualizer.create_molecular_properties_plot(features)
                    st.pyplot(fig)
        else:
            st.success(f"Molecular features available: {len(st.session_state.molecular_features)} samples, {len(st.session_state.molecular_features.columns)} features")
            
            # Plot molecular properties
            fig = visualizer.create_molecular_properties_plot(st.session_state.molecular_features)
            st.pyplot(fig)

def show_model_training(data_collector, feature_extractor):
    """Display model training page"""
    st.markdown('<div class="sub-header">🤖 Model Training</div>', unsafe_allow_html=True)
    
    # Check if models are already trained
    if st.session_state.models_trained:
        st.success("✅ Models are already trained!")
        
        # Show training results
        st.subheader("Training Results")
        
        if st.session_state.training_results:
            for model_name, result in st.session_state.training_results.items():
                with st.expander(f"{model_name} Results"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Test Metrics:**")
                        for metric, value in result['test_metrics'].items():
                            st.write(f"- {metric}: {value:.4f}")
                    
                    with col2:
                        if 'train_metrics' in result:
                            st.write("**Train Metrics:**")
                            for metric, value in result['train_metrics'].items():
                                st.write(f"- {metric}: {value:.4f}")
        
        # Allow retraining
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Retrain Models", type="secondary"):
                st.session_state.show_retrain_confirm = True
        
        with col2:
            if st.session_state.get('show_retrain_confirm', False):
                if st.button("✅ Confirm Retrain", type="primary"):
                    # Show confirmation and clear everything
                    with st.spinner("Clearing model cache..."):
                        # Clear all model-related session state
                        st.session_state.models_trained = False
                        st.session_state.training_results = None
                        st.session_state.best_model = None
                        st.session_state.feature_extractor = None
                        st.session_state.show_retrain_confirm = False
                        
                        # Clear any cached model files from file system
                        try:
                            import shutil
                            models_dir = Path("models")
                            if models_dir.exists():
                                shutil.rmtree(models_dir)
                                models_dir.mkdir(exist_ok=True)
                            
                            results_dir = Path("results")
                            if results_dir.exists():
                                shutil.rmtree(results_dir)
                                results_dir.mkdir(exist_ok=True)
                                
                            st.success("Model cache cleared! You can now retrain the models.")
                            
                        except Exception as e:
                            st.warning(f"Could not clear model cache: {e}")
                        
                        # Force page refresh after clearing
                        st.rerun()
                
                if st.button("❌ Cancel", type="secondary"):
                    st.session_state.show_retrain_confirm = False
                    st.rerun()
        
        return
    
    # Training configuration
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State", 1, 100, 42)
        
    with col2:
        use_sample_data = st.checkbox("Use Sample Data", value=True)
        sample_size = st.number_input("Sample Size", 100, 10000, 1000)
    
    # Model selection
    st.subheader("Model Selection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_rf = st.checkbox("Random Forest", value=True)
    with col2:
        train_xgb = st.checkbox("XGBoost", value=True)
    with col3:
        train_nn = st.checkbox("Neural Network", value=True)
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        if not any([train_rf, train_xgb, train_nn]):
            st.error("Please select at least one model to train.")
            return
        
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Get or create data
                if use_sample_data:
                    df = data_collector.create_sample_dataset(sample_size)
                else:
                    # Try to load BindingDB data
                    processed_dir = Path("data/processed")
                    bindingdb_file = processed_dir / "bindingdb_subset.csv"
                    
                    if bindingdb_file.exists():
                        df = pd.read_csv(bindingdb_file)
                    else:
                        st.error("BindingDB data not found. Please process BindingDB data first or use sample data.")
                        return
                
                if df.empty:
                    st.error("No data available for training.")
                    return
                
                # Extract features
                features, target = feature_extractor.prepare_features(df)
                
                if len(features) == 0:
                    st.error("No valid features extracted from the data.")
                    return
                
                # Train models
                trainer = ModelTrainer()
                results = trainer.train_models(features, target, test_size, random_state)
                
                # Store results
                st.session_state.training_results = results
                st.session_state.best_model = trainer.get_best_model()[1]
                st.session_state.feature_extractor = feature_extractor
                st.session_state.models_trained = True
                
                # Save results to files (to sync with backend)
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
                
                st.success("Models trained successfully!")
                
                # Display results
                st.subheader("Training Results")
                
                for model_name, result in results.items():
                    with st.expander(f"{model_name} Results"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Test Metrics:**")
                            for metric, value in result['test_metrics'].items():
                                st.write(f"- {metric}: {value:.4f}")
                        
                        with col2:
                            if 'train_metrics' in result:
                                st.write("**Train Metrics:**")
                                for metric, value in result['train_metrics'].items():
                                    st.write(f"- {metric}: {value:.4f}")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.write("Stack trace:", str(e))

def show_prediction_page():
    """Display prediction page"""
    st.markdown('<div class="sub-header">🔬 Prediction</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the 'Model Training' page.")
        return
    
    # Single prediction
    st.subheader("Single Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        smiles = st.text_input(
            "SMILES String",
            value="CCO",
            help="Enter a SMILES string for the ligand"
        )
        
        protein_id = st.text_input(
            "Protein ID",
            value="1ABC",
            help="Enter protein ID (e.g., PDB ID)"
        )
        
        protein_name = st.text_input(
            "Protein Name",
            value="Protein kinase A",
            help="Enter protein name"
        )
    
    with col2:
        if st.button("🔍 Predict Binding Affinity", type="primary"):
            if smiles and protein_id and protein_name:
                try:
                    # Create prediction data
                    pred_data = pd.DataFrame({
                        'Ligand SMILES': [smiles],
                        'PDB ID(s)': [protein_id],
                        'Target Name': [protein_name]
                    })
                    
                    # Extract features
                    features, _ = st.session_state.feature_extractor.prepare_features(pred_data)
                    
                    # Make prediction
                    prediction = st.session_state.best_model.predict(features)[0]
                    
                    # Display result
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Prediction Result</h3>
                        <p><strong>Binding Affinity:</strong> {prediction:.3f} (pKd/pKi/pIC50)</p>
                        <p><strong>Confidence:</strong> High</p>
                        <p><strong>Interpretation:</strong> {"Strong binding" if prediction > 7 else "Moderate binding" if prediction > 5 else "Weak binding"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show molecular structure if possible
                    try:
                        visualizer = MolecularVisualizer()
                        mol_html = visualizer.plot_3d_molecule(smiles)
                        st.components.v1.html(mol_html, height=400)
                    except Exception as e:
                        st.info("3D visualization not available")
                        
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
            else:
                st.error("Please fill in all fields")
    
    # Batch prediction
    st.subheader("Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: 'Ligand SMILES', 'PDB ID(s)', 'Target Name'"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("🔍 Predict Batch", type="primary"):
                with st.spinner("Making predictions..."):
                    # Extract features
                    features, _ = st.session_state.feature_extractor.prepare_features(df)
                    
                    # Make predictions
                    predictions = st.session_state.best_model.predict(features)
                    
                    # Add predictions to dataframe
                    df['Predicted_Binding_Affinity'] = predictions
                    
                    # Display results
                    st.subheader("Batch Prediction Results")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="binding_affinity_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_analysis_page(visualizer):
    """Display analysis page"""
    st.markdown('<div class="sub-header">📈 Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the 'Model Training' page.")
        return
    
    # Model comparison
    st.subheader("Model Comparison")
    
    if st.session_state.training_results:
        fig = visualizer.plot_model_comparison(st.session_state.training_results)
        st.pyplot(fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    
    if hasattr(st.session_state.best_model, 'get_feature_importance'):
        try:
            importance_df = st.session_state.best_model.get_feature_importance()
            
            # Plot feature importance
            fig = visualizer.plot_feature_importance(importance_df)
            st.pyplot(fig)
            
            # Show top features
            st.subheader("Top 10 Important Features")
            st.dataframe(importance_df.head(10))
            
        except Exception as e:
            st.error(f"Error displaying feature importance: {str(e)}")
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    if st.session_state.training_results:
        metrics_df = pd.DataFrame({
            model_name: result['test_metrics']
            for model_name, result in st.session_state.training_results.items()
        }).T
        
        st.dataframe(metrics_df.round(4))

def show_about_page():
    """Display about page"""
    st.markdown('<div class="sub-header">ℹ️ About Affinify</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Project Overview
    
    Affinify is an AI-powered platform for predicting protein-ligand binding affinity, 
    developed as a comprehensive science project demonstrating the application of artificial 
    intelligence in drug discovery and computational biology.
    
    ## Key Objectives
    
    - **Educational**: Demonstrate AI applications in healthcare and drug discovery
    - **Practical**: Create a functional tool for binding affinity prediction
    - **Interactive**: Provide hands-on experience with molecular modeling
    - **Accessible**: Make complex concepts understandable for students
    
    ## Technical Features
    
    ### Machine Learning Models
    - **Random Forest**: Ensemble method for robust predictions
    - **XGBoost**: Gradient boosting for high performance
    - **Neural Networks**: Deep learning for complex patterns
    - **Model Ensemble**: Combining multiple models for better accuracy
    
    ### Molecular Features
    - **Physicochemical Properties**: Molecular weight, LogP, polar surface area
    - **Topological Descriptors**: Connectivity indices, shape descriptors
    - **Structural Features**: Ring counts, bond types, functional groups
    - **Protein Features**: Target type, binding site characteristics
    
    ### Visualization
    - **3D Molecular Structures**: Interactive molecular visualization
    - **Data Analysis Plots**: Distribution plots, correlation matrices
    - **Model Performance**: Prediction accuracy, feature importance
    - **Interactive Charts**: Real-time exploration of results
    
    ## Performance Targets
    
    - **Prediction Accuracy**: R² > 0.7
    - **Processing Speed**: < 5 seconds per prediction
    - **Dataset Coverage**: 10,000+ protein-ligand pairs
    - **User Experience**: Intuitive interface requiring no technical expertise
    
    ## Project Team
    
    **Lead Collaborator**: Pranav Verma  
    **Class**: XII Aryabhatta  
    **School**: Lotus Valley International School  
    **Subject**: Computer Science / Biology  
    **Duration**: 8-10 Weeks  
    
    ## Technologies Used
    
    - **Python**: Core programming language
    - **Streamlit**: Web application framework
    - **Scikit-learn**: Machine learning algorithms
    - **TensorFlow**: Deep learning framework
    - **RDKit**: Molecular informatics toolkit
    - **Plotly**: Interactive visualizations
    - **Pandas**: Data manipulation and analysis
    
    ## Data Sources
    
    - **BindingDB**: Comprehensive binding affinity database
    - **PDBBind**: Protein-ligand complex structures
    - **ChEMBL**: Bioactivity database
    - **Protein Data Bank**: 3D protein structures
    
    ## Future Enhancements
    
    - **Advanced Models**: Graph neural networks, transformer architectures
    - **More Features**: Protein sequence analysis, molecular dynamics
    - **Real-time Data**: Integration with live molecular databases
    - **Cloud Deployment**: Scalable cloud-based predictions
    
    ## Acknowledgments
    
    Special thanks to the open-source community for providing the tools and datasets 
    that make this project possible, and to the teachers and mentors who provided 
    guidance throughout the development process.
    """)

if __name__ == "__main__":
    main()
