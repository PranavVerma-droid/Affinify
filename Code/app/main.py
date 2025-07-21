#!/usr/bin/env python3
"""
Affinify Web App - Unified Streamlit Interface
A clean, unified web application for the Affinify protein-ligand binding affinity predictor.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import subprocess
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

from models.ml_models import AffinityPredictor
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import py3Dmol
import base64
from io import BytesIO

try:
    from data_processing.data_collector import DataCollector
    from data_processing.feature_extractor import MolecularFeatureExtractor
    from models.ml_models import ModelTrainer
    from visualization.molecular_viz import MolecularVisualizer
    from utils.ollama_chat import create_ollama_chat
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Affinify - AI-Powered Binding Affinity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 1rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.sub-header {
    font-size: 1.5rem;
    color: #34495e;
    margin: 1rem 0;
    border-left: 4px solid #3498db;
    padding-left: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #6c7b7f 0%, #435055 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: transform 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
}
.success-card {
    background: linear-gradient(135deg, #a8b5b3 0%, #7a8b87 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: #2c3e50;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 5px solid #27ae60;
}
.warning-card {
    background: linear-gradient(135deg, #d5dbda 0%, #a8b5b3 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: #2c3e50;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 5px solid #e67e22;
}
.stButton > button {
    background: linear-gradient(135deg, #6c7b7f 0%, #435055 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.75rem 1.5rem;
    font-weight: bold;
    box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

class AffinifyApp:
    """Main application class for the Affinify web interface"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.cli_script = self.project_root / "scripts" / "affinity_cli.py"
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        
        # Initialize Ollama chat
        self.chat = create_ollama_chat()
        
        # Initialize session state
        self.init_session_state()
        
        # Load system status
        self.load_system_status()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'system_status' not in st.session_state:
            st.session_state.system_status = {}
        if 'processing_log' not in st.session_state:
            st.session_state.processing_log = []
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'getting_response' not in st.session_state:
            st.session_state.getting_response = False
        if 'ollama_available' not in st.session_state:
            st.session_state.ollama_available = False
        if 'protein_search' not in st.session_state:
            st.session_state.protein_search = ''
    
    def load_system_status(self):
        """Load current system status"""
        status = {
            'bindingdb_downloaded': self.check_bindingdb_data(),
            'data_processed': self.check_processed_data(),
            'models_trained': self.check_trained_models(),
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.system_status = status
    
    def check_bindingdb_data(self):
        """Check if BindingDB data is available"""
        bindingdb_dir = self.data_dir / "raw" / "bindingdb"
        tsv_files = list(bindingdb_dir.glob("*.tsv")) if bindingdb_dir.exists() else []
        return len(tsv_files) > 0
    
    def check_processed_data(self):
        """Check if processed data exists"""
        processed_dir = self.data_dir / "processed"
        if not processed_dir.exists():
            return False
        
        required_files = ["processed_features.csv", "target_values.csv"]
        return all((processed_dir / f).exists() for f in required_files)
    
    def check_trained_models(self):
        """Check if trained models exist"""
        if not self.models_dir.exists():
            return False
        
        model_files = list(self.models_dir.glob("*.pkl"))
        metrics_file = self.results_dir / "model_metrics.json"
        
        return len(model_files) > 0 and metrics_file.exists()
    
    def run_cli_command(self, command_args, description="Processing"):
        """Run CLI command with progress tracking"""
        try:
            # Build full command
            command = ["python", str(self.cli_script)] + command_args
            
            # Show command being run
            st.info(f"Running: {' '.join(command)}")
            
            # Create progress placeholder
            progress_placeholder = st.empty()
            log_placeholder = st.empty()
            
            # Run command
            with st.spinner(f"{description}..."):
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
            
            # Clear progress
            progress_placeholder.empty()
            
            # Show results
            if result.returncode == 0:
                st.success(f"✅ {description} completed successfully!")
                if result.stdout:
                    with st.expander("📋 Process Output"):
                        st.code(result.stdout)
                
                # Update system status
                self.load_system_status()
                
                # Log the operation
                st.session_state.processing_log.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'operation': description,
                    'status': 'success',
                    'command': ' '.join(command)
                })
                
                return True
            else:
                st.error(f"❌ {description} failed!")
                if result.stderr:
                    st.error(f"Error: {result.stderr}")
                if result.stdout:
                    with st.expander("📋 Process Output"):
                        st.code(result.stdout)
                
                # Log the error
                st.session_state.processing_log.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'operation': description,
                    'status': 'failed',
                    'error': result.stderr
                })
                
                return False
                
        except Exception as e:
            st.error(f"❌ Exception during {description}: {str(e)}")
            
            # Log the exception
            st.session_state.processing_log.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'operation': description,
                'status': 'exception',
                'error': str(e)
            })
            
            return False
    
    def show_header(self):
        """Show application header"""
        st.markdown('<div class="main-header">🧬 Affinify</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">AI-Powered Protein-Ligand Binding Affinity Predictor</div>', unsafe_allow_html=True)
    
    def show_sidebar(self):
        """Show sidebar with navigation and status"""
        st.sidebar.title("🧬 Affinify")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigate",
            ["🏠 Home", "📊 Data Pipeline", "🔬 Predictions", "📈 Analysis", "🤖 AI Assistant", "⚙️ Settings", "ℹ️ About"]
        )
        
        st.sidebar.markdown("---")
        
        # System Status
        st.sidebar.markdown("### 📊 System Status")
        
        status = st.session_state.system_status
        
        # BindingDB Status
        if status.get('bindingdb_downloaded', False):
            st.sidebar.markdown("✅ **BindingDB Data**: Available")
        else:
            st.sidebar.markdown("❌ **BindingDB Data**: Not Found")
        
        # Processed Data Status
        if status.get('data_processed', False):
            st.sidebar.markdown("✅ **Processed Data**: Ready")
        else:
            st.sidebar.markdown("❌ **Processed Data**: Not Ready")
        
        # Models Status
        if status.get('models_trained', False):
            st.sidebar.markdown("✅ **Models**: Trained")
            # Show model performance if available
            try:
                metrics_file = self.results_dir / "model_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    best_r2 = max(
                        model_data['test_metrics']['r2_score'] 
                        for model_data in metrics.values()
                    )
                    st.sidebar.markdown(f"📈 **Best R²**: {best_r2:.3f}")
            except:
                pass
        else:
            st.sidebar.markdown("❌ **Models**: Not Trained")
        
        st.sidebar.markdown(f"🕐 **Last Update**: {status.get('last_update', 'Unknown')}")
        
        st.sidebar.markdown("---")
        
        # Quick Actions
        st.sidebar.markdown("### ⚡ Quick Actions")
        
        if st.sidebar.button("🔄 Refresh Status"):
            self.load_system_status()
            st.rerun()
        
        if st.sidebar.button("🗂️ Open Data Folder"):
            st.sidebar.info(f"Data folder: {self.data_dir}")
        
        return page
    
    def show_home_page(self):
        """Show home page with overview"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Purpose</h3>
                <p>Predict protein-ligand binding affinity using AI</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🧬 Data</h3>
                <p>Real BindingDB molecular data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 Models</h3>
                <p>RandomForest, XGBoost, Neural Networks</p>
            </div>
            """, unsafe_allow_html=True)
        
        # System Overview
        st.markdown('<div class="sub-header">📊 System Overview</div>', unsafe_allow_html=True)
        
        status = st.session_state.system_status
        
        if all([status.get('bindingdb_downloaded'), status.get('data_processed'), status.get('models_trained')]):
            st.markdown("""
            <div class="success-card">
                <h3>🎉 System Ready!</h3>
                <p>All components are set up and ready for predictions.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-card">
                <h3>⚠️ Setup Required</h3>
                <p>Please complete the data pipeline setup to start making predictions.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Getting Started
        st.markdown('<div class="sub-header">🚀 Getting Started</div>', unsafe_allow_html=True)
        
        steps = [
            ("1️⃣ **Data Pipeline**", "Download and process molecular data"),
            ("2️⃣ **Train Models**", "Train AI models on the processed data"),
            ("3️⃣ **Make Predictions**", "Predict binding affinity for new compounds"),
            ("4️⃣ **Analyze Results**", "Explore model performance and insights")
        ]
        
        for step, description in steps:
            st.markdown(f"**{step}**: {description}")
        
        # Recent Activity
        if st.session_state.processing_log:
            st.markdown('<div class="sub-header">📋 Recent Activity</div>', unsafe_allow_html=True)
            
            recent_logs = st.session_state.processing_log[-5:]  # Show last 5 operations
            
            for log in reversed(recent_logs):
                status_icon = "✅" if log['status'] == 'success' else "❌"
                st.markdown(f"{status_icon} **{log['operation']}** - {log['timestamp']}")
    
    def show_data_pipeline_page(self):
        """Show data pipeline management page"""
        st.markdown('<div class="sub-header">📊 Data Pipeline</div>', unsafe_allow_html=True)
        
        # Pipeline Status
        col1, col2, col3 = st.columns(3)
        
        status = st.session_state.system_status
        
        with col1:
            if status.get('bindingdb_downloaded'):
                st.success("✅ **BindingDB Data**\nDownloaded and ready")
            else:
                st.error("❌ **BindingDB Data**\nNot found")
        
        with col2:
            if status.get('data_processed'):
                st.success("✅ **Data Processing**\nFeatures extracted")
            else:
                st.warning("⚠️ **Data Processing**\nNot completed")
        
        with col3:
            if status.get('models_trained'):
                st.success("✅ **Model Training**\nModels ready")
            else:
                st.warning("⚠️ **Model Training**\nNot completed")
        
        st.markdown("---")
        
        # Data Operations
        st.markdown("### 🔧 Data Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📥 Data Download")
            st.info("BindingDB data must be downloaded manually due to size (6GB+)")
            
            if st.button("📖 Show Download Instructions"):
                st.markdown("""
                **Manual Download Steps:**
                1. Go to: https://www.bindingdb.org/rwd/bind/downloads/
                2. Find: 'BindingDB_All_202507_tsv.zip' (488 MB)
                3. Download the file
                4. Extract to: `data/raw/bindingdb/`
                5. Refresh this page
                """)
            
            if st.button("🔄 Check for BindingDB Data"):
                self.load_system_status()
                if st.session_state.system_status.get('bindingdb_downloaded'):
                    st.success("✅ BindingDB data found!")
                else:
                    st.error("❌ BindingDB data not found")
        
        with col2:
            st.markdown("#### 🔄 Data Processing")
            
            # Sample Data Option
            col2a, col2b = st.columns(2)
            
            with col2a:
                sample_size = st.selectbox("Sample Size", [1000, 5000, 10000, 25000], index=1)
                
                if st.button("🧪 Process Sample Data"):
                    success = self.run_cli_command(
                        ["--process", "--data-source", "sample", "--sample-size", str(sample_size)],
                        "Processing Sample Data"
                    )
                    if success:
                        st.rerun()
            
            with col2b:
                max_rows = st.selectbox("Max BindingDB Rows", [10000, 25000, 50000, 100000], index=1)
                
                if st.button("🧬 Process BindingDB Data"):
                    if not status.get('bindingdb_downloaded'):
                        st.error("❌ BindingDB data not found. Please download first.")
                    else:
                        success = self.run_cli_command(
                            ["--process", "--data-source", "bindingdb", "--max-rows", str(max_rows)],
                            "Processing BindingDB Data"
                        )
                        if success:
                            st.rerun()
        
        st.markdown("---")
        
        # Model Training
        st.markdown("### 🤖 Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Training Configuration")
            
            # Model selection
            models = st.multiselect(
                "Select Models",
                ["RandomForest", "XGBoost", "NeuralNetwork"],
                default=["RandomForest", "XGBoost"]
            )
            
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        with col2:
            st.markdown("#### 🚀 Training Actions")
            
            if st.button("🔥 Train Models"):
                if not status.get('data_processed'):
                    st.error("❌ No processed data found. Please process data first.")
                elif not models:
                    st.error("❌ Please select at least one model to train.")
                else:
                    success = self.run_cli_command(
                        ["--train", "--models"] + models + ["--test-size", str(test_size)],
                        "Training Models"
                    )
                    if success:
                        st.rerun()
            
            if st.button("⚡ Full Pipeline"):
                if not status.get('bindingdb_downloaded'):
                    st.error("❌ BindingDB data not found. Using sample data instead.")
                    success = self.run_cli_command(
                        ["--process", "--train", "--data-source", "sample", "--sample-size", "5000", "--models"] + models,
                        "Running Full Pipeline (Sample Data)"
                    )
                else:
                    success = self.run_cli_command(
                        ["--process", "--train", "--data-source", "bindingdb", "--max-rows", "25000", "--models"] + models,
                        "Running Full Pipeline (BindingDB)"
                    )
                if success:
                    st.rerun()
        
        # Data Summary
        if status.get('data_processed'):
            st.markdown("---")
            st.markdown("### 📈 Data Summary")
            
            try:
                # Load processed data info
                summary_file = self.data_dir / "processed" / "data_summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Datasets Processed", len(summary.get('datasets_processed', [])))
                    
                    with col2:
                        files_created = summary.get('files_created', {})
                        total_records = sum(f.get('records', 0) for f in files_created.values() if f.get('records'))
                        st.metric("Total Records", f"{total_records:,}")
                    
                    with col3:
                        st.metric("Processing Time", summary.get('processing_timestamp', 'Unknown'))
                
                # Show file details
                if summary_file.exists():
                    with st.expander("📁 File Details"):
                        files_df = pd.DataFrame([
                            {
                                'File': name,
                                'Size (MB)': f"{info.get('size_mb', 0):.2f}",
                                'Records': f"{info.get('records', 0):,}" if info.get('records') else "N/A",
                                'Columns': info.get('columns', 'N/A')
                            }
                            for name, info in files_created.items()
                        ])
                        st.dataframe(files_df, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error loading data summary: {e}")
    
    def init_predictor(self):
        """Initialize the affinity predictor"""
        if not hasattr(self, 'predictor'):
            self.predictor = AffinityPredictor()
            if not self.predictor.load_data():
                st.error("❌ Could not load prediction models. Please ensure data processing and model training are complete.")
                return False
        return True

    def render_molecule(self, smiles: str, size: tuple = (400, 400)) -> str:
        """Render molecule as SVG image"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Draw molecule
            d2d = Draw.rdDepictor.Compute2DCoords(mol)
            drawer = Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            
            return svg.replace('svg:', '')
        except Exception as e:
            st.error(f"Error rendering molecule: {e}")
            return None

    def render_molecule_3d(self, smiles: str, size: tuple = (400, 400)) -> str:
        """Generate 3D visualization of molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Convert to PDB format
            pdb = Chem.MolToPDBBlock(mol)
            
            # Create py3Dmol view
            view = py3Dmol.view(width=size[0], height=size[1])
            view.addModel(pdb, "pdb")
            view.setStyle({'stick':{}})
            view.zoomTo()
            
            return view.render()
        except Exception as e:
            st.error(f"Error rendering 3D molecule: {e}")
            return None

    def show_predictions_page(self):
        """Show predictions page with enhanced visuals"""
        st.markdown('<div class="sub-header">🔬 Predictions</div>', unsafe_allow_html=True)
        
        if not self.init_predictor():
            return
        
        # Model Status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖 Model Status</h3>
                <p>Ready for predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Database</h3>
                <p>Real BindingDB data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Accuracy</h3>
                <p>R² > 0.35 on test data</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Prediction Interface
        tab1, tab2, tab3 = st.tabs(["Smart Binding", "Single Prediction", "Batch Prediction"])
        
        with tab1:
            st.markdown("### 🧠 Smart Binding Recommendations")
            st.markdown("""
            Enter a target protein to find the most promising ligands from our database.
            The AI will analyze similar proteins and their known interactions to recommend
            the best potential binders.
            """)
            
            # Get database statistics and suggestions
            if hasattr(self, 'predictor'):
                suggestions = self.predictor.get_search_suggestions()
                if suggestions:
                    # Show database stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Proteins", f"{suggestions['total_proteins']:,}")
                    with col2:
                        st.metric("Total Ligands", f"{suggestions['total_ligands']:,}")
                    
                    # Show protein families in expandable section
                    with st.expander("🔍 Available Protein Families", expanded=True):
                        st.markdown("Click any example to use it in your search:")
                        
                        for family, examples in suggestions['protein_families'].items():
                            st.markdown(f"**{family}**")
                            cols = st.columns(len(examples))
                            for i, example in enumerate(examples):
                                with cols[i]:
                                    if st.button(
                                        example, 
                                        key=f"protein_{family}_{i}",
                                        help=f"Search for {example}"
                                    ):
                                        st.session_state.protein_search = example
                                        st.rerun()
            
            # Search box
            target_protein = st.text_input(
                "Target Protein",
                value=st.session_state.get('protein_search', ''),
                placeholder="e.g., Protein kinase A",
                help="Enter the name of your target protein"
            )
            
            if st.button("🔍 Find Best Binders", use_container_width=True):
                if target_protein:
                    with st.spinner("🤔 Analyzing database for best binders..."):
                        recommendations = self.predictor.recommend_ligands(target_protein)
                        
                        if recommendations:
                            st.success(f"Found {len(recommendations)} potential binders!")
                            
                            for i, rec in enumerate(recommendations, 1):
                                with st.expander(f"🔬 Recommendation #{i} - {rec['binding_strength']} Binding", expanded=i==1):
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        # Show 2D structure
                                        svg = self.render_molecule(rec['smiles'])
                                        if svg:
                                            st.markdown(f"""
                                            <div style="text-align: center;">
                                                <h4>2D Structure</h4>
                                                {svg}
                                            </div>
                                            """, unsafe_allow_html=True)
                                    
                                    with col2:
                                        # Show 3D structure
                                        st.markdown("<h4>3D Structure</h4>", unsafe_allow_html=True)
                                        viewer = self.render_molecule_3d(rec['smiles'])
                                        if viewer:
                                            st.components.html(viewer, height=400)
                                    
                                    # Show metrics with animation
                                    st.markdown("""
                                    <style>
                                    @keyframes slideIn {
                                        from { transform: translateX(-100%); opacity: 0; }
                                        to { transform: translateX(0); opacity: 1; }
                                    }
                                    .metric-row > div {
                                        animation: slideIn 0.5s ease-out forwards;
                                    }
                                    .metric-row > div:nth-child(2) {
                                        animation-delay: 0.1s;
                                    }
                                    .metric-row > div:nth-child(3) {
                                        animation-delay: 0.2s;
                                    }
                                    </style>
                                    """, unsafe_allow_html=True)
                                    
                                    col1, col2, col3 = st.columns(3)
                                    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
                                    
                                    with col1:
                                        st.metric(
                                            "Binding Affinity (pKd)", 
                                            f"{rec['binding_affinity']:.2f}",
                                            help="Higher values indicate stronger binding"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            "Confidence", 
                                            f"{rec['confidence']*100:.1f}%",
                                            help="Based on protein similarity and data quality"
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            "Similar Protein", 
                                            f"{rec['similarity_score']*100:.1f}%",
                                            help="How similar this protein is to your target"
                                        )
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Show SMILES with copy button
                                    st.markdown("##### Molecule SMILES")
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.code(rec['smiles'], language=None)
                                    with col2:
                                        if st.button("📋 Copy", key=f"copy_{i}"):
                                            st.write("Copied to clipboard!")
                                    
                                    # Show similar protein
                                    st.info(f"💡 Based on known binding with: {rec['target_protein']}")
                        else:
                            st.warning("""
                            No recommendations found. Try:
                            - Using a more general protein family name (e.g., 'kinase' instead of a specific kinase)
                            - Checking for typos
                            - Using one of the suggested protein families above
                            """)
                else:
                    st.error("Please enter a target protein name")
        
        with tab2:
            st.markdown("### 🎯 Single Molecule Prediction")
            st.markdown("""
            Predict binding affinity for a specific ligand-protein pair.
            Enter the SMILES string of your ligand and the target protein name.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                smiles = st.text_input(
                    "SMILES String",
                    placeholder="e.g., CCO",
                    help="Enter a SMILES string for the ligand"
                )
                
                if smiles:
                    svg = self.render_molecule(smiles)
                    if svg:
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <h4>Structure Preview</h4>
                            {svg}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                protein = st.text_input(
                    "Target Protein",
                    placeholder="e.g., Protein kinase A",
                    help="Enter the target protein name"
                )
                
                if protein:
                    similar = self.predictor.find_similar_proteins(protein, top_k=3)
                    if similar:
                        st.markdown("##### Similar proteins in database:")
                        for p, score in similar:
                            st.markdown(f"- {p} ({score*100:.1f}% match)")
            
            if st.button("🔍 Predict Binding Affinity", use_container_width=True):
                if smiles and protein:
                    with st.spinner("🧪 Calculating binding affinity..."):
                        result = self.predictor.predict_binding(smiles, protein)
                        
                        if result:
                            # Show results with animation
                            st.markdown("""
                            <style>
                            @keyframes fadeIn {
                                from { opacity: 0; transform: translateY(20px); }
                                to { opacity: 1; transform: translateY(0); }
                            }
                            .prediction-result {
                                animation: fadeIn 0.5s ease-out;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div class="prediction-result">
                                <div class="success-card">
                                    <h3>🎯 Prediction Result</h3>
                                    <p><strong>Binding Strength:</strong> {}</p>
                                    <p><strong>Confidence:</strong> {:.1f}%</p>
                                </div>
                            </div>
                            """.format(
                                result['binding_strength'],
                                result['confidence'] * 100
                            ), unsafe_allow_html=True)
                            
                            # Show detailed metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("pKd", f"{result['pKd']:.2f}")
                            
                            with col2:
                                st.metric("Kd (nM)", f"{result['Kd_nM']:.2f}")
                            
                            with col3:
                                st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                            
                            # Show 3D visualization
                            st.markdown("##### 3D Molecular Visualization")
                            viewer = self.render_molecule_3d(smiles)
                            if viewer:
                                st.components.html(viewer, height=450)
                        else:
                            st.error("Could not make prediction. Please check your input.")
                else:
                    st.error("Please provide both SMILES string and protein name")
        
        with tab3:
            st.markdown("### 📊 Batch Prediction")
            st.markdown("""
            Upload a CSV file with multiple ligand-protein pairs for batch prediction.
            The file should have columns: 'Ligand SMILES' and 'Target Name'.
            """)
            
            # Show sample data
            with st.expander("📋 View Sample Format"):
                sample_data = pd.DataFrame({
                    'Ligand SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O'],
                    'Target Name': ['Protein kinase A', 'GABA receptor', 'Ion channel']
                })
                st.dataframe(sample_data)
                
                # Download sample
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Sample CSV",
                    csv,
                    "sample_batch.csv",
                    "text/csv"
                )
            
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload a CSV file with required columns"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    required_cols = ['Ligand SMILES', 'Target Name']
                    
                    if all(col in df.columns for col in required_cols):
                        st.markdown("**📋 Data Preview:**")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        if st.button("🔍 Run Batch Prediction", use_container_width=True):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            with st.spinner("Processing batch predictions..."):
                                results = self.predictor.batch_predict(df)
                                
                                if results is not None:
                                    st.success("🎉 Batch prediction completed!")
                                    
                                    # Show results
                                    st.markdown("### 📊 Results")
                                    
                                    # Summary statistics
                                    summary = pd.DataFrame({
                                        'Binding Strength': results['binding_strength'].value_counts()
                                    })
                                    
                                    # Create donut chart
                                    fig = go.Figure(data=[go.Pie(
                                        labels=summary.index,
                                        values=summary['Binding Strength'],
                                        hole=0.4
                                    )])
                                    fig.update_layout(title="Binding Strength Distribution")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show detailed results
                                    st.dataframe(results, use_container_width=True)
                                    
                                    # Download results
                                    csv = results.to_csv(index=False)
                                    st.download_button(
                                        "📥 Download Results",
                                        csv,
                                        f"binding_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        "text/csv"
                                    )
                                else:
                                    st.error("Error processing predictions")
                    else:
                        st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    def show_analysis_page(self):
        """Show analysis and results page"""
        st.markdown('<div class="sub-header">📈 Analysis & Results</div>', unsafe_allow_html=True)
        
        if not st.session_state.system_status.get('models_trained'):
            st.warning("⚠️ No trained models found. Please complete the data pipeline first.")
            return
        
        # Load model metrics
        try:
            metrics_file = self.results_dir / "model_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Model Performance Comparison
                st.markdown("### 🏆 Model Performance Comparison")
                
                # Create comparison dataframe
                comparison_data = []
                for model_name, model_data in metrics.items():
                    test_metrics = model_data['test_metrics']
                    comparison_data.append({
                        'Model': model_name,
                        'R² Score': test_metrics['r2_score'],
                        'RMSE': test_metrics['rmse'],
                        'MAE': test_metrics.get('mae', 'N/A'),
                        'Status': '✅ Trained'
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display metrics table
                st.dataframe(comparison_df, use_container_width=True)
                
                # Create performance visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # R² Score comparison
                    fig_r2 = px.bar(
                        comparison_df,
                        x='Model',
                        y='R² Score',
                        title='R² Score Comparison',
                        color='R² Score',
                        color_continuous_scale='viridis'
                    )
                    fig_r2.update_layout(height=400)
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                with col2:
                    # RMSE comparison
                    fig_rmse = px.bar(
                        comparison_df,
                        x='Model',
                        y='RMSE',
                        title='RMSE Comparison (Lower is Better)',
                        color='RMSE',
                        color_continuous_scale='viridis_r'
                    )
                    fig_rmse.update_layout(height=400)
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                # Best model highlight
                best_model = comparison_df.loc[comparison_df['R² Score'].idxmax()]
                st.markdown(f"""
                <div class="success-card">
                    <h3>🏆 Best Performing Model</h3>
                    <p><strong>Model:</strong> {best_model['Model']}</p>
                    <p><strong>R² Score:</strong> {best_model['R² Score']:.4f}</p>
                    <p><strong>RMSE:</strong> {best_model['RMSE']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Data Statistics
                st.markdown("### 📊 Training Data Statistics")
                
                try:
                    # Load processed data for analysis
                    features_file = self.data_dir / "processed" / "processed_features.csv"
                    target_file = self.data_dir / "processed" / "target_values.csv"
                    
                    if features_file.exists() and target_file.exists():
                        features = pd.read_csv(features_file)
                        target = pd.read_csv(target_file)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Samples", len(features))
                        
                        with col2:
                            st.metric("Features", len(features.columns))
                        
                        with col3:
                            st.metric("Target Mean", f"{target.iloc[:, 0].mean():.3f}")
                        
                        with col4:
                            st.metric("Target Std", f"{target.iloc[:, 0].std():.3f}")
                        
                        # Target distribution
                        st.markdown("#### 🎯 Target Distribution")
                        
                        fig_dist = px.histogram(
                            target,
                            x=target.columns[0],
                            nbins=50,
                            title="Binding Affinity Distribution",
                            labels={'x': 'Binding Affinity (pKd/pKi)', 'y': 'Frequency'}
                        )
                        fig_dist.update_layout(height=400)
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Feature importance (if available)
                        st.markdown("#### 🔍 Feature Analysis")
                        
                        # Show feature correlation with target
                        if len(features.columns) <= 20:  # Only for reasonable number of features
                            corr_with_target = features.corrwith(target.iloc[:, 0]).abs().sort_values(ascending=False)
                            
                            fig_corr = px.bar(
                                x=corr_with_target.index,
                                y=corr_with_target.values,
                                title="Feature Correlation with Target (Absolute)",
                                labels={'x': 'Features', 'y': 'Correlation'}
                            )
                            fig_corr.update_layout(height=400)
                            st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Feature statistics
                        with st.expander("📋 Feature Statistics"):
                            st.dataframe(features.describe(), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error loading training data: {e}")
                
            else:
                st.error("❌ Model metrics file not found")
                
        except Exception as e:
            st.error(f"❌ Error loading analysis data: {e}")
    
    def show_settings_page(self):
        """Show settings and configuration page"""
        st.markdown('<div class="sub-header">⚙️ Settings & Configuration</div>', unsafe_allow_html=True)
        
        # System Information
        st.markdown("### 🖥️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Project Root:** {self.project_root}")
            st.info(f"**CLI Script:** {self.cli_script}")
            st.info(f"**Data Directory:** {self.data_dir}")
        
        with col2:
            st.info(f"**Models Directory:** {self.models_dir}")
            st.info(f"**Results Directory:** {self.results_dir}")
            st.info(f"**CLI Available:** {'✅' if self.cli_script.exists() else '❌'}")
        
        st.markdown("---")
        
        # Configuration Options
        st.markdown("### 🔧 Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data Processing")
            
            default_sample_size = st.number_input("Default Sample Size", 1000, 100000, 5000)
            default_max_rows = st.number_input("Default Max BindingDB Rows", 1000, 1000000, 50000)
            default_test_size = st.slider("Default Test Size", 0.1, 0.5, 0.2, 0.05)
        
        with col2:
            st.markdown("#### 🤖 Model Training")
            
            default_models = st.multiselect(
                "Default Models",
                ["RandomForest", "XGBoost", "NeuralNetwork"],
                default=["RandomForest", "XGBoost"]
            )
            
            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1
            )
        
        if st.button("💾 Save Configuration"):
            st.success("✅ Configuration saved (placeholder)")
        
        st.markdown("---")
        
        # Maintenance Actions
        st.markdown("### 🧹 Maintenance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🗑️ Clean Up")
            
            if st.button("🧹 Clear Processing Logs"):
                st.session_state.processing_log = []
                st.success("✅ Processing logs cleared")
            
            if st.button("🔄 Reset System Status"):
                self.load_system_status()
                st.success("✅ System status refreshed")
        
        with col2:
            st.markdown("#### 📁 Data Management")
            
            if st.button("📋 Show Data Summary"):
                try:
                    summary_file = self.data_dir / "processed" / "data_summary.json"
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                        st.json(summary)
                    else:
                        st.warning("⚠️ No data summary found")
                except Exception as e:
                    st.error(f"❌ Error loading data summary: {e}")
            
            if st.button("🗂️ List Model Files"):
                if self.models_dir.exists():
                    model_files = list(self.models_dir.glob("*"))
                    if model_files:
                        for file in model_files:
                            st.text(f"📄 {file.name}")
                    else:
                        st.warning("⚠️ No model files found")
                else:
                    st.warning("⚠️ Models directory not found")
        
        st.markdown("---")
        
        # Processing History
        if st.session_state.processing_log:
            st.markdown("### 📋 Processing History")
            
            history_df = pd.DataFrame(st.session_state.processing_log)
            
            # Show recent operations
            st.dataframe(history_df.tail(10), use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.processing_log = []
                st.rerun()
    
    def show_about_page(self):
        """Show about page"""
        st.markdown('<div class="sub-header">ℹ️ About Affinify</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Project Overview
            
            Affinify is an AI-powered platform for predicting protein-ligand binding affinity, 
            developed as a comprehensive computer science project that demonstrates the 
            application of artificial intelligence in drug discovery and computational biology.
            
            ### 🚀 Key Features
            
            - **Unified CLI Interface**: Single command-line tool for all operations
            - **Real Data Processing**: Works with actual BindingDB molecular data
            - **Multiple ML Models**: RandomForest, XGBoost, and Neural Networks
            - **Interactive Web Interface**: User-friendly Streamlit application
            - **Comprehensive Analysis**: Model performance evaluation and visualization
            - **Educational Focus**: Designed for learning and demonstration
            
            ### 🛠️ Technical Stack
            
            - **Python**: Core programming language
            - **Streamlit**: Web application framework
            - **Scikit-learn**: Machine learning algorithms
            - **XGBoost**: Gradient boosting framework
            - **Pandas**: Data manipulation and analysis
            - **Plotly**: Interactive visualizations
            - **NumPy**: Numerical computing
            """)
        
        with col2:
            st.markdown("""
            ### 👨‍💻 Project Information
            
            **Lead Developer**: Pranav Verma  
            **Class**: XII Aryabhatta  
            **School**: Lotus Valley International School  
            **Subject**: Computer Science Project  
            **Duration**: 8-10 Weeks  
            
            ### 📊 Performance Metrics
            
            - **Dataset Size**: 42,000+ protein-ligand pairs
            - **Processing Speed**: < 5 seconds per prediction
            - **Model Accuracy**: R² > 0.35 on real data
            - **Feature Count**: 12 molecular descriptors
            
            ### 🔬 Scientific Applications
            
            - **Drug Discovery**: Predict binding affinity for new compounds
            - **Lead Optimization**: Identify promising drug candidates
            - **Molecular Design**: Guide synthesis of new molecules
            - **Research**: Understand protein-ligand interactions
            
            ### 📚 Educational Value
            
            This project demonstrates:
            - Data science pipeline development
            - Machine learning model training and evaluation
            - Software engineering best practices
            - Computational biology applications
            - Web application development
            """)
        
        st.markdown("---")
        
        # Performance Statistics
        st.markdown("### 📈 Current Performance")
        
        if st.session_state.system_status.get('models_trained'):
            try:
                metrics_file = self.results_dir / "model_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    model_count = len(metrics)
                    best_r2 = max(model_data['test_metrics']['r2_score'] for model_data in metrics.values())
                    avg_rmse = np.mean([model_data['test_metrics']['rmse'] for model_data in metrics.values()])
                    
                    with col1:
                        st.metric("Models Trained", model_count)
                    
                    with col2:
                        st.metric("Best R² Score", f"{best_r2:.4f}")
                    
                    with col3:
                        st.metric("Average RMSE", f"{avg_rmse:.4f}")
                        
            except Exception as e:
                st.error(f"Error loading performance metrics: {e}")
        else:
            st.info("⚠️ Train models to see performance statistics")
        
        st.markdown("---")
        
        # Acknowledgments
        st.markdown("""
        ### 🙏 Acknowledgments
        
        Special thanks to:
        - **BindingDB**: For providing the comprehensive molecular binding database
        - **Open Source Community**: For the amazing tools and libraries
        - **Teachers and Mentors**: For guidance throughout the development process
        - **Scikit-learn**: For the robust machine learning framework
        - **Streamlit**: For the excellent web application framework
        
        ### 📝 License
        
        This project is developed for educational purposes. Please refer to individual 
        data source licenses for BindingDB and other datasets used.
        """)
    
    def show_ai_assistant_page(self):
        """Show AI Assistant page with ChatGPT-like interface"""
        # Add ChatGPT-like styling
        st.markdown("""
        <style>
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .user-message {
            display: flex;
            justify-content: flex-end;
            margin: 1rem 0;
        }
        
        .user-bubble {
            background: #10a37f;
            color: white;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
            font-size: 14px;
        }
        
        .assistant-message {
            display: flex;
            justify-content: flex-start;
            margin: 1rem 0;
        }
        
        .assistant-bubble {
            background: #f7f7f8;
            color: #374151;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
            font-size: 14px;
            border: 1px solid #e5e7eb;
        }
        
        .stTextInput > div > div > input {
            border-radius: 25px;
            border: 1px solid #d1d5db;
            padding: 12px 16px;
            background-color: white;
            color: #374151 !important;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: #9ca3af !important;
        }
        
        .stForm {
            background: transparent;
            border: none;
        }
        
        .main-container {
            background-color: #ffffff;
            min-height: 100vh;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Check if Ollama is enabled
        if not self.chat.enabled:
            st.error("🔴 AI Assistant is disabled")
            st.markdown("""
            **To enable the AI Assistant:**
            1. Install Ollama from https://ollama.ai
            2. Start Ollama: `ollama serve`
            3. Run setup: `python setup_ollama.py`
            4. Restart this application
            """)
            return
        
        # Check if Ollama is available
        if not st.session_state.ollama_available:
            # Try to check availability
            try:
                st.session_state.ollama_available = self.chat.check_ollama_availability()
            except:
                st.session_state.ollama_available = False
            
            if not st.session_state.ollama_available:
                st.error("🔴 AI Assistant not available")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("""
                    **Troubleshooting:**
                    - Ensure Ollama is running: `ollama serve`
                    - Check if model is installed: `ollama list`
                    - Install model: `ollama pull pranavverma/Affinify-AI:latest`
                    """)
                
                with col2:
                    if st.button("🔄 Retry Connection", type="primary"):
                        st.session_state.ollama_available = self.chat.check_ollama_availability()
                        st.rerun()
                return
        
        # Main chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_messages):
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <div class="user-bubble">
                        {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # If this is the last message and we're currently streaming, show placeholder
                if (i == len(st.session_state.chat_messages) - 1 and 
                    st.session_state.get('getting_response', False)):
                    # Create placeholder for streaming response
                    streaming_placeholder = st.empty()
                    st.session_state.streaming_placeholder = streaming_placeholder
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <div class="assistant-bubble">
                            {message['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show thinking indicator when getting response
        if st.session_state.get('getting_response', False):
            st.markdown("""
            <div class="assistant-message">
                <div class="assistant-bubble">
                    <em>🤔 Affinify Assistant is thinking...</em>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Close chat container
        st.markdown('</div>', unsafe_allow_html=True)
        
        
        # Create input form
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input(
                "",
                placeholder="Message Affinify Assistant..." if not st.session_state.get('getting_response', False) else "Affinify Assistant is thinking...",
                key="chat_input_field",
                disabled=st.session_state.get('getting_response', False)
            )
            
            send_button = st.form_submit_button(
                "🤔 Thinking..." if st.session_state.get('getting_response', False) else "Send", 
                type="primary", 
                use_container_width=True,
                disabled=st.session_state.get('getting_response', False)
            )
        
        # Handle message sending
        if send_button and user_input.strip():
            # Add user message to history FIRST
            self.chat.add_message('user', user_input)
            
            # Rerun to show the user message immediately
            st.rerun()
        
        # Check if we need to get a response (if last message is from user and no response yet)
        if (st.session_state.chat_messages and 
            st.session_state.chat_messages[-1]['role'] == 'user' and
            not st.session_state.get('getting_response', False)):
            
            # Set flag to prevent multiple responses
            st.session_state.getting_response = True
            
            # Get the last user message
            last_user_message = st.session_state.chat_messages[-1]['content']
            
            # Add empty assistant message to display streaming
            self.chat.add_message('assistant', "")
            
            # Rerun to show the streaming placeholder
            st.rerun()
        
        # Handle streaming if we're in the middle of getting a response
        if (st.session_state.get('getting_response', False) and 
            hasattr(st.session_state, 'streaming_placeholder') and
            st.session_state.chat_messages and
            st.session_state.chat_messages[-1]['role'] == 'assistant' and
            st.session_state.chat_messages[-1]['content'] == ""):
            
            # Get the last user message
            last_user_message = st.session_state.chat_messages[-2]['content']
            
            # Initialize streaming response
            streaming_response = ""
            
            try:
                # Get streaming response
                for chunk in self.chat.send_message_stream(last_user_message):
                    streaming_response += chunk
                    
                    # Update the last message content with streaming text
                    st.session_state.chat_messages[-1]['content'] = streaming_response
                    
                    # Display streaming response in the correct position
                    with st.session_state.streaming_placeholder.container():
                        st.markdown(f"""
                        <div class="assistant-message">
                            <div class="assistant-bubble">
                                {streaming_response}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Small delay to make streaming visible
                    time.sleep(0.05)
                
                # Final update with complete response
                st.session_state.chat_messages[-1]['content'] = streaming_response
                
            except Exception as e:
                error_msg = self.chat.config.get('error_message', 'Sorry, I encountered an error.')
                st.session_state.chat_messages[-1]['content'] = error_msg
                
                with st.session_state.streaming_placeholder.container():
                    st.markdown(f"""
                    <div class="assistant-message">
                        <div class="assistant-bubble">
                            {error_msg}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Clear the flag and placeholder
            st.session_state.getting_response = False
            if hasattr(st.session_state, 'streaming_placeholder'):
                delattr(st.session_state, 'streaming_placeholder')
            
            # Rerun to show the final state
            st.rerun()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                self.chat.clear_chat()
                st.rerun()
    
    def run(self):
        """Run the main application"""
        self.show_header()
        
        # Show sidebar and get selected page
        page = self.show_sidebar()
        
        # Route to appropriate page
        if page == "🏠 Home":
            self.show_home_page()
        elif page == "📊 Data Pipeline":
            self.show_data_pipeline_page()
        elif page == "🔬 Predictions":
            self.show_predictions_page()
        elif page == "📈 Analysis":
            self.show_analysis_page()
        elif page == "🤖 AI Assistant":
            self.show_ai_assistant_page()
        elif page == "⚙️ Settings":
            self.show_settings_page()
        elif page == "ℹ️ About":
            self.show_about_page()

def main():
    """Main application entry point"""
    app = AffinifyApp()
    app.run()

if __name__ == "__main__":
    main()