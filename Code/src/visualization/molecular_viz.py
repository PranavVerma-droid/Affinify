import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularVisualizer:
    """
    Handles molecular visualization and plotting
    """
    
    def __init__(self, style: str = 'default'):
        self.style = style
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        """Setup plotting style"""
        plt.style.use(self.style)
        sns.set_palette("husl")
        
    def plot_3d_molecule(self, smiles: str, size: Tuple[int, int] = (400, 400)) -> str:
        """
        Generate 3D molecular visualization
        
        Args:
            smiles: SMILES string
            size: Size of the visualization
            
        Returns:
            HTML string for 3D visualization
        """
        try:
            import py3Dmol
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "<p>Invalid SMILES string</p>"
            
            # Add hydrogens and generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            
            # Create 3D viewer
            viewer = py3Dmol.view(width=size[0], height=size[1])
            
            # Add molecule
            mol_block = Chem.MolToMolBlock(mol)
            viewer.addModel(mol_block, 'mol')
            
            # Set style
            viewer.setStyle({'stick': {'radius': 0.1}})
            viewer.addSurface(py3Dmol.VDW, {'opacity': 0.3})
            viewer.zoomTo()
            
            return viewer._make_html()
            
        except ImportError:
            logger.warning("py3Dmol or RDKit not available")
            return self._create_simple_molecule_plot(smiles, size)
    
    def _create_simple_molecule_plot(self, smiles: str, size: Tuple[int, int]) -> str:
        """Create a simple 2D molecular representation"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            import base64
            from io import BytesIO
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "<p>Invalid SMILES string</p>"
            
            # Generate 2D image
            img = Draw.MolToImage(mol, size=size)
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f'<img src="data:image/png;base64,{img_str}" />'
            
        except ImportError:
            return f"<p>Molecule: {smiles}</p><p>RDKit not available for visualization</p>"
    
    def plot_binding_affinity_distribution(self, binding_affinities: pd.Series, 
                                         title: str = "Binding Affinity Distribution") -> plt.Figure:
        """
        Plot distribution of binding affinities
        
        Args:
            binding_affinities: Series of binding affinity values
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(binding_affinities.dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Binding Affinity (pKd/pKi/pIC50)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{title} - Histogram')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(binding_affinities.dropna(), vert=True)
        ax2.set_ylabel('Binding Affinity (pKd/pKi/pIC50)')
        ax2.set_title(f'{title} - Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                              top_n: int = 20) -> plt.Figure:
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, (_, row) in enumerate(top_features.iterrows()):
            ax.text(row['importance'], i, f'{row["importance"]:.3f}', 
                   va='center', ha='left' if row['importance'] > 0 else 'right')
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 title: str = "Predictions vs Actual") -> plt.Figure:
        """
        Plot predicted vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        
        ax.set_xlabel('Actual Binding Affinity')
        ax.set_ylabel('Predicted Binding Affinity')
        ax.set_title(f'{title} (R² = {r2:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = "Residual Plot") -> plt.Figure:
        """
        Plot residuals
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        residuals = y_true - y_pred
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'{title} - vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{title} - Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict]) -> plt.Figure:
        """
        Compare multiple models
        
        Args:
            results: Dictionary with model results
            
        Returns:
            matplotlib Figure
        """
        model_names = list(results.keys())
        r2_scores = [results[name]['test_metrics']['r2_score'] for name in model_names]
        rmse_scores = [results[name]['test_metrics']['rmse'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # R² comparison
        bars1 = ax1.bar(model_names, r2_scores, alpha=0.7)
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Comparison - R² Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # RMSE comparison
        bars2 = ax2.bar(model_names, rmse_scores, alpha=0.7, color='orange')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Model Comparison - RMSE')
        
        # Add value labels
        for bar, score in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_learning_curve(self, train_scores: List[float], 
                          val_scores: List[float],
                          title: str = "Learning Curve") -> plt.Figure:
        """
        Plot learning curve
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_scores) + 1)
        
        ax.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2)
        ax.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_molecular_properties_plot(self, df: pd.DataFrame) -> plt.Figure:
        """
        Create molecular properties visualization
        
        Args:
            df: DataFrame with molecular properties
            
        Returns:
            matplotlib Figure
        """
        # Select key molecular properties
        properties = ['molecular_weight', 'logp', 'num_hbd', 'num_hba', 'tpsa', 'num_rings']
        available_props = [prop for prop in properties if prop in df.columns]
        
        if not available_props:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No molecular properties available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        n_props = len(available_props)
        n_cols = min(3, n_props)
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, prop in enumerate(available_props):
            ax = axes[i]
            values = df[prop].dropna()
            
            if len(values) > 0:
                ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel(prop.replace('_', ' ').title())
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {prop.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(available_props), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                              title: str = "Feature Correlation Matrix") -> plt.Figure:
        """
        Plot correlation matrix
        
        Args:
            df: DataFrame with features
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Not enough numeric features for correlation', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', ax=ax)
        
        ax.set_title(title)
        plt.tight_layout()
        return fig

class InteractivePlotter:
    """
    Create interactive plots using Plotly
    """
    
    def __init__(self):
        self.plotly_available = True
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            self.go = go
            self.px = px
        except ImportError:
            self.plotly_available = False
            logger.warning("Plotly not available. Interactive plots will be disabled.")
    
    def create_interactive_scatter(self, df: pd.DataFrame, x_col: str, y_col: str,
                                 color_col: Optional[str] = None,
                                 title: str = "Interactive Scatter Plot") -> Any:
        """
        Create interactive scatter plot
        
        Args:
            df: DataFrame with data
            x_col: X-axis column
            y_col: Y-axis column
            color_col: Column for color coding
            title: Plot title
            
        Returns:
            Plotly figure or None
        """
        if not self.plotly_available:
            return None
        
        if color_col and color_col in df.columns:
            fig = self.px.scatter(df, x=x_col, y=y_col, color=color_col,
                                title=title, hover_data=df.columns)
        else:
            fig = self.px.scatter(df, x=x_col, y=y_col, title=title,
                                hover_data=df.columns)
        
        return fig
    
    def create_interactive_histogram(self, df: pd.DataFrame, column: str,
                                   title: str = "Interactive Histogram") -> Any:
        """
        Create interactive histogram
        
        Args:
            df: DataFrame with data
            column: Column to plot
            title: Plot title
            
        Returns:
            Plotly figure or None
        """
        if not self.plotly_available:
            return None
        
        fig = self.px.histogram(df, x=column, title=title)
        return fig
    
    def create_3d_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                         color_col: Optional[str] = None,
                         title: str = "3D Scatter Plot") -> Any:
        """
        Create 3D scatter plot
        
        Args:
            df: DataFrame with data
            x_col: X-axis column
            y_col: Y-axis column
            z_col: Z-axis column
            color_col: Column for color coding
            title: Plot title
            
        Returns:
            Plotly figure or None
        """
        if not self.plotly_available:
            return None
        
        if color_col and color_col in df.columns:
            fig = self.px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                                   title=title)
        else:
            fig = self.px.scatter_3d(df, x=x_col, y=y_col, z=z_col, title=title)
        
        return fig

if __name__ == "__main__":
    # Example usage
    visualizer = MolecularVisualizer()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'molecular_weight': np.random.normal(400, 100, n_samples),
        'logp': np.random.normal(2.5, 1.5, n_samples),
        'num_hbd': np.random.poisson(3, n_samples),
        'num_hba': np.random.poisson(5, n_samples),
        'tpsa': np.random.normal(80, 30, n_samples),
        'num_rings': np.random.poisson(2, n_samples),
        'binding_affinity': np.random.normal(7, 2, n_samples)
    })
    
    # Test visualizations
    print("Creating molecular properties plot...")
    fig1 = visualizer.create_molecular_properties_plot(sample_data)
    plt.show()
    
    print("Creating binding affinity distribution plot...")
    fig2 = visualizer.plot_binding_affinity_distribution(sample_data['binding_affinity'])
    plt.show()
    
    print("Creating correlation matrix...")
    fig3 = visualizer.plot_correlation_matrix(sample_data)
    plt.show()
    
    print("Visualizations created successfully!")
