import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available")

logger = logging.getLogger(__name__)

class PlotGenerator:
    """Class for generating various plots and visualizations."""
    
    def __init__(self, style: str = 'plotly_white'):
        """Initialize plot generator with specified style."""
        self.style = style
        self.color_palette = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#5E548E',
            'warning': '#F4A261',
            'light': '#E9C46A',
            'dark': '#264653'
        }
    
    def binding_affinity_scatter(self, df: pd.DataFrame, 
                               x_col: str = 'mol_weight', 
                               y_col: str = 'binding_affinity_nm',
                               color_col: str = 'protein_name',
                               title: Optional[str] = None) -> Any:
        """Create scatter plot of binding affinity vs molecular property."""
        if not PLOTLY_AVAILABLE:
            return self._matplotlib_scatter(df, x_col, y_col, color_col, title)
        
        if title is None:
            title = f"Binding Affinity vs {x_col.replace('_', ' ').title()}"
        
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            color=color_col,
            title=title,
            labels={
                x_col: x_col.replace('_', ' ').title(),
                y_col: 'Binding Affinity (nM)',
                color_col: color_col.replace('_', ' ').title()
            },
            hover_data=['ligand_smiles'] if 'ligand_smiles' in df.columns else None,
            template=self.style
        )
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title='Binding Affinity (nM)',
            height=500
        )
        
        # Log scale for binding affinity if values span large range
        if df[y_col].max() / df[y_col].min() > 100:
            fig.update_yaxes(type="log")
        
        return fig
    
    def binding_affinity_heatmap(self, df: pd.DataFrame, 
                               protein_col: str = 'protein_name',
                               activity_col: str = 'activity_type',
                               value_col: str = 'pKd',
                               title: Optional[str] = None) -> Any:
        """Create heatmap of binding affinities across proteins and activity types."""
        if not PLOTLY_AVAILABLE:
            return self._matplotlib_heatmap(df, protein_col, activity_col, value_col, title)
        
        # Create pivot table for heatmap
        pivot_df = df.pivot_table(
            values=value_col, 
            index=protein_col, 
            columns=activity_col, 
            aggfunc='mean'
        )
        
        if title is None:
            title = f"Average {value_col} by Protein and Activity Type"
        
        fig = px.imshow(
            pivot_df,
            title=title,
            labels=dict(
                x="Activity Type",
                y="Protein",
                color="pKd Value"
            ),
            color_continuous_scale='Viridis',
            template=self.style
        )
        
        fig.update_layout(height=500)
        return fig
    
    def molecular_property_scatter(self, df: pd.DataFrame,
                                 properties: List[str] = ['mol_weight', 'logp', 'tpsa'],
                                 title: Optional[str] = None) -> Any:
        """Create scatter plot matrix of molecular properties."""
        if not PLOTLY_AVAILABLE:
            return self._matplotlib_scatter_matrix(df, properties, title)
        
        if title is None:
            title = "Molecular Properties Scatter Matrix"
        
        # Select only the specified properties
        plot_df = df[properties + ['binding_affinity_nm']].copy()
        
        fig = px.scatter_matrix(
            plot_df,
            dimensions=properties,
            color='binding_affinity_nm',
            title=title,
            template=self.style,
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=600)
        
        return fig
    
    def model_performance_comparison(self, results_df: pd.DataFrame,
                                  title: Optional[str] = None) -> Any:
        """Create bar chart comparing model performance."""
        if not PLOTLY_AVAILABLE:
            return self._matplotlib_bar_comparison(results_df, title)
        
        if title is None:
            title = "Model Performance Comparison"
        
        # Create subplots for different metrics
        metrics = ['R²', 'RMSE', 'MAE']
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=metrics,
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = [self.color_palette['primary'], self.color_palette['secondary'], self.color_palette['accent']]
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=results_df['Model'],
                        y=results_df[metric],
                        name=metric,
                        marker_color=colors[i % len(colors)]
                    ),
                    row=1, col=i+1
                )
        
        fig.update_layout(
            title_text=title,
            showlegend=False,
            template=self.style,
            height=400
        )
        
        return fig
    
    def prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                           title: Optional[str] = None) -> Any:
        """Create prediction vs actual values plot."""
        if not PLOTLY_AVAILABLE:
            return self._matplotlib_pred_vs_actual(y_true, y_pred, title)
        
        if title is None:
            title = "Predicted vs Actual Binding Affinity"
        
        # Calculate R² for display
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(
                color=self.color_palette['primary'],
                opacity=0.6
            )
        ))
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"{title} (R² = {r2:.3f})",
            xaxis_title="Actual pKd",
            yaxis_title="Predicted pKd",
            template=self.style,
            height=500
        )
        
        return fig
    
    def feature_importance_plot(self, importance_df: pd.DataFrame,
                              top_n: int = 15,
                              title: Optional[str] = None) -> Any:
        """Create feature importance bar plot."""
        if not PLOTLY_AVAILABLE:
            return self._matplotlib_feature_importance(importance_df, top_n, title)
        
        if title is None:
            title = "Feature Importance"
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color=self.color_palette['accent']
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Features",
            template=self.style,
            height=500,
            yaxis=dict(autorange="reversed")  # Highest importance at top
        )
        
        return fig
    
    def residuals_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                      title: Optional[str] = None) -> Any:
        """Create residuals plot."""
        if not PLOTLY_AVAILABLE:
            return self._matplotlib_residuals(y_true, y_pred, title)
        
        if title is None:
            title = "Residuals Plot"
        
        residuals = y_true - y_pred
        
        fig = go.Figure()
        
        # Residuals scatter
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                color=self.color_palette['secondary'],
                opacity=0.6
            )
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            template=self.style,
            height=400
        )
        
        return fig
    
    # Matplotlib fallback methods
    def _matplotlib_scatter(self, df, x_col, y_col, color_col, title):
        """Matplotlib fallback for scatter plot."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        plt.figure(figsize=(10, 6))
        
        if color_col in df.columns:
            unique_colors = df[color_col].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_colors)))
            
            for i, category in enumerate(unique_colors):
                subset = df[df[color_col] == category]
                plt.scatter(subset[x_col], subset[y_col], 
                           label=category, c=[colors[i]], alpha=0.6)
            plt.legend()
        else:
            plt.scatter(df[x_col], df[y_col], alpha=0.6)
        
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel('Binding Affinity (nM)')
        plt.title(title)
        
        if df[y_col].max() / df[y_col].min() > 100:
            plt.yscale('log')
        
        plt.tight_layout()
        return plt.gcf()
    
    def _matplotlib_heatmap(self, df, protein_col, activity_col, value_col, title):
        """Matplotlib fallback for heatmap."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        pivot_df = df.pivot_table(
            values=value_col, 
            index=protein_col, 
            columns=activity_col, 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    def _matplotlib_scatter_matrix(self, df, properties, title):
        """Matplotlib fallback for scatter matrix."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        from pandas.plotting import scatter_matrix
        
        plot_df = df[properties + ['binding_affinity_nm']].copy()
        fig = plt.figure(figsize=(12, 10))
        scatter_matrix(plot_df, alpha=0.6, figsize=(12, 10), diagonal='hist')
        plt.suptitle(title)
        return fig
    
    def _matplotlib_bar_comparison(self, results_df, title):
        """Matplotlib fallback for bar comparison."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        metrics = ['R²', 'RMSE', 'MAE']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                axes[i].bar(results_df['Model'], results_df[metric])
                axes[i].set_title(metric)
                axes[i].set_ylabel(metric)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def _matplotlib_pred_vs_actual(self, y_true, y_pred, title):
        """Matplotlib fallback for prediction vs actual."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        return plt.gcf()
    
    def _matplotlib_feature_importance(self, importance_df, top_n, title):
        """Matplotlib fallback for feature importance."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        return plt.gcf()
    
    def _matplotlib_residuals(self, y_true, y_pred, title):
        """Matplotlib fallback for residuals plot."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        residuals = y_true - y_pred
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()


def create_model_performance_dashboard(results_dict: Dict[str, Dict[str, float]]) -> Any:
    """Create comprehensive model performance dashboard."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available for dashboard creation")
        return None
    
    # Convert results to DataFrame
    results_data = []
    for model_name, metrics in results_dict.items():
        result = {'Model': model_name}
        result.update(metrics)
        results_data.append(result)
    
    results_df = pd.DataFrame(results_data)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['R² Score', 'RMSE', 'MAE', 'Model Comparison'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5E548E']
    
    # R² Score
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['r2'], 
               marker_color=colors[0], name='R²'),
        row=1, col=1
    )
    
    # RMSE
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['rmse'], 
               marker_color=colors[1], name='RMSE'),
        row=1, col=2
    )
    
    # MAE
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['mae'], 
               marker_color=colors[2], name='MAE'),
        row=2, col=1
    )
    
    # Combined comparison (normalized metrics)
    normalized_df = results_df.copy()
    normalized_df['r2_norm'] = normalized_df['r2'] / normalized_df['r2'].max()
    normalized_df['rmse_norm'] = 1 - (normalized_df['rmse'] / normalized_df['rmse'].max())
    normalized_df['mae_norm'] = 1 - (normalized_df['mae'] / normalized_df['mae'].max())
    normalized_df['combined_score'] = (normalized_df['r2_norm'] + 
                                     normalized_df['rmse_norm'] + 
                                     normalized_df['mae_norm']) / 3
    
    fig.add_trace(
        go.Bar(x=normalized_df['Model'], y=normalized_df['combined_score'], 
               marker_color=colors[3], name='Combined Score'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Model Performance Dashboard",
        showlegend=False,
        height=600,
        template='plotly_white'
    )
    
    return fig
