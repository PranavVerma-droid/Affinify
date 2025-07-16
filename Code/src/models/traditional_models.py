import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

logger = logging.getLogger(__name__)

class BindingAffinityPredictor:
    """Traditional machine learning models for binding affinity prediction."""
    
    def __init__(self, model_type: str = 'rf'):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model ('rf', 'xgb', 'svr')
        """
        self.model_type = model_type.lower()
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
        # Initialize model based on type
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgb' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svr':
            self.model = SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_type.upper()} model with {len(X_train)} samples")
        
        # Store feature columns
        self.feature_columns = list(X_train.columns)
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_train)
        metrics = self.calculate_metrics(y_train, y_pred_train)
        
        logger.info(f"Training complete. R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.3f}")
        return metrics
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure feature columns match
        if self.feature_columns:
            X_test = X_test[self.feature_columns]
        
        predictions = self.model.predict(X_test)
        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        y_pred = self.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        logger.info(f"Evaluation complete. R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.3f}")
        return metrics
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance scores."""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning(f"Feature importance not available for {self.model_type}")
            return None
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")


class EnsemblePredictor:
    """Ensemble of multiple models for improved prediction."""
    
    def __init__(self, model_types: List[str] = ['rf', 'xgb', 'svr']):
        """
        Initialize ensemble predictor.
        
        Args:
            model_types: List of model types to include in ensemble
        """
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
        # Initialize individual models
        for model_type in model_types:
            try:
                self.models[model_type] = BindingAffinityPredictor(model_type)
                self.weights[model_type] = 1.0  # Equal weights initially
            except ValueError as e:
                logger.warning(f"Skipping {model_type}: {e}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Dict[str, float]]:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training metrics for each model
        """
        logger.info(f"Training ensemble with {len(self.models)} models")
        
        all_metrics = {}
        
        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} model...")
            metrics = model.train(X_train, y_train)
            all_metrics[model_name] = metrics
        
        # Calculate weights based on validation performance if available
        if X_val is not None and y_val is not None:
            self._calculate_weights(X_val, y_val)
        
        self.is_trained = True
        logger.info("Ensemble training complete")
        
        return all_metrics
    
    def _calculate_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Calculate ensemble weights based on validation performance."""
        val_scores = {}
        
        for model_name, model in self.models.items():
            metrics = model.evaluate(X_val, y_val)
            val_scores[model_name] = metrics['r2']  # Use R² for weighting
        
        # Convert R² scores to weights (higher R² = higher weight)
        total_score = sum(max(0, score) for score in val_scores.values())
        
        if total_score > 0:
            for model_name in self.models:
                self.weights[model_name] = max(0, val_scores[model_name]) / total_score
        
        logger.info(f"Ensemble weights: {self.weights}")
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            Weighted ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X_test)
        
        # Calculate weighted average
        ensemble_pred = np.zeros(len(X_test))
        total_weight = sum(self.weights.values())
        
        for model_name, pred in predictions.items():
            weight = self.weights[model_name] / total_weight
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        y_pred = self.predict(X_test)
        metrics = BindingAffinityPredictor.calculate_metrics(y_test, y_pred)
        
        logger.info(f"Ensemble evaluation - R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.3f}")
        return metrics
    
    def get_individual_predictions(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual models."""
        predictions = {}
        
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X_test)
        
        return predictions
    
    def save_ensemble(self, directory: str):
        """Save the entire ensemble."""
        os.makedirs(directory, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = os.path.join(directory, f"{model_name}_model.joblib")
            model.save_model(model_path)
        
        # Save ensemble metadata
        ensemble_data = {
            'weights': self.weights,
            'is_trained': self.is_trained,
            'model_names': list(self.models.keys())
        }
        
        ensemble_path = os.path.join(directory, "ensemble_metadata.joblib")
        joblib.dump(ensemble_data, ensemble_path)
        
        logger.info(f"Ensemble saved to {directory}")
    
    def load_ensemble(self, directory: str):
        """Load a saved ensemble."""
        # Load ensemble metadata
        ensemble_path = os.path.join(directory, "ensemble_metadata.joblib")
        ensemble_data = joblib.load(ensemble_path)
        
        self.weights = ensemble_data['weights']
        self.is_trained = ensemble_data['is_trained']
        model_names = ensemble_data['model_names']
        
        # Load individual models
        self.models = {}
        for model_name in model_names:
            model_path = os.path.join(directory, f"{model_name}_model.joblib")
            if os.path.exists(model_path):
                self.models[model_name] = BindingAffinityPredictor(model_name)
                self.models[model_name].load_model(model_path)
        
        logger.info(f"Ensemble loaded from {directory}")


def compare_models(X_train: pd.DataFrame, y_train: pd.Series, 
                  X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Compare performance of different models."""
    model_types = ['rf', 'svr']
    if XGBOOST_AVAILABLE:
        model_types.append('xgb')
    
    results = []
    
    for model_type in model_types:
        logger.info(f"Evaluating {model_type.upper()} model...")
        
        # Train model
        model = BindingAffinityPredictor(model_type)
        model.train(X_train, y_train)
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test)
        
        # Store results
        result = {
            'Model': model_type.upper(),
            'R²': test_metrics['r2'],
            'RMSE': test_metrics['rmse'],
            'MAE': test_metrics['mae']
        }
        results.append(result)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    
    logger.info("Model comparison complete")
    return comparison_df
