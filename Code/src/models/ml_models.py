import os
import sys
import numpy as np
import pandas as pd #type: ignore
from pathlib import Path
import logging
import joblib #type: ignore
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor #type: ignore
from sklearn.model_selection import train_test_split, cross_val_score #type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
from sklearn.base import BaseEstimator, RegressorMixin #type: ignore
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import xgboost as xgb #type: ignore
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf #type: ignore
    from tensorflow import keras #type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class RandomForestModel(BaseEstimator, RegressorMixin):
    """Random Forest model for binding affinity prediction"""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.is_trained = False
    
    def fit(self, X, y):
        """Train the Random Forest model"""
        logger.info("Training RandomForest model...")
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Store feature importance
        if hasattr(X, 'columns'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 5 important features:")
            for _, row in self.feature_importance.head().iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.feature_importance is None:
            return None
        return self.feature_importance
    
    def save_model(self, filepath):
        """Save model to file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

class XGBoostModel(BaseEstimator, RegressorMixin):
    """XGBoost model for binding affinity prediction"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 subsample=0.8, colsample_bytree=0.8, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.is_trained = False
    
    def fit(self, X, y):
        """Train the XGBoost model"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        logger.info("Training XGBoost model...")
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Store feature importance
        if hasattr(X, 'columns'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 5 important features:")
            for _, row in self.feature_importance.head().iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.feature_importance is None:
            return None
        return self.feature_importance
    
    def save_model(self, filepath):
        """Save model to file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

class NeuralNetworkModel(BaseEstimator, RegressorMixin):
    """Neural Network model for binding affinity prediction"""
    
    def __init__(self, hidden_layers=[128, 64, 32], dropout_rate=0.2, 
                 learning_rate=0.001, batch_size=32, epochs=100, random_state=42):
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.is_trained = False
    
    def fit(self, X, y):
        """Train the Neural Network model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        logger.info("Training NeuralNetwork model...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Set random seed
        tf.random.set_seed(self.random_state)
        
        # Build model
        self.model = keras.Sequential()
        
        # Input layer
        self.model.add(keras.layers.Dense(
            self.hidden_layers[0], 
            activation='relu', 
            input_shape=(X_scaled.shape[1],)
        ))
        self.model.add(keras.layers.Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            self.model.add(keras.layers.Dense(units, activation='relu'))
            self.model.add(keras.layers.Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(keras.layers.Dense(1, activation='linear'))
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = self.model.fit(
            X_scaled, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            verbose=0
        )
        
        self.is_trained = True
        
        # Log final metrics
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        logger.info(f"Final training loss: {final_loss:.4f}")
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()
    
    def save_model(self, filepath):
        """Save model to file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save both the model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

class EnsembleModel(BaseEstimator, RegressorMixin):
    """Ensemble model combining multiple models"""
    
    def __init__(self, models=None, weights=None):
        self.models = models or []
        self.weights = weights
        self.is_trained = False
    
    def add_model(self, model):
        """Add a model to the ensemble"""
        self.models.append(model)
    
    def fit(self, X, y):
        """Train ensemble models and calculate weights"""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Train all models
        trained_models = []
        for model in self.models:
            model.fit(X, y)
            trained_models.append(model)
        
        self.models = trained_models
        
        # Calculate weights based on cross-validation performance
        if self.weights is None:
            self._calculate_weights(X, y)
        
        self.is_trained = True
        return self
    
    def _calculate_weights(self, X, y):
        """Calculate model weights based on cross-validation performance"""
        model_scores = []
        
        for model in self.models:
            try:
                # Use the actual sklearn model for cross-validation
                if hasattr(model, 'model'):
                    sklearn_model = model.model
                else:
                    sklearn_model = model
                
                scores = cross_val_score(sklearn_model, X, y, cv=3, scoring='r2')
                model_scores.append(np.mean(scores))
            except Exception as e:
                logger.warning(f"Error calculating cross-validation score for model: {e}")
                model_scores.append(0.0)
        
        # Convert scores to weights (higher score = higher weight)
        scores = np.array(model_scores)
        # Ensure no negative weights
        scores = np.maximum(scores, 0.1)
        self.weights = scores / np.sum(scores)
        
        logger.info(f"Ensemble weights: {self.weights}")
    
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred

class ModelTrainer:
    """Trainer class for managing multiple models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """Train multiple models and compare performance"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Initialize models
        models = {
            'RandomForest': RandomForestModel(random_state=random_state),
            'XGBoost': XGBoostModel(random_state=random_state) if XGBOOST_AVAILABLE else None,
            'NeuralNetwork': NeuralNetworkModel(random_state=random_state, epochs=50) if TENSORFLOW_AVAILABLE else None
        }
        
        # Remove None models
        models = {k: v for k, v in models.items() if v is not None}
        
        results = {}
        
        # Train individual models
        for name, model in models.items():
            try:
                logger.info(f"\nTraining {name} model...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                # Save model
                model_path = Path("models") / f"{name.lower()}_model.pkl"
                model.save_model(model_path)
                
                # Store results
                results[name] = {
                    'model': model,
                    'train_metrics': {
                        'r2_score': train_r2,
                        'rmse': np.sqrt(mean_squared_error(y_train, train_pred))
                    },
                    'test_metrics': {
                        'r2_score': test_r2,
                        'rmse': test_rmse
                    }
                }
                
                logger.info(f"{name} Results:")
                logger.info(f"  Train R²: {train_r2:.4f}")
                logger.info(f"  Test R²: {test_r2:.4f}")
                logger.info(f"  Test RMSE: {test_rmse:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        # Train ensemble model
        if len(results) > 1:
            try:
                logger.info("Training ensemble models...")
                
                ensemble = EnsembleModel()
                for name, result in results.items():
                    # Create fresh model instance for ensemble
                    if name == 'RandomForest':
                        fresh_model = RandomForestModel(random_state=random_state)
                    elif name == 'XGBoost':
                        fresh_model = XGBoostModel(random_state=random_state)
                    elif name == 'NeuralNetwork':
                        fresh_model = NeuralNetworkModel(random_state=random_state, epochs=50)
                    
                    ensemble.add_model(fresh_model)
                
                ensemble.fit(X_train, y_train)
                
                # Make predictions
                ensemble_pred = ensemble.predict(X_test)
                ensemble_r2 = r2_score(y_test, ensemble_pred)
                ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                
                results['Ensemble'] = {
                    'model': ensemble,
                    'test_metrics': {
                        'r2_score': ensemble_r2,
                        'rmse': ensemble_rmse
                    }
                }
                
                logger.info(f"Ensemble Results:")
                logger.info(f"  Test R²: {ensemble_r2:.4f}")
                logger.info(f"  Test RMSE: {ensemble_rmse:.4f}")
                
            except Exception as e:
                logger.error(f"Error training ensemble: {str(e)}")
        
        # Find best model
        best_r2 = -float('inf')
        for name, result in results.items():
            test_r2 = result['test_metrics']['r2_score']
            if test_r2 > best_r2:
                best_r2 = test_r2
                self.best_model_name = name
                self.best_model = result['model']
        
        self.results = results
        return results
    
    def get_best_model(self):
        """Get the best performing model"""
        if self.best_model is None:
            return None, None
        return self.best_model_name, self.best_model
    
    def predict(self, X, model_name=None):
        """Make predictions using specified model or best model"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.results[model_name]['model']
        return model.predict(X)