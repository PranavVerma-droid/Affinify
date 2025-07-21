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
from sklearn.feature_extraction.text import TfidfVectorizer #type: ignore
from sklearn.metrics.pairwise import cosine_similarity #type: ignore
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

import json

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
    
    def train_random_forest(self, features, target, test_size=0.2, random_state=42):
        """Train only Random Forest model"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        # Train Random Forest
        rf_model = RandomForestModel(n_estimators=100, random_state=random_state)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        test_metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred) ** 0.5
        }
        
        # Save model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_file = models_dir / "randomforest_model.pkl"
        rf_model.save_model(model_file)
        
        return {
            'model': rf_model,
            'test_metrics': test_metrics,
            'model_file': str(model_file)
        }
    
    def train_xgboost(self, features, target, test_size=0.2, random_state=42):
        """Train only XGBoost model"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        # Train XGBoost
        xgb_model = XGBoostModel(n_estimators=100, random_state=random_state)
        xgb_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = xgb_model.predict(X_test)
        
        # Calculate metrics
        test_metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred) ** 0.5
        }
        
        # Save model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_file = models_dir / "xgboost_model.pkl"
        xgb_model.save_model(model_file)
        
        return {
            'model': xgb_model,
            'test_metrics': test_metrics,
            'model_file': str(model_file)
        }
    
    def train_neural_network(self, features, target, test_size=0.2, random_state=42):
        """Train only Neural Network model"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        # Train Neural Network
        nn_model = NeuralNetworkModel(hidden_layers=[128, 64], random_state=random_state)
        nn_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = nn_model.predict(X_test)
        
        # Calculate metrics
        test_metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred) ** 0.5
        }
        
        # Save model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_file = models_dir / "neuralnetwork_model.pkl"
        nn_model.save_model(model_file)
        
        return {
            'model': nn_model,
            'test_metrics': test_metrics,
            'model_file': str(model_file)
        }

class AffinityPredictor:
    """Advanced predictor for real protein-ligand binding affinity predictions."""
    
    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.feature_extractor = None
        self.processed_data = None
        self.similarity_model = None
        self.target_proteins = None
    
    def load_data(self):
        """Load and prepare the processed dataset."""
        try:
            # Load processed data
            data_dir = Path("data/processed")
            features_file = data_dir / "processed_features.csv"
            target_file = data_dir / "target_values.csv"
            
            if not features_file.exists() or not target_file.exists():
                raise FileNotFoundError("Processed data not found. Please run data processing first.")
            
            # Load original data first to get protein and SMILES information
            bindingdb_file = data_dir / "bindingdb_subset.csv"
            if not bindingdb_file.exists():
                raise FileNotFoundError("BindingDB subset not found. Please run data processing first.")
            
            # Load all data into DataFrames
            original_data = pd.read_csv(bindingdb_file)
            features = pd.read_csv(features_file)
            target = pd.read_csv(target_file).iloc[:, 0]
            
            # Create index based on target data length
            valid_indices = target.index
            
            # Align all data using the valid indices
            self.processed_data = {
                'features': features.loc[valid_indices],
                'target': target,
                'proteins': original_data.loc[valid_indices, 'Target Name'].tolist(),
                'smiles': original_data.loc[valid_indices, 'Ligand SMILES'].tolist()
            }
            
            # Store unique proteins for similarity matching
            self.target_proteins = pd.Series(self.processed_data['proteins']).unique()
            
            # Initialize feature extractor
            from data_processing.feature_extractor import MolecularFeatureExtractor
            self.feature_extractor = MolecularFeatureExtractor()
            
            # Load best model
            self.load_best_model()
            
            # Train similarity model for recommendations
            self._train_similarity_model()
            
            # Verify data alignment
            data_lengths = {
                'features': len(self.processed_data['features']),
                'target': len(self.processed_data['target']),
                'proteins': len(self.processed_data['proteins']),
                'smiles': len(self.processed_data['smiles'])
            }
            
            logger.info(f"Data loaded - lengths: {data_lengths}")
            
            # Check if all lengths match
            if len(set(data_lengths.values())) != 1:
                raise ValueError(f"Data length mismatch: {data_lengths}")
            
            logger.info("Data and models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def load_best_model(self):
        """Load the best trained model."""
        try:
            metrics_file = Path("results") / "model_metrics.json"
            if not metrics_file.exists():
                raise FileNotFoundError("Model metrics not found. Please train models first.")
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Find best model
            best_r2 = -float('inf')
            best_model_name = None
            best_model_path = None
            
            for name, data in metrics.items():
                if name != 'Ensemble':  # Skip ensemble for now
                    r2 = data['test_metrics']['r2_score']
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_name = name
                        best_model_path = data.get('model_file')
            
            if best_model_name and best_model_path:
                model_path = Path(best_model_path)
                if model_path.exists():
                    self.model = joblib.load(model_path)
                    logger.info(f"Loaded best model: {best_model_name} (R² = {best_r2:.3f})")
                    return
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            else:
                raise ValueError("No valid model found")
            
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            raise
    
    def _train_similarity_model(self):
        """Train a model for protein similarity matching."""
        try:
            # Create TF-IDF vectorizer for protein names
            self.similarity_model = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 4),
                min_df=2
            )
            
            # Fit on protein names
            if self.target_proteins is not None:
                self.similarity_model.fit(self.target_proteins)
                logger.info("Trained similarity model for protein matching")
            
        except Exception as e:
            logger.error(f"Error training similarity model: {e}")
            self.similarity_model = None
    
    def find_similar_proteins(self, target_protein: str, top_k: int = 5) -> List[str]:
        """Find similar proteins in the database."""
        if self.similarity_model is None or self.target_proteins is None:
            return []
        
        try:
            # Clean and normalize protein name
            target_protein = target_protein.strip().lower()
            
            # First try exact match
            exact_matches = [p for p in self.target_proteins if target_protein in p.lower()]
            if exact_matches:
                # Sort by length to prefer shorter, more exact matches
                exact_matches.sort(key=len)
                similarities = [1.0] * len(exact_matches)
                return list(zip(exact_matches[:top_k], similarities))
            
            # If no exact match, try fuzzy matching
            # Transform query protein
            query_vec = self.similarity_model.transform([target_protein])
            
            # Transform all proteins
            all_vecs = self.similarity_model.transform(self.target_proteins)
            
            # Calculate similarities
            similarities = cosine_similarity(query_vec, all_vecs)[0]
            
            # Get top-k similar proteins
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Only include proteins with reasonable similarity
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append((self.target_proteins[idx], similarities[idx]))
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar proteins: {e}")
            return []

    def recommend_ligands(self, target_protein: str, top_k: int = 5) -> List[Dict]:
        """Find top-k ligands that might bind well with the target protein."""
        if self.processed_data is None:
            return []
        
        try:
            # Find similar proteins
            similar_proteins = self.find_similar_proteins(target_protein)
            if not similar_proteins:
                logger.info(f"No similar proteins found for: {target_protein}")
                return []
            
            recommendations = []
            seen_smiles = set()
            
            # Convert data to pandas DataFrame for easier handling
            df = pd.DataFrame({
                'protein': self.processed_data['proteins'],
                'smiles': self.processed_data['smiles'],
                'target': self.processed_data['target']
            })
            
            # For each similar protein
            for protein, similarity in similar_proteins:
                logger.info(f"Found similar protein: {protein} (similarity: {similarity:.2f})")
                
                # Find all ligands for this protein (case-insensitive)
                protein_data = df[df['protein'].str.lower() == protein.lower()].copy()
                
                if len(protein_data) == 0:
                    logger.info(f"No ligands found for protein: {protein}")
                    continue
                
                # Sort by binding affinity
                protein_data = protein_data.sort_values('target', ascending=False)
                
                # Add top ligands
                for _, row in protein_data.iterrows():
                    if len(recommendations) >= top_k:
                        break
                    
                    current_smiles = row['smiles']
                    if current_smiles in seen_smiles:
                        continue
                    
                    seen_smiles.add(current_smiles)
                    binding_affinity = float(row['target'])
                    confidence = similarity * min(1.0, binding_affinity / 10.0)  # Scale to 0-1
                    
                    recommendations.append({
                        'smiles': current_smiles,
                        'target_protein': protein,
                        'similarity_score': similarity,
                        'binding_affinity': binding_affinity,
                        'confidence': confidence,
                        'binding_strength': self._classify_binding_strength(binding_affinity)
                    })
                    
                    logger.info(f"Found ligand with binding affinity: {binding_affinity:.2f}")
            
            if not recommendations:
                logger.info("No recommendations found despite similar proteins")
                logger.info(f"Available proteins: {len(set(df['protein'].str.lower()))}")
                logger.info(f"Sample proteins: {list(set(df['protein']))[:5]}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error recommending ligands: {e}")
            logger.error(f"Data shape - proteins: {len(self.processed_data['proteins'])}, smiles: {len(self.processed_data['smiles'])}, target: {len(self.processed_data['target'])}")
            return []
    
    def predict_binding(self, smiles: str, target_protein: str) -> Dict:
        """Predict binding affinity for a specific ligand-protein pair."""
        try:
            # Extract features
            features = self.feature_extractor.prepare_features(pd.DataFrame({
                'Ligand SMILES': [smiles],
                'Target Name': [target_protein]
            }))
            
            if features is None or len(features[0]) == 0:
                raise ValueError("Could not extract features")
            
            # Make prediction
            prediction = self.model.predict(features[0])
            
            # Find similar known interactions for confidence
            similar_proteins = self.find_similar_proteins(target_protein, top_k=1)
            confidence = similar_proteins[0][1] if similar_proteins else 0.5
            
            # Scale prediction to pKd range (typically 3-12)
            pKd = float(prediction[0])
            scaled_pKd = np.clip(pKd, 3, 12)
            
            # Calculate Kd from pKd
            kd = 10 ** (-scaled_pKd + 9)  # Convert to nM
            
            return {
                'pKd': scaled_pKd,
                'Kd_nM': kd,
                'confidence': confidence,
                'binding_strength': self._classify_binding_strength(scaled_pKd),
                'similar_protein_confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error predicting binding: {e}")
            return None
    
    def batch_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for multiple ligand-protein pairs."""
        try:
            # Extract features
            features = self.feature_extractor.prepare_features(data)
            if features is None or len(features[0]) == 0:
                raise ValueError("Could not extract features")
            
            # Make predictions
            predictions = self.model.predict(features[0])
            
            # Prepare results
            results = []
            for i, (_, row) in enumerate(data.iterrows()):
                similar_proteins = self.find_similar_proteins(row['Target Name'], top_k=1)
                confidence = similar_proteins[0][1] if similar_proteins else 0.5
                
                pKd = float(predictions[i])
                scaled_pKd = np.clip(pKd, 3, 12)
                kd = 10 ** (-scaled_pKd + 9)  # Convert to nM
                
                results.append({
                    'Ligand SMILES': row['Ligand SMILES'],
                    'Target Name': row['Target Name'],
                    'pKd': scaled_pKd,
                    'Kd_nM': kd,
                    'confidence': confidence,
                    'binding_strength': self._classify_binding_strength(scaled_pKd)
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return None
    
    def _classify_binding_strength(self, pKd: float) -> str:
        """Classify binding strength based on pKd value."""
        if pKd >= 9:
            return "Very Strong"
        elif pKd >= 7:
            return "Strong"
        elif pKd >= 5:
            return "Moderate"
        elif pKd >= 4:
            return "Weak"
        else:
            return "Very Weak"
            
    def get_protein_families(self) -> Dict[str, List[str]]:
        """Get available protein families and example proteins."""
        if self.processed_data is None or 'proteins' not in self.processed_data:
            return {}
        
        try:
            proteins = pd.Series(self.processed_data['proteins'])
            
            # Common protein family keywords
            families = {
                'Kinase': r'kinase|kinases',
                'Receptor': r'receptor|receptors',
                'Channel': r'channel|channels',
                'Transporter': r'transporter|transporters',
                'Enzyme': r'enzyme|enzymes|ase$',
                'Protease': r'protease|proteases',
                'Factor': r'factor|factors',
                'Protein': r'protein|proteins'
            }
            
            results = {}
            for family, pattern in families.items():
                # Find proteins matching this family
                matching = proteins[proteins.str.lower().str.contains(pattern, regex=True)]
                if not matching.empty:
                    # Get up to 5 examples, sorted by length (shorter names first)
                    examples = sorted(matching.unique(), key=len)[:5]
                    results[family] = examples
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting protein families: {e}")
            return {}
    
    def get_search_suggestions(self) -> Dict[str, Any]:
        """Get search suggestions and database statistics."""
        if self.processed_data is None:
            return {}
        
        try:
            proteins = pd.Series(self.processed_data['proteins'])
            
            return {
                'total_proteins': len(proteins.unique()),
                'total_ligands': len(self.processed_data['smiles']),
                'protein_families': self.get_protein_families(),
                'example_searches': [
                    "kinase",
                    "receptor",
                    "ion channel",
                    "protease",
                    "transporter"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {e}")
            return {}