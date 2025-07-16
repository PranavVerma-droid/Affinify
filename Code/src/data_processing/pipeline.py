import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

logger = logging.getLogger(__name__)

class DataPipeline:
    """Class for creating data preprocessing pipelines."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.target_column = 'binding_affinity_nm'
        
    def preprocess_binding_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess binding affinity data."""
        logger.info(f"Preprocessing {len(df)} binding records")
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean binding affinity values
        processed_df = self._clean_binding_values(processed_df)
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Convert activity types to numerical
        processed_df = self._encode_activity_types(processed_df)
        
        # Create additional features
        processed_df = self._create_additional_features(processed_df)
        
        logger.info(f"Preprocessing complete. Final dataset: {len(processed_df)} records")
        return processed_df
    
    def _clean_binding_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize binding affinity values."""
        # Remove rows with missing binding affinity
        df = df.dropna(subset=['binding_affinity_nm'])
        
        # Convert to numeric if needed
        df['binding_affinity_nm'] = pd.to_numeric(df['binding_affinity_nm'], errors='coerce')
        
        # Remove unrealistic values (e.g., negative or extremely large)
        df = df[(df['binding_affinity_nm'] > 0) & (df['binding_affinity_nm'] < 1e9)]
        
        # Calculate pKd if not present
        if 'pKd' not in df.columns:
            df['pKd'] = -np.log10(df['binding_affinity_nm'] * 1e-9)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill missing molecular descriptors with median values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['binding_affinity_nm', 'pKd']:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['ligand_smiles', 'pdb_id']:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def _encode_activity_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode activity types to numerical values."""
        if 'activity_type' in df.columns:
            # Create binary columns for each activity type
            activity_types = df['activity_type'].unique()
            for activity in activity_types:
                df[f'activity_{activity}'] = (df['activity_type'] == activity).astype(int)
        
        return df
    
    def _create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from existing data."""
        # Molecular complexity features
        if 'mol_weight' in df.columns and 'heavy_atom_count' in df.columns:
            df['mol_complexity'] = df['mol_weight'] / df['heavy_atom_count'].replace(0, 1)
        
        # Lipophilicity efficiency
        if 'logp' in df.columns and 'mol_weight' in df.columns:
            df['lipophilic_efficiency'] = df['logp'] / (df['mol_weight'] / 100)
        
        # Hydrogen bonding potential
        if 'hbd' in df.columns and 'hba' in df.columns:
            df['total_hb'] = df['hbd'] + df['hba']
            df['hb_ratio'] = df['hbd'] / (df['hba'] + 1)  # +1 to avoid division by zero
        
        # Aromatic fraction
        if 'aromatic_rings' in df.columns and 'heavy_atom_count' in df.columns:
            df['aromatic_fraction'] = df['aromatic_rings'] / df['heavy_atom_count'].replace(0, 1)
        
        # Flexibility index
        if 'rotatable_bonds' in df.columns and 'heavy_atom_count' in df.columns:
            df['flexibility_index'] = df['rotatable_bonds'] / df['heavy_atom_count'].replace(0, 1)
        
        return df
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train-test split maintaining diversity."""
        logger.info(f"Creating train-test split with test_size={test_size}")
        
        # Stratify by protein target if available
        if 'protein_name' in df.columns:
            # Group proteins with few samples
            protein_counts = df['protein_name'].value_counts()
            df['protein_group'] = df['protein_name'].map(
                lambda x: x if protein_counts[x] >= 10 else 'other'
            )
            stratify_column = 'protein_group'
        else:
            stratify_column = None
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df[stratify_column] if stratify_column else None
        )
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables."""
        # Define feature columns (exclude target and identifier columns)
        exclude_columns = [
            'binding_affinity_nm', 'pKd', 'ligand_smiles', 'pdb_id', 
            'protein_name', 'activity_type', 'assay_type', 'protein_group'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        self.feature_columns = feature_columns
        
        # Prepare features
        X = df[feature_columns].select_dtypes(include=[np.number])
        
        # Prepare target (use pKd if available, otherwise convert from nM)
        if 'pKd' in df.columns:
            y = df['pKd']
        else:
            y = -np.log10(df['binding_affinity_nm'] * 1e-9)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        
        return X, y
    
    def normalize_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Normalize features using StandardScaler."""
        logger.info("Normalizing features")
        
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data if provided
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            X_test_scaled = None
        
        return X_train_scaled, X_test_scaled
    
    def save_pipeline(self, save_path: str):
        """Save the preprocessing pipeline."""
        import joblib
        
        pipeline_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(pipeline_data, save_path)
        logger.info(f"Pipeline saved to {save_path}")
    
    def load_pipeline(self, load_path: str):
        """Load a preprocessing pipeline."""
        import joblib
        
        pipeline_data = joblib.load(load_path)
        self.scaler = pipeline_data['scaler']
        self.label_encoder = pipeline_data['label_encoder']
        self.feature_columns = pipeline_data['feature_columns']
        self.target_column = pipeline_data['target_column']
        
        logger.info(f"Pipeline loaded from {load_path}")
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality and return quality metrics."""
        quality_metrics = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'invalid_smiles': 0,
            'invalid_pdb_ids': 0,
            'outliers': 0
        }
        
        # Check SMILES validity
        if 'ligand_smiles' in df.columns:
            try:
                from rdkit import Chem
                invalid_smiles = df['ligand_smiles'].apply(
                    lambda x: Chem.MolFromSmiles(x) is None if pd.notna(x) else True
                ).sum()
                quality_metrics['invalid_smiles'] = invalid_smiles
            except ImportError:
                logger.warning("RDKit not available for SMILES validation")
        
        # Check PDB ID format
        if 'pdb_id' in df.columns:
            invalid_pdb = df['pdb_id'].apply(
                lambda x: not (isinstance(x, str) and len(x) == 4 and x.isalnum()) if pd.notna(x) else True
            ).sum()
            quality_metrics['invalid_pdb_ids'] = invalid_pdb
        
        # Check for outliers in binding affinity
        if 'binding_affinity_nm' in df.columns:
            Q1 = df['binding_affinity_nm'].quantile(0.25)
            Q3 = df['binding_affinity_nm'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df['binding_affinity_nm'] < Q1 - 1.5 * IQR) | 
                       (df['binding_affinity_nm'] > Q3 + 1.5 * IQR)).sum()
            quality_metrics['outliers'] = outliers
        
        logger.info(f"Data quality validation complete: {quality_metrics}")
        return quality_metrics
