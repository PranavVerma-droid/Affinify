import pandas as pd #type: ignore
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularFeatureExtractor:
    """
    Extract molecular features from SMILES strings and protein information
    """
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = None
        
    def extract_molecular_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Extract molecular descriptors from SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            pd.DataFrame: Molecular descriptors
        """
        try:
            from rdkit import Chem #type: ignore
            from rdkit.Chem import Descriptors, Crippen, Lipinski, QED #type: ignore
            
            features = []
            
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    # Handle invalid SMILES
                    features.append(self._get_null_descriptors())
                    continue
                
                # Calculate descriptors
                desc = {
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Crippen.MolLogP(mol),
                    'num_hbd': Descriptors.NumHDonors(mol),
                    'num_hba': Descriptors.NumHAcceptors(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                    'num_rings': Descriptors.RingCount(mol),
                    'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
                    'num_heavy_atoms': Descriptors.HeavyAtomCount(mol),
                    'fractional_csp3': Descriptors.FractionCsp3(mol),
                    'num_aliphatic_rings': Descriptors.NumAliphaticRings(mol),
                    'num_saturated_rings': Descriptors.NumSaturatedRings(mol),
                    'balaban_j': Descriptors.BalabanJ(mol),
                    'bertz_ct': Descriptors.BertzCT(mol),
                    'chi0': Descriptors.Chi0(mol),
                    'chi1': Descriptors.Chi1(mol),
                    'kappa1': Descriptors.Kappa1(mol),
                    'kappa2': Descriptors.Kappa2(mol),
                    'kappa3': Descriptors.Kappa3(mol),
                    'qed': QED.qed(mol),
                    'slogp_vsa1': Descriptors.SlogP_VSA1(mol),
                    'slogp_vsa2': Descriptors.SlogP_VSA2(mol),
                    'smr_vsa1': Descriptors.SMR_VSA1(mol),
                    'smr_vsa2': Descriptors.SMR_VSA2(mol),
                    'peoe_vsa1': Descriptors.PEOE_VSA1(mol),
                    'peoe_vsa2': Descriptors.PEOE_VSA2(mol),
                    'estate_vsa1': Descriptors.EState_VSA1(mol),
                    'estate_vsa2': Descriptors.EState_VSA2(mol),
                    'vsa_estate1': Descriptors.VSA_EState1(mol),
                    'vsa_estate2': Descriptors.VSA_EState2(mol),
                    'max_partial_charge': Descriptors.MaxPartialCharge(mol),
                    'min_partial_charge': Descriptors.MinPartialCharge(mol),
                    'max_abs_partial_charge': Descriptors.MaxAbsPartialCharge(mol),
                    'min_abs_partial_charge': Descriptors.MinAbsPartialCharge(mol),
                    'lipinski_hbd': Lipinski.NumHDonors(mol),
                    'lipinski_hba': Lipinski.NumHAcceptors(mol),
                    'lipinski_violations': self._count_lipinski_violations(mol),
                    'formal_charge': Chem.rdmolops.GetFormalCharge(mol)
                }
                
                features.append(desc)
            
            df = pd.DataFrame(features)
            logger.info(f"Extracted {len(df.columns)} molecular descriptors")
            return df
            
        except ImportError:
            logger.warning("RDKit not available. Using simplified descriptors.")
            return self._extract_simple_descriptors(smiles_list)
    
    def _get_null_descriptors(self) -> Dict:
        """Return dictionary of null descriptors for invalid SMILES"""
        return {
            'molecular_weight': np.nan,
            'logp': np.nan,
            'num_hbd': np.nan,
            'num_hba': np.nan,
            'tpsa': np.nan,
            'num_rotatable_bonds': np.nan,
            'num_aromatic_rings': np.nan,
            'num_rings': np.nan,
            'num_heteroatoms': np.nan,
            'num_heavy_atoms': np.nan,
            'fractional_csp3': np.nan,
            'num_aliphatic_rings': np.nan,
            'num_saturated_rings': np.nan,
            'balaban_j': np.nan,
            'bertz_ct': np.nan,
            'chi0': np.nan,
            'chi1': np.nan,
            'kappa1': np.nan,
            'kappa2': np.nan,
            'kappa3': np.nan,
            'qed': np.nan,
            'slogp_vsa1': np.nan,
            'slogp_vsa2': np.nan,
            'smr_vsa1': np.nan,
            'smr_vsa2': np.nan,
            'peoe_vsa1': np.nan,
            'peoe_vsa2': np.nan,
            'estate_vsa1': np.nan,
            'estate_vsa2': np.nan,
            'vsa_estate1': np.nan,
            'vsa_estate2': np.nan,
            'max_partial_charge': np.nan,
            'min_partial_charge': np.nan,
            'max_abs_partial_charge': np.nan,
            'min_abs_partial_charge': np.nan,
            'lipinski_hbd': np.nan,
            'lipinski_hba': np.nan,
            'lipinski_violations': np.nan,
            'formal_charge': np.nan
        }
    
    def _count_lipinski_violations(self, mol) -> int:
        """Count Lipinski rule violations"""
        from rdkit.Chem import Descriptors, Lipinski #type: ignore
        
        violations = 0
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        if Descriptors.MolLogP(mol) > 5:
            violations += 1
        if Lipinski.NumHDonors(mol) > 5:
            violations += 1
        if Lipinski.NumHAcceptors(mol) > 10:
            violations += 1
        
        return violations
    
    def _extract_simple_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Extract simple descriptors when RDKit is not available
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            pd.DataFrame: Simple descriptors
        """
        features = []
        
        for smiles in smiles_list:
            desc = {
                'smiles_length': len(smiles),
                'num_carbons': smiles.count('C'),
                'num_nitrogens': smiles.count('N'),
                'num_oxygens': smiles.count('O'),
                'num_sulfurs': smiles.count('S'),
                'num_rings': smiles.count('c'),
                'num_double_bonds': smiles.count('='),
                'num_triple_bonds': smiles.count('#'),
                'num_branches': smiles.count('('),
                'has_aromatic': int('c' in smiles.lower()),
                'has_charge': int('+' in smiles or '-' in smiles),
                'molecular_complexity': len(set(smiles))
            }
            features.append(desc)
        
        df = pd.DataFrame(features)
        logger.info(f"Extracted {len(df.columns)} simple descriptors")
        return df
    
    def extract_protein_features(self, protein_ids: List[str], 
                               protein_names: List[str]) -> pd.DataFrame:
        """
        Extract protein features from protein IDs and names
        
        Args:
            protein_ids: List of protein IDs
            protein_names: List of protein names
            
        Returns:
            pd.DataFrame: Protein features
        """
        features = []
        
        for protein_id, protein_name in zip(protein_ids, protein_names):
            desc = {
                'protein_id_length': len(str(protein_id)) if protein_id else 0,
                'protein_name_length': len(str(protein_name)) if protein_name else 0,
                'has_pdb_id': int(bool(protein_id and len(str(protein_id)) >= 4)),
                'is_kinase': int('kinase' in str(protein_name).lower() if protein_name else False),
                'is_receptor': int('receptor' in str(protein_name).lower() if protein_name else False),
                'is_enzyme': int('enzyme' in str(protein_name).lower() if protein_name else False),
                'is_channel': int('channel' in str(protein_name).lower() if protein_name else False),
                'is_transporter': int('transporter' in str(protein_name).lower() if protein_name else False),
                'has_alpha': int('alpha' in str(protein_name).lower() if protein_name else False),
                'has_beta': int('beta' in str(protein_name).lower() if protein_name else False),
                'has_gamma': int('gamma' in str(protein_name).lower() if protein_name else False),
                'word_count': len(str(protein_name).split()) if protein_name else 0,
                'has_numbers': int(any(c.isdigit() for c in str(protein_name)) if protein_name else False)
            }
            features.append(desc)
        
        df = pd.DataFrame(features)
        logger.info(f"Extracted {len(df.columns)} protein features")
        return df
    
    def create_interaction_features(self, mol_features: pd.DataFrame,
                                  protein_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between molecular and protein features
        
        Args:
            mol_features: Molecular features
            protein_features: Protein features
            
        Returns:
            pd.DataFrame: Interaction features
        """
        interaction_features = []
        
        # Get numeric molecular features
        mol_numeric = mol_features.select_dtypes(include=[np.number])
        
        for idx in range(len(mol_features)):
            interactions = {}
            
            # Molecular weight interactions
            if 'molecular_weight' in mol_numeric.columns:
                mw = mol_numeric.iloc[idx]['molecular_weight']
                if not np.isnan(mw):
                    interactions['mw_kinase_interaction'] = mw * protein_features.iloc[idx]['is_kinase']
                    interactions['mw_receptor_interaction'] = mw * protein_features.iloc[idx]['is_receptor']
                    interactions['mw_enzyme_interaction'] = mw * protein_features.iloc[idx]['is_enzyme']
                    interactions['mw_high'] = int(mw > 500)
                    interactions['mw_low'] = int(mw < 200)
            
            # LogP interactions
            if 'logp' in mol_numeric.columns:
                logp = mol_numeric.iloc[idx]['logp']
                if not np.isnan(logp):
                    interactions['logp_kinase_interaction'] = logp * protein_features.iloc[idx]['is_kinase']
                    interactions['logp_receptor_interaction'] = logp * protein_features.iloc[idx]['is_receptor']
                    interactions['logp_high'] = int(logp > 5)
                    interactions['logp_low'] = int(logp < 0)
            
            # Ring interactions
            if 'num_rings' in mol_numeric.columns:
                rings = mol_numeric.iloc[idx]['num_rings']
                if not np.isnan(rings):
                    interactions['rings_kinase_interaction'] = rings * protein_features.iloc[idx]['is_kinase']
                    interactions['rings_receptor_interaction'] = rings * protein_features.iloc[idx]['is_receptor']
                    interactions['many_rings'] = int(rings > 3)
            
            # Default values for missing features
            for key in ['mw_kinase_interaction', 'mw_receptor_interaction', 'mw_enzyme_interaction',
                       'mw_high', 'mw_low', 'logp_kinase_interaction', 'logp_receptor_interaction',
                       'logp_high', 'logp_low', 'rings_kinase_interaction', 'rings_receptor_interaction',
                       'many_rings']:
                if key not in interactions:
                    interactions[key] = 0
            
            interaction_features.append(interactions)
        
        df = pd.DataFrame(interaction_features)
        logger.info(f"Created {len(df.columns)} interaction features")
        return df
    
    def process_binding_affinity(self, ki_values: pd.Series, 
                                ic50_values: pd.Series,
                                kd_values: pd.Series) -> pd.Series:
        """
        Process binding affinity values to create a unified target variable
        
        Args:
            ki_values: Ki values in nM
            ic50_values: IC50 values in nM
            kd_values: Kd values in nM
            
        Returns:
            pd.Series: Processed binding affinity values (pKd/pKi/pIC50)
        """
        # Convert to pKd/pKi/pIC50 scale (negative log10 of nM values)
        def convert_to_p_value(values):
            # Convert nM to M and take negative log
            values_m = values * 1e-9
            return -np.log10(values_m)
        
        # Priority: Kd > Ki > IC50
        binding_affinity = pd.Series(index=ki_values.index, dtype=float)
        
        # Use Kd values where available
        mask = pd.notna(kd_values) & (kd_values > 0)
        binding_affinity[mask] = convert_to_p_value(kd_values[mask])
        
        # Use Ki values where Kd is not available
        mask = pd.isna(binding_affinity) & pd.notna(ki_values) & (ki_values > 0)
        binding_affinity[mask] = convert_to_p_value(ki_values[mask])
        
        # Use IC50 values where neither Kd nor Ki is available
        mask = pd.isna(binding_affinity) & pd.notna(ic50_values) & (ic50_values > 0)
        binding_affinity[mask] = convert_to_p_value(ic50_values[mask])
        
        logger.info(f"Processed {binding_affinity.count()} binding affinity values")
        return binding_affinity
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare all features for machine learning
        
        Args:
            df: Raw dataset
            
        Returns:
            Tuple of (features, target)
        """
        logger.info("Preparing features for machine learning...")
        
        # Check for required columns with flexible naming
        smiles_col = None
        pdb_col = None
        target_name_col = None
        
        # Look for SMILES column
        for col in df.columns:
            if 'smiles' in col.lower():
                smiles_col = col
                break
        
        # Look for PDB/protein ID column
        for col in df.columns:
            if 'pdb' in col.lower() or 'protein' in col.lower():
                pdb_col = col
                break
        
        # Look for target name column
        for col in df.columns:
            if 'target' in col.lower() or 'protein' in col.lower():
                target_name_col = col
                break
        
        # Use defaults if not found
        if smiles_col is None:
            smiles_col = 'Ligand SMILES'
        if pdb_col is None:
            pdb_col = 'PDB ID(s)'
        if target_name_col is None:
            target_name_col = 'Target Name'
        
        # Extract molecular features
        smiles_data = df[smiles_col].fillna('').tolist() if smiles_col in df.columns else [''] * len(df)
        mol_features = self.extract_molecular_descriptors(smiles_data)
        
        # Extract protein features
        pdb_data = df[pdb_col].fillna('').tolist() if pdb_col in df.columns else [''] * len(df)
        target_data = df[target_name_col].fillna('').tolist() if target_name_col in df.columns else [''] * len(df)
        protein_features = self.extract_protein_features(pdb_data, target_data)
        
        # Create interaction features
        interaction_features = self.create_interaction_features(mol_features, protein_features)
        
        # Combine all features
        all_features = pd.concat([mol_features, protein_features, interaction_features], axis=1)
        
        # Process target variable
        target = self._process_binding_affinity(df)
        
        # Remove rows with missing target values
        if len(target) > 0:
            valid_mask = pd.notna(target)
            all_features = all_features[valid_mask]
            target = target[valid_mask]
        else:
            logger.warning("No valid target values found, creating dummy target")
            target = pd.Series([5.0] * len(all_features))
        
        # Handle missing values in features
        all_features = all_features.fillna(0)
        
    def _process_binding_affinity(self, df):
        """Process binding affinity values and convert to pKd/pKi/pIC50 scale"""
        logger.info("Processing binding affinity values")
        
        # Look for affinity columns
        affinity_columns = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
        available_affinity_columns = [col for col in affinity_columns if col in df.columns]
        
        if not available_affinity_columns:
            logger.warning("No binding affinity columns found")
            return pd.Series([5.0] * len(df))  # Default value
        
        # Use the first available affinity column
        affinity_col = available_affinity_columns[0]
        affinity_values = df[affinity_col].copy()
        
        # Convert to numeric, handling non-numeric values
        affinity_values = pd.to_numeric(affinity_values, errors='coerce')
        
        # Remove invalid values
        valid_mask = (affinity_values > 0) & (affinity_values < 1000000) & pd.notna(affinity_values)
        
        # Create target series
        target = pd.Series(index=df.index, dtype=float)
        
        if valid_mask.any():
            # Convert nM to pKd/pKi/pIC50 scale: pKd = -log10(Kd in M) = -log10(Kd_nM * 1e-9)
            # This gives us values typically in the range 4-10, with higher values = stronger binding
            p_values = -np.log10(affinity_values[valid_mask] * 1e-9)
            
            # Clip to reasonable range
            p_values = np.clip(p_values, 3, 12)
            
            target[valid_mask] = p_values
        
        # Fill missing values with median or default
        if target.notna().any():
            target = target.fillna(target.median())
        else:
            target = target.fillna(5.0)  # Default value
        
        logger.info(f"Processed {len(target)} binding affinity values")
        logger.info(f"pKd/pKi range: {target.min():.2f} to {target.max():.2f}")
        
        return target

if __name__ == "__main__":
    # Example usage
    extractor = MolecularFeatureExtractor()
    
    # Sample data
    sample_data = pd.DataFrame({
        'Ligand SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN(CC)CC'],
        'PDB ID(s)': ['1ABC', '2DEF', '3GHI', '4JKL'],
        'Target Name': ['Protein kinase A', 'GABA receptor', 'Acetylcholine enzyme', 'Ion channel'],
        'Ki (nM)': [10.5, 250.0, 5.2, 1500.0],
        'IC50 (nM)': [None, 180.0, None, 2000.0],
        'Kd (nM)': [8.5, None, 4.8, None]
    })
    
    # Process features
    features, target = extractor.prepare_features(sample_data)
    
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Sample features:\n{features.head()}")
    print(f"Sample target values:\n{target.head()}")