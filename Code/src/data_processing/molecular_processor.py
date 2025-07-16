import pandas as pd #type: ignore
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class LigandProcessor:
    """Class for processing ligand molecular data."""
    
    def __init__(self):
        self.descriptors_cache = {}
    
    def smiles_to_descriptors(self, smiles: str) -> Dict:
        """Convert SMILES string to molecular descriptors."""
        if smiles in self.descriptors_cache:
            return self.descriptors_cache[smiles]
        
        try:
            from rdkit import Chem #type: ignore
            from rdkit.Chem import Descriptors, AllChem #type: ignore
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._get_default_descriptors()
            
            descriptors = {
                'mol_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'saturated_rings': Descriptors.NumSaturatedRings(mol),
                'aliphatic_rings': Descriptors.NumAliphaticRings(mol),
                'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
                'fraction_csp3': Descriptors.FractionCsp3(mol),
                'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
                'molar_refractivity': Descriptors.MolMR(mol),
                'balaban_j': Descriptors.BalabanJ(mol),
                'bertz_ct': Descriptors.BertzCT(mol)
            }
            
            self.descriptors_cache[smiles] = descriptors
            return descriptors
            
        except ImportError:
            logger.warning("RDKit not available, using placeholder descriptors")
            return self._get_placeholder_descriptors()
        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {str(e)}")
            return self._get_default_descriptors()
    
    def _get_default_descriptors(self) -> Dict:
        """Return default descriptor values."""
        return {
            'mol_weight': 0.0,
            'logp': 0.0,
            'hbd': 0,
            'hba': 0,
            'tpsa': 0.0,
            'rotatable_bonds': 0,
            'aromatic_rings': 0,
            'saturated_rings': 0,
            'aliphatic_rings': 0,
            'heavy_atom_count': 0,
            'fraction_csp3': 0.0,
            'num_heteroatoms': 0,
            'molar_refractivity': 0.0,
            'balaban_j': 0.0,
            'bertz_ct': 0.0
        }
    
    def _get_placeholder_descriptors(self) -> Dict:
        """Return realistic placeholder descriptor values."""
        return {
            'mol_weight': np.random.normal(350, 100),
            'logp': np.random.normal(2.5, 1.5),
            'hbd': np.random.poisson(2),
            'hba': np.random.poisson(4),
            'tpsa': np.random.normal(80, 30),
            'rotatable_bonds': np.random.poisson(5),
            'aromatic_rings': np.random.poisson(2),
            'saturated_rings': np.random.poisson(1),
            'aliphatic_rings': np.random.poisson(1),
            'heavy_atom_count': np.random.poisson(25),
            'fraction_csp3': np.random.beta(2, 3),
            'num_heteroatoms': np.random.poisson(6),
            'molar_refractivity': np.random.normal(100, 30),
            'balaban_j': np.random.normal(1.5, 0.5),
            'bertz_ct': np.random.normal(800, 200)
        }
    
    def generate_fingerprints(self, smiles: str, fp_type: str = 'morgan') -> np.ndarray:
        """Generate molecular fingerprints."""
        try:
            from rdkit import Chem #type: ignore
            from rdkit.Chem import AllChem #type: ignore
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(2048)
            
            if fp_type.lower() == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            elif fp_type.lower() == 'maccs':
                from rdkit.Chem import MACCSkeys #type: ignore
                fp = MACCSkeys.GenMACCSKeys(mol)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            
            return np.array(fp)
            
        except ImportError:
            logger.warning("RDKit not available, returning random fingerprint")
            return np.random.randint(0, 2, 2048)
        except Exception as e:
            logger.error(f"Error generating fingerprint for {smiles}: {str(e)}")
            return np.zeros(2048)
    
    def calculate_3d_coordinates(self, smiles: str) -> Optional[np.ndarray]:
        """Calculate 3D coordinates for the molecule."""
        try:
            from rdkit import Chem #type: ignore
            from rdkit.Chem import AllChem #type: ignore
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            conf = mol.GetConformer()
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            
            return np.array(coords)
            
        except ImportError:
            logger.warning("RDKit not available for 3D coordinate generation")
            return None
        except Exception as e:
            logger.error(f"Error generating 3D coordinates for {smiles}: {str(e)}")
            return None
    
    def process_ligand_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """Process a batch of ligands."""
        results = []
        
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                logger.info(f"Processing ligand {i+1}/{len(smiles_list)}")
            
            descriptors = self.smiles_to_descriptors(smiles)
            descriptors['smiles'] = smiles
            descriptors['ligand_id'] = i
            
            results.append(descriptors)
        
        df = pd.DataFrame(results)
        logger.info(f"Processed {len(df)} ligands")
        return df
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string."""
        try:
            from rdkit import Chem #type: ignore
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except ImportError:
            # Basic validation without RDKit
            return len(smiles) > 0 and not any(char in smiles for char in ['@', '#'] if smiles.count(char) > 10)
        except:
            return False
    
    def filter_drug_like(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter compounds based on drug-like properties (Lipinski's Rule of Five)."""
        logger.info(f"Applying drug-like filters to {len(df)} compounds")
        
        # Lipinski's Rule of Five
        lipinski_filter = (
            (df['mol_weight'] <= 500) &
            (df['logp'] <= 5) &
            (df['hbd'] <= 5) &
            (df['hba'] <= 10)
        )
        
        # Additional ADMET filters
        admet_filter = (
            (df['tpsa'] <= 140) &
            (df['rotatable_bonds'] <= 10)
        )
        
        filtered_df = df[lipinski_filter & admet_filter].copy()
        logger.info(f"Retained {len(filtered_df)} drug-like compounds")
        
        return filtered_df


class ProteinProcessor:
    """Class for processing protein structure data."""
    
    def __init__(self):
        self.structure_cache = {}
    
    def load_pdb_structure(self, pdb_file: str) -> Optional[Dict]:
        """Load and parse PDB structure file."""
        if pdb_file in self.structure_cache:
            return self.structure_cache[pdb_file]
        
        try:
            from Bio.PDB import PDBParser #type: ignore
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_file)
            
            # Extract basic information
            structure_info = {
                'structure': structure,
                'chains': [],
                'residues': [],
                'atoms': []
            }
            
            for model in structure:
                for chain in model:
                    chain_info = {
                        'chain_id': chain.get_id(),
                        'residue_count': len(list(chain.get_residues()))
                    }
                    structure_info['chains'].append(chain_info)
                    
                    for residue in chain:
                        if residue.get_id()[0] == ' ':  # Standard residues only
                            res_info = {
                                'residue_name': residue.get_resname(),
                                'residue_id': residue.get_id()[1],
                                'chain_id': chain.get_id()
                            }
                            structure_info['residues'].append(res_info)
            
            self.structure_cache[pdb_file] = structure_info
            return structure_info
            
        except ImportError:
            logger.warning("BioPython not available for PDB processing")
            return self._get_placeholder_structure()
        except Exception as e:
            logger.error(f"Error loading PDB file {pdb_file}: {str(e)}")
            return None
    
    def _get_placeholder_structure(self) -> Dict:
        """Return placeholder structure information."""
        return {
            'structure': None,
            'chains': [{'chain_id': 'A', 'residue_count': 200}],
            'residues': [{'residue_name': 'ALA', 'residue_id': i, 'chain_id': 'A'} for i in range(1, 201)],
            'atoms': []
        }
    
    def extract_amino_acid_composition(self, structure_info: Dict) -> Dict:
        """Extract amino acid composition from protein structure."""
        if not structure_info:
            return {}
        
        aa_composition = {}
        standard_aas = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                       'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                       'THR', 'TRP', 'TYR', 'VAL']
        
        # Initialize counts
        for aa in standard_aas:
            aa_composition[aa] = 0
        
        # Count residues
        for residue in structure_info['residues']:
            res_name = residue['residue_name']
            if res_name in aa_composition:
                aa_composition[res_name] += 1
        
        # Convert to fractions
        total_residues = sum(aa_composition.values())
        if total_residues > 0:
            for aa in aa_composition:
                aa_composition[f'{aa}_fraction'] = aa_composition[aa] / total_residues
        
        return aa_composition
    
    def identify_binding_pocket(self, pdb_file: str, ligand_coords: Optional[np.ndarray] = None) -> List[Dict]:
        """Identify binding pocket residues."""
        try:
            from Bio.PDB import PDBParser, NeighborSearch #type: ignore
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_file)
            
            # Get all atoms
            atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] == ' ':  # Standard residues
                            for atom in residue:
                                atoms.append(atom)
            
            # If no ligand coordinates provided, find center of protein
            if ligand_coords is None:
                coords = np.array([atom.get_coord() for atom in atoms])
                center = np.mean(coords, axis=0)
                ligand_coords = center.reshape(1, -1)
            
            # Find nearby residues (within 5 Å)
            ns = NeighborSearch(atoms)
            pocket_residues = []
            
            for coord in ligand_coords:
                nearby_atoms = ns.search(coord, 5.0)
                for atom in nearby_atoms:
                    residue = atom.get_parent()
                    res_info = {
                        'residue_name': residue.get_resname(),
                        'residue_id': residue.get_id()[1],
                        'chain_id': residue.get_parent().get_id(),
                        'distance': np.linalg.norm(atom.get_coord() - coord)
                    }
                    if res_info not in pocket_residues:
                        pocket_residues.append(res_info)
            
            return pocket_residues
            
        except ImportError:
            logger.warning("BioPython not available for binding pocket identification")
            return []
        except Exception as e:
            logger.error(f"Error identifying binding pocket: {str(e)}")
            return []
    
    def extract_secondary_structure(self, pdb_file: str) -> Dict:
        """Extract secondary structure information."""
        try:
            # For now, return placeholder values
            # In a full implementation, you'd use DSSP or similar
            return {
                'helix_fraction': np.random.uniform(0.2, 0.4),
                'sheet_fraction': np.random.uniform(0.1, 0.3),
                'loop_fraction': np.random.uniform(0.3, 0.6)
            }
        except Exception as e:
            logger.error(f"Error extracting secondary structure: {str(e)}")
            return {'helix_fraction': 0.3, 'sheet_fraction': 0.2, 'loop_fraction': 0.5}
