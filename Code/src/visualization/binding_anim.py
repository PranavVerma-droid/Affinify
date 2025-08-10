#!/usr/bin/env python3
"""
Binding Animation Module for Affinify
Creates 3D animated visualizations of protein-ligand binding interactions.
This version uses RDKit to generate and display accurate molecular structures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from pathlib import Path
import uuid
import logging
from typing import Tuple, Dict, Optional, List
import random
import time
import os
import sys

# Try to import RDKit, which is essential for molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not found. Some features may not work properly.")
    print("To install RDKit: pip install rdkit-pypi")
    RDKIT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define CPK colors for common elements
ATOM_COLORS = {
    'C': '#606060',  # Carbon (gray)
    'O': '#FF0D0D',  # Oxygen (red)
    'N': '#0000FF',  # Nitrogen (blue)
    'H': '#CCCCCC',  # Hydrogen (light gray)
    'S': '#FFFF30',  # Sulfur (yellow)
    'P': '#FF8000',  # Phosphorus (orange)
    'F': '#90E050',  # Fluorine (light green)
    'Cl': '#1FF01F', # Chlorine (green)
    'Br': '#A62929', # Bromine (dark red)
    'I': '#940094',  # Iodine (purple)
    'DEFAULT': '#FFC0CB' # Default (pink)
}

class ProteinLigandAnimator:
    """Creates animated visualizations of protein-ligand binding interactions"""
    
    def __init__(self, output_dir: str = "videos"):
        """
        Initialize the animator.
        
        Args:
            output_dir: Directory to save animation videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Animation parameters
        self.fps = 30
        self.duration = 8
        self.frames = self.fps * self.duration
        
    def smiles_to_3d_structure(self, smiles: str) -> Optional[Dict]:
        """
        Convert a SMILES string to a 3D molecular structure using RDKit.
        
        Args:
            smiles: The SMILES string of the molecule.
            
        Returns:
            A dictionary with atom symbols, positions, and bonds, or None on failure.
        """
        if not RDKIT_AVAILABLE:
            logger.error("RDKit not available for molecular structure generation")
            return None
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                logger.error(f"Could not create molecule from SMILES: {smiles}")
                return None

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception as e:
                logger.warning(f"Could not optimize molecule geometry: {e}")

            conf = mol.GetConformer()
            atoms = []
            symbols = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                atoms.append([pos.x, pos.y, pos.z])
                symbols.append(atom.GetSymbol())

            bonds = []
            for bond in mol.GetBonds():
                bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

            return {
                'atoms': np.array(atoms),
                'symbols': symbols,
                'bonds': bonds,
                'n_atoms': len(atoms)
            }
        except Exception as e:
            logger.error(f"An error occurred in smiles_to_3d_structure: {e}")
            return None

    def create_protein_structure(self, n_atoms: int = 80) -> Dict:
        """
        Create a randomized, simplified 3D protein structure representation.
        
        Args:
            n_atoms: Number of atoms to generate for the protein.
            
        Returns:
            Dictionary containing atom positions and bonds.
        """
        atoms = []
        bonds = []
        
        # Generate random but somewhat clustered atom positions
        center = np.random.rand(3) * 5
        for _ in range(n_atoms):
            atoms.append(center + np.random.randn(3) * 2.5)
        
        # Connect nearby atoms to form a mesh-like structure
        atoms_np = np.array(atoms)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(atoms_np[i] - atoms_np[j])
                if dist < 2.5 and random.random() > 0.7:
                    bonds.append([i, j])
                    
        return {
            'atoms': atoms_np,
            'bonds': bonds,
            'n_atoms': n_atoms
        }
    
    def animate_binding(self, protein_name: str, ligand_smiles: str, 
                       binding_affinity: float, confidence: float,
                       target_protein: str) -> Optional[str]:
        """
        Create binding animation between protein and ligand.
        
        Args:
            protein_name: Name of the target protein
            ligand_smiles: SMILES string of the ligand
            binding_affinity: Predicted binding affinity
            confidence: Prediction confidence
            target_protein: Original target protein name
            
        Returns:
            Path to the generated video file or None on failure.
        """
        # Check for valid inputs
        if not ligand_smiles or not isinstance(ligand_smiles, str) or len(ligand_smiles) < 2:
            logger.error(f"Invalid SMILES string provided: {ligand_smiles}")
            return None
            
        if not protein_name or not isinstance(protein_name, str):
            logger.error(f"Invalid protein name provided: {protein_name}")
            return None
        
        # Check if RDKit is available
        if not RDKIT_AVAILABLE:
            logger.error("RDKit is required for animation generation but is not available")
            return None
            
        video_id = str(uuid.uuid4())[:8]
        video_path = self.output_dir / f"binding_{video_id}.gif"
        
        logger.info(f"Creating binding animation for {protein_name} + {ligand_smiles}...")
        logger.info(f"Output path will be: {video_path.absolute()}")
        
        # Create molecular structures
        protein_structure = self.create_protein_structure(120)
        ligand_structure = self.smiles_to_3d_structure(ligand_smiles)
        
        if not ligand_structure:
            logger.error("Failed to generate ligand structure. Aborting animation.")
            return None

        logger.info("Successfully generated molecular structures")
        logger.info(f"Protein atoms: {protein_structure['n_atoms']}, Ligand atoms: {ligand_structure['n_atoms']}")
        
        # Set up the figure
        try:
            fig = plt.figure(figsize=(16, 10), facecolor='#1E1E1E')
            gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[3, 1], width_ratios=[1, 1, 1])
            
            ax_main = fig.add_subplot(gs[0, :], projection='3d', facecolor='#1E1E1E')
            ax_info1 = fig.add_subplot(gs[1, 0], facecolor='#2D2D2D')
            ax_info2 = fig.add_subplot(gs[1, 1], facecolor='#2D2D2D')
            ax_info3 = fig.add_subplot(gs[1, 2], facecolor='#2D2D2D')
            
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            
            logger.info("Starting animation generation with FuncAnimation")
        except Exception as e:
            logger.error(f"Error setting up matplotlib figure: {e}")
            return None

        def animate_frame(frame):
            # ...existing code...
            ax_main.clear()
            
            # --- Animation Phases ---
            total_frames = self.frames
            phase1_end = total_frames * 0.2
            phase2_end = total_frames * 0.5
            phase3_end = total_frames * 0.8
            
            # --- Camera and Positioning ---
            # Base positions before any movement or zoom adjustment
            base_protein_pos = np.array([-5, 0, 0])
            base_ligand_pos = np.array([5, 0, 0])
            
            # Calculate positions based on animation phase
            if frame < phase1_end:
                # Initial approach phase
                protein_pos = base_protein_pos.copy()
                ligand_pos = base_ligand_pos.copy()
            elif frame < phase2_end:
                # Movement toward binding site
                t = (frame - phase1_end) / (phase2_end - phase1_end)
                ease_t = 1 - (1 - t) ** 3
                protein_pos = np.array([-5 + 3 * ease_t, 0, 0])
                ligand_pos = np.array([5 - 4.5 * ease_t, 0, 0])
            else:
                # In binding phase - keep positions constant
                protein_pos = np.array([-2, 0, 0])
                ligand_pos = np.array([0.5, 0, 0])
            
            # Define binding center as the midpoint between protein and ligand for all phases
            binding_center = np.array([0.5 * (protein_pos[0] + ligand_pos[0]), 
                                     0.5 * (protein_pos[1] + ligand_pos[1]), 
                                     0.5 * (protein_pos[2] + ligand_pos[2])])
            
            # Zoom calculation 
            zoom_factor = 8.0
            if frame > phase2_end:
                # Calculate zoom factor for binding focus phase
                t_zoom = min(1.0, (frame - phase2_end) / (total_frames * 0.1))
                # Use easing function for smooth zoom
                ease_zoom = t_zoom * t_zoom * (3.0 - 2.0 * t_zoom)
                zoom_factor = 8.0 - 4.5 * ease_zoom
                
                # Set camera to focus on binding site
                ax_main.view_init(elev=30, azim=frame % 360)
            
            # Set consistent view limits based on zoom factor
            ax_main.set_xlim([binding_center[0] - zoom_factor, binding_center[0] + zoom_factor])
            ax_main.set_ylim([binding_center[1] - zoom_factor, binding_center[1] + zoom_factor])
            ax_main.set_zlim([binding_center[2] - zoom_factor, binding_center[2] + zoom_factor])
            
            # --- Set plot style ---
            ax_main.set_title("Protein-Ligand Binding Simulation", fontsize=16, color='white', pad=20)
            ax_main.grid(False)
            ax_main.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            ax_main.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            ax_main.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            ax_main.xaxis.line.set_color('white')
            ax_main.yaxis.line.set_color('white')
            ax_main.zaxis.line.set_color('white')
            ax_main.tick_params(axis='x', colors='white')
            ax_main.tick_params(axis='y', colors='white')
            ax_main.tick_params(axis='z', colors='white')
            ax_main.set_xlabel('X', color='white')
            ax_main.set_ylabel('Y', color='white')
            ax_main.set_zlabel('Z', color='white')

            # Center ligand structure before applying position
            ligand_atoms_centered = ligand_structure['atoms'] - ligand_structure['atoms'].mean(axis=0)
            ligand_atoms = ligand_atoms_centered + ligand_pos
            
            # --- Drawing Protein (Conceptual) ---
            protein_atoms = protein_structure['atoms'] + protein_pos
            protein_color = '#3498db'
            ax_main.scatter(protein_atoms[:, 0], protein_atoms[:, 1], protein_atoms[:, 2],
                          c=protein_color, s=50, alpha=0.4)
            for bond in protein_structure['bonds']:
                p1, p2 = protein_atoms[bond[0]], protein_atoms[bond[1]]
                ax_main.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                           color=protein_color, alpha=0.3, linewidth=1.5)

            # --- Drawing Ligand (Accurate) ---
            # Draw ligand atoms with specific colors
            for i, symbol in enumerate(ligand_structure['symbols']):
                color = ATOM_COLORS.get(symbol, ATOM_COLORS['DEFAULT'])
                atom_pos = ligand_atoms[i]
                ax_main.scatter(atom_pos[0], atom_pos[1], atom_pos[2],
                              c=color, s=80, alpha=0.9, edgecolors='w', linewidth=0.5)
            
            # Draw ligand bonds
            for bond in ligand_structure['bonds']:
                a1, a2 = ligand_atoms[bond[0]], ligand_atoms[bond[1]]
                ax_main.plot([a1[0], a2[0]], [a1[1], a2[1]], [a1[2], a2[2]],
                           color='white', alpha=0.8, linewidth=3)
            
            # --- Interaction Lines ---
            if frame > phase2_end:
                t_interaction = min(1.0, (frame - phase2_end) / (total_frames * 0.2))
                # Draw some interaction lines
                for i in range(min(4, len(ligand_atoms))):
                    closest_protein_idx = np.argmin(np.linalg.norm(
                        protein_atoms - ligand_atoms[i], axis=1))
                    
                    ax_main.plot([ligand_atoms[i][0], protein_atoms[closest_protein_idx][0]],
                               [ligand_atoms[i][1], protein_atoms[closest_protein_idx][1]],
                               [ligand_atoms[i][2], protein_atoms[closest_protein_idx][2]],
                               '#FFD700', linestyle='--', alpha=0.6 * t_interaction, linewidth=1.5)
            
            # Update info panels
            self._update_info_panels(ax_info1, ax_info2, ax_info3, frame, total_frames,
                                   protein_name, ligand_smiles, binding_affinity, confidence,
                                   target_protein)
            
            return []
        
        try:
            anim = FuncAnimation(fig, animate_frame, frames=self.frames, interval=1000/self.fps, blit=False)
            
            logger.info(f"Saving animation to {video_path}")
            writer = PillowWriter(fps=self.fps)
            anim.save(video_path, writer=writer, dpi=120)
            logger.info(f"Animation saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving animation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            plt.close(fig)
            return None
        finally:
            plt.close(fig)
        
        logger.info(f"Animation saved to: {video_path}")
        return str(video_path)
    
    def _update_info_panels(self, ax1, ax2, ax3, frame, total_frames,
                           protein_name, ligand_smiles, binding_affinity, confidence,
                           target_protein):
        for ax in [ax1, ax2, ax3]:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        # Panel 1: Target Information
        ax1.text(0.5, 0.85, "TARGET INFO", ha='center', color='white', fontsize=14, weight='bold')
        ax1.text(0.5, 0.60, f"Original Target:\n{target_protein}", ha='center', va='top', color='#CCCCCC', fontsize=10, wrap=True)
        ax1.text(0.5, 0.25, f"Homologous Protein:\n{protein_name}", ha='center', va='top', color='#CCCCCC', fontsize=10, wrap=True)
        
        # Panel 2: Binding Metrics
        ax2.text(0.5, 0.85, "BINDING METRICS", ha='center', color='white', fontsize=14, weight='bold')
        ax2.text(0.5, 0.60, f"Affinity (pKd): {binding_affinity:.2f}", ha='center', color='#4CAF50', fontsize=12)
        ax2.text(0.5, 0.40, f"Confidence: {confidence*100:.1f}%", ha='center', color='#2196F3', fontsize=12)
        
        # Progress bar
        progress = frame / total_frames
        ax2.add_patch(patches.Rectangle((0.1, 0.1), 0.8, 0.1, facecolor='#444444', transform=ax2.transAxes))
        ax2.add_patch(patches.Rectangle((0.1, 0.1), 0.8*progress, 0.1, facecolor='#00BCD4', alpha=0.7, transform=ax2.transAxes))
        
        # Panel 3: Ligand Information
        ax3.text(0.5, 0.85, "LIGAND", ha='center', color='white', fontsize=14, weight='bold')
        display_smiles = ligand_smiles[:30] + "..." if len(ligand_smiles) > 30 else ligand_smiles
        ax3.text(0.5, 0.5, f"SMILES:\n{display_smiles}", ha='center', va='top', color='#CCCCCC', fontsize=9, family='monospace', wrap=True)

if __name__ == "__main__":
    # Test the animation system with a common molecule (Aspirin)
    animator = ProteinLigandAnimator()
    
    test_recommendation = {
        'target_protein': 'Protein kinase A',
        'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'binding_affinity': 6.8,
        'confidence': 0.92
    }
    
    video_path = animator.animate_binding(
        protein_name=test_recommendation['target_protein'],
        ligand_smiles=test_recommendation['smiles'],
        binding_affinity=test_recommendation['binding_affinity'],
        confidence=test_recommendation['confidence'],
        target_protein="Cyclooxygenase Enzyme Family"
    )
    
    if video_path:
        print(f"Test animation created successfully: {video_path}")
    else:
        print("Failed to create test animation.")