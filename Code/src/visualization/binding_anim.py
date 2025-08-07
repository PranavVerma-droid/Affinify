#!/usr/bin/env python3
"""
Binding Animation Module for Affinify
Creates 3D animated visualizations of protein-ligand binding interactions
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
import colorsys
import time

# Configure logging
logger = logging.getLogger(__name__)

class ProteinLigandAnimator:
    """Creates animated visualizations of protein-ligand binding interactions"""
    
    def __init__(self, output_dir: str = "videos"):
        """
        Initialize the animator.
        
        Args:
            output_dir: Directory to save animation videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Animation parameters
        self.fps = 30  # Reduced for faster processing
        self.duration = 8  # Reduced duration
        self.frames = self.fps * self.duration
        
        # Color schemes
        self.protein_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        self.ligand_colors = ['#e67e22', '#1abc9c', '#34495e', '#e91e63', '#ff5722']
        
    def create_molecule_structure(self, size: str = "medium") -> Dict:
        """
        Create a simplified 3D molecular structure representation.
        
        Args:
            size: Size of the molecule ("small", "medium", "large")
            
        Returns:
            Dictionary containing atom positions and bonds
        """
        if size == "small":
            n_atoms = random.randint(5, 12)
        elif size == "medium":
            n_atoms = random.randint(15, 30)
        else:  # large
            n_atoms = random.randint(40, 80)
            
        # Generate random but structured atom positions
        atoms = []
        bonds = []
        
        # Create a core structure
        center = np.array([0, 0, 0])
        
        # Add core atoms
        for i in range(min(5, n_atoms)):
            angle = i * 2 * np.pi / 5
            radius = 1.0
            pos = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
            atoms.append(pos)
            
            # Connect to center (if we add a center atom)
            if i > 0:
                bonds.append([0, i])
                
        # Add peripheral atoms
        for i in range(5, n_atoms):
            # Connect to random existing atom
            parent_idx = random.randint(0, min(i-1, len(atoms)-1))
            parent_pos = atoms[parent_idx]
            
            # Random direction with some bias toward spreading out
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            # Bond length
            bond_length = random.uniform(0.8, 1.5)
            pos = parent_pos + bond_length * direction
            
            atoms.append(pos)
            bonds.append([parent_idx, i])
            
        return {
            'atoms': np.array(atoms),
            'bonds': bonds,
            'n_atoms': n_atoms
        }
    
    def animate_binding(self, protein_name: str, ligand_smiles: str, 
                       binding_affinity: float, confidence: float,
                       target_protein: str) -> str:
        """
        Create binding animation between protein and ligand.
        
        Args:
            protein_name: Name of the target protein
            ligand_smiles: SMILES string of the ligand
            binding_affinity: Predicted binding affinity
            confidence: Prediction confidence
            target_protein: Original target protein name
            
        Returns:
            Path to the generated video file
        """
        # Generate unique filename
        video_id = str(uuid.uuid4())[:8]
        video_path = self.output_dir / f"binding_{video_id}.gif"
        
        logger.info(f"Creating binding animation: {protein_name} + {ligand_smiles[:20]}...")
        
        # Create molecular structures
        protein_structure = self.create_molecule_structure("large")
        ligand_structure = self.create_molecule_structure("small")
        
        # Set up the figure
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid layout
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[3, 1], width_ratios=[1, 1, 1])
        
        # Main 3D animation subplot
        ax_main = fig.add_subplot(gs[0, :], projection='3d')
        
        # Info panels
        ax_info1 = fig.add_subplot(gs[1, 0])
        ax_info2 = fig.add_subplot(gs[1, 1])
        ax_info3 = fig.add_subplot(gs[1, 2])
        
        # Remove axes for info panels
        for ax in [ax_info1, ax_info2, ax_info3]:
            ax.set_xticks([])
            ax.set_yticks([])
            
        # Set up 3D plot
        ax_main.set_xlim([-8, 8])
        ax_main.set_ylim([-8, 8])
        ax_main.set_zlim([-8, 8])
        ax_main.set_title("Protein-Ligand Binding Simulation", fontsize=16, pad=20)
        
        # Colors
        protein_color = random.choice(self.protein_colors)
        ligand_color = random.choice(self.ligand_colors)
        
        def animate_frame(frame):
            """Animation function for each frame"""
            ax_main.clear()
            ax_main.set_xlim([-8, 8])
            ax_main.set_ylim([-8, 8])
            ax_main.set_zlim([-8, 8])
            ax_main.set_title("Protein-Ligand Binding Simulation", fontsize=16, pad=20)
            
            # Animation phases
            total_frames = self.frames
            phase1_end = total_frames * 0.2  # Initial separation
            phase2_end = total_frames * 0.6  # Approach
            phase3_end = total_frames * 0.8  # Binding
            phase4_end = total_frames * 1.0  # Bound state
            
            if frame < phase1_end:
                # Phase 1: Show separated molecules
                t = frame / phase1_end
                
                # Protein position (left side)
                protein_pos = np.array([-5, 0, 0])
                # Ligand position (right side)
                ligand_pos = np.array([5, 0, 0])
                
                # Add some rotation
                rotation_angle = t * 2 * np.pi
                
            elif frame < phase2_end:
                # Phase 2: Approach each other
                t = (frame - phase1_end) / (phase2_end - phase1_end)
                
                # Smooth approach using easing function
                ease_t = 1 - (1 - t) ** 3  # Ease-out cubic
                
                protein_pos = np.array([-5 + 3 * ease_t, 0, 0])
                ligand_pos = np.array([5 - 4 * ease_t, 0, 0])
                
                rotation_angle = t * 4 * np.pi
                
            elif frame < phase3_end:
                # Phase 3: Binding interaction
                t = (frame - phase2_end) / (phase3_end - phase2_end)
                
                # Final approach and binding
                protein_pos = np.array([-2, 0.2 * np.sin(t * 8 * np.pi), 0])
                ligand_pos = np.array([1 - 0.5 * t, 0.1 * np.sin(t * 10 * np.pi), 0])
                
                rotation_angle = t * 6 * np.pi
                
            else:
                # Phase 4: Bound state
                t = (frame - phase3_end) / (phase4_end - phase3_end)
                
                # Stable bound complex with slight movement
                protein_pos = np.array([-2, 0.1 * np.sin(t * 4 * np.pi), 0])
                ligand_pos = np.array([0.5, 0.05 * np.sin(t * 6 * np.pi), 0])
                
                rotation_angle = t * 2 * np.pi
                
            # Draw protein
            protein_atoms = protein_structure['atoms'] + protein_pos
            ax_main.scatter(protein_atoms[:, 0], protein_atoms[:, 1], protein_atoms[:, 2],
                          c=protein_color, s=80, alpha=0.8, label='Target Protein')
            
            # Draw protein bonds
            for bond in protein_structure['bonds']:
                atom1, atom2 = protein_atoms[bond[0]], protein_atoms[bond[1]]
                ax_main.plot([atom1[0], atom2[0]], [atom1[1], atom2[1]], [atom1[2], atom2[2]],
                           color=protein_color, alpha=0.6, linewidth=2)
            
            # Draw ligand
            ligand_atoms = ligand_structure['atoms'] + ligand_pos
            ax_main.scatter(ligand_atoms[:, 0], ligand_atoms[:, 1], ligand_atoms[:, 2],
                          c=ligand_color, s=60, alpha=0.8, label='Ligand')
            
            # Draw ligand bonds
            for bond in ligand_structure['bonds']:
                atom1, atom2 = ligand_atoms[bond[0]], ligand_atoms[bond[1]]
                ax_main.plot([atom1[0], atom2[0]], [atom1[1], atom2[1]], [atom1[2], atom2[2]],
                           color=ligand_color, alpha=0.6, linewidth=2)
            
            # Add binding site representation in phase 3 and 4
            if frame >= phase2_end:
                # Draw binding site as a translucent sphere
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x_sphere = 0.8 * np.outer(np.cos(u), np.sin(v)) + protein_pos[0] + 2
                y_sphere = 0.8 * np.outer(np.sin(u), np.sin(v)) + protein_pos[1]
                z_sphere = 0.8 * np.outer(np.ones(np.size(u)), np.cos(v)) + protein_pos[2]
                
                ax_main.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='yellow')
            
            # Add interaction lines in phases 3 and 4
            if frame >= phase3_end:
                # Draw some interaction lines
                for i in range(min(3, len(ligand_atoms))):
                    closest_protein_idx = np.argmin(np.linalg.norm(
                        protein_atoms - ligand_atoms[i], axis=1))
                    
                    ax_main.plot([ligand_atoms[i][0], protein_atoms[closest_protein_idx][0]],
                               [ligand_atoms[i][1], protein_atoms[closest_protein_idx][1]],
                               [ligand_atoms[i][2], protein_atoms[closest_protein_idx][2]],
                               'r--', alpha=0.5, linewidth=1)
            
            ax_main.legend(loc='upper right')
            ax_main.set_xlabel('X')
            ax_main.set_ylabel('Y')
            ax_main.set_zlabel('Z')
            
            # Update info panels
            self._update_info_panels(ax_info1, ax_info2, ax_info3, frame, total_frames,
                                   protein_name, ligand_smiles, binding_affinity, confidence,
                                   target_protein)
            
            return []
        
        # Create animation
        anim = FuncAnimation(fig, animate_frame, frames=self.frames, interval=1000/self.fps, blit=False)
        
        # Save animation as GIF (PIL supports this)
        try:
            writer = PillowWriter(fps=self.fps)
            anim.save(video_path, writer=writer, dpi=80)
        except Exception as e:
            logger.error(f"Error saving animation: {e}")
            # Fallback: try with lower quality
            try:
                writer = PillowWriter(fps=10)  # Lower FPS
                anim.save(video_path, writer=writer, dpi=60)
            except Exception as e2:
                logger.error(f"Fallback save also failed: {e2}")
                plt.close(fig)
                return None
        
        plt.close(fig)
        
        logger.info(f"Animation saved to: {video_path}")
        return str(video_path)
    
    def _update_info_panels(self, ax1, ax2, ax3, frame, total_frames,
                           protein_name, ligand_smiles, binding_affinity, confidence,
                           target_protein):
        """Update the information panels during animation"""
        
        # Clear panels
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Remove ticks
        for ax in [ax1, ax2, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Panel 1: Target Information
        ax1.text(0.5, 0.8, "TARGET PROTEIN", ha='center', va='center', 
                fontsize=12, weight='bold', transform=ax1.transAxes)
        ax1.text(0.5, 0.6, target_protein, ha='center', va='center', 
                fontsize=10, transform=ax1.transAxes, wrap=True)
        ax1.text(0.5, 0.4, "SIMILAR PROTEIN", ha='center', va='center', 
                fontsize=10, weight='bold', transform=ax1.transAxes)
        ax1.text(0.5, 0.2, protein_name, ha='center', va='center', 
                fontsize=9, transform=ax1.transAxes, wrap=True)
        
        # Panel 2: Binding Metrics
        ax2.text(0.5, 0.8, "BINDING METRICS", ha='center', va='center', 
                fontsize=12, weight='bold', transform=ax2.transAxes)
        ax2.text(0.5, 0.6, f"Affinity: {binding_affinity:.2f} pKd", ha='center', va='center', 
                fontsize=10, transform=ax2.transAxes)
        ax2.text(0.5, 0.4, f"Confidence: {confidence*100:.1f}%", ha='center', va='center', 
                fontsize=10, transform=ax2.transAxes)
        
        # Progress bar
        progress = frame / total_frames
        ax2.add_patch(patches.Rectangle((0.1, 0.1), 0.8*progress, 0.1, 
                                       facecolor='green', alpha=0.7, transform=ax2.transAxes))
        ax2.add_patch(patches.Rectangle((0.1, 0.1), 0.8, 0.1, 
                                       facecolor='none', edgecolor='black', 
                                       transform=ax2.transAxes))
        
        # Panel 3: Ligand Information
        ax3.text(0.5, 0.8, "LIGAND MOLECULE", ha='center', va='center', 
                fontsize=12, weight='bold', transform=ax3.transAxes)
        ax3.text(0.5, 0.6, "SMILES:", ha='center', va='center', 
                fontsize=10, weight='bold', transform=ax3.transAxes)
        
        # Truncate SMILES if too long
        display_smiles = ligand_smiles[:20] + "..." if len(ligand_smiles) > 20 else ligand_smiles
        ax3.text(0.5, 0.4, display_smiles, ha='center', va='center', 
                fontsize=9, transform=ax3.transAxes, family='monospace')
        
        # Animation phase indicator
        phase1_end = total_frames * 0.2
        phase2_end = total_frames * 0.6
        phase3_end = total_frames * 0.8
        
        if frame < phase1_end:
            phase_text = "INITIALIZING"
        elif frame < phase2_end:
            phase_text = "APPROACHING"
        elif frame < phase3_end:
            phase_text = "BINDING"
        else:
            phase_text = "BOUND COMPLEX"
            
        ax3.text(0.5, 0.2, phase_text, ha='center', va='center', 
                fontsize=10, weight='bold', color='red', transform=ax3.transAxes)

class BindingAnimationManager:
    """Manager class for handling binding animations in the Streamlit app"""
    
    def __init__(self, videos_dir: str = "videos"):
        """
        Initialize the animation manager.
        
        Args:
            videos_dir: Directory to store generated videos
        """
        self.animator = ProteinLigandAnimator(videos_dir)
        self.videos_dir = Path(videos_dir)
        self.videos_dir.mkdir(exist_ok=True)
        
    def create_binding_animation(self, recommendation: Dict, target_protein: str) -> Optional[str]:
        """
        Create a binding animation for a specific recommendation.
        
        Args:
            recommendation: Recommendation dictionary from predictor
            target_protein: Original target protein name
            
        Returns:
            Path to generated video file or None if failed
        """
        try:
            video_path = self.animator.animate_binding(
                protein_name=recommendation['target_protein'],
                ligand_smiles=recommendation['smiles'],
                binding_affinity=recommendation['binding_affinity'],
                confidence=recommendation['confidence'],
                target_protein=target_protein
            )
            return video_path
        except Exception as e:
            logger.error(f"Failed to create binding animation: {e}")
            return None
    
    def cleanup_old_videos(self, max_age_hours: int = 24):
        """
        Clean up old video files to save storage space.
        
        Args:
            max_age_hours: Maximum age of videos in hours before deletion
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for video_file in self.videos_dir.glob("*.mp4"):
                if current_time - video_file.stat().st_mtime > max_age_seconds:
                    video_file.unlink()
                    logger.info(f"Deleted old video: {video_file}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old videos: {e}")

if __name__ == "__main__":
    # Test the animation system
    animator = ProteinLigandAnimator()
    
    test_recommendation = {
        'target_protein': 'Protein Kinase A',
        'smiles': 'CC(=O)Oc1ccccc1C(=O)O',
        'binding_affinity': 7.5,
        'confidence': 0.85
    }
    
    video_path = animator.animate_binding(
        protein_name=test_recommendation['target_protein'],
        ligand_smiles=test_recommendation['smiles'],
        binding_affinity=test_recommendation['binding_affinity'],
        confidence=test_recommendation['confidence'],
        target_protein="Protein Kinase Family"
    )
    
    print(f"Test animation created: {video_path}")