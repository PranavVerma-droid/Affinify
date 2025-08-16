#!/usr/bin/env python3
"""
Animation CLI for Affinify
Command-line interface for generating protein-ligand binding animations.
"""

import argparse
import sys
import json
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

# Check imports before proceeding
try:
    from visualization.binding_anim import ProteinLigandAnimator
    print("✓ Successfully imported ProteinLigandAnimator")
except ImportError as e:
    print(f"✗ Error importing ProteinLigandAnimator: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Generate protein-ligand binding animations')
    
    parser.add_argument('--protein', required=True, help='Target protein name')
    parser.add_argument('--smiles', required=True, help='Ligand SMILES string')
    parser.add_argument('--affinity', type=float, required=True, help='Binding affinity (pKd)')
    parser.add_argument('--confidence', type=float, required=True, help='Prediction confidence (0-1)')
    parser.add_argument('--target', required=True, help='Original target protein name')
    parser.add_argument('--output-dir', default='videos', help='Output directory for videos')
    parser.add_argument('--output-json', help='Output JSON file with result path')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not (0 <= args.confidence <= 1):
        print("Error: Confidence must be between 0 and 1")
        sys.exit(1)
    
    if len(args.smiles) < 2:
        print("Error: Invalid SMILES string")
        sys.exit(1)
    
    try:
        # Create animator
        print(f"Creating animator with output directory: {args.output_dir}")
        animator = ProteinLigandAnimator(output_dir=args.output_dir)
        
        # Generate animation
        print(f"Generating animation for:")
        print(f"  Protein: {args.protein}")
        print(f"  SMILES: {args.smiles}")
        print(f"  Affinity: {args.affinity}")
        print(f"  Confidence: {args.confidence}")
        print(f"  Target: {args.target}")
        print("This process may take 2-3 minutes...")
        
        video_path = animator.animate_binding(
            protein_name=args.protein,
            ligand_smiles=args.smiles,
            binding_affinity=args.affinity,
            confidence=args.confidence,
            target_protein=args.target
        )
        
        if video_path:
            print(f"✓ Animation saved to: {video_path}")
            
            # Save result to JSON if requested
            if args.output_json:
                result = {
                    'success': True,
                    'video_path': video_path,
                    'protein': args.protein,
                    'smiles': args.smiles,
                    'affinity': args.affinity,
                    'confidence': args.confidence
                }
                with open(args.output_json, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"✓ Result saved to JSON: {args.output_json}")
            
            sys.exit(0)
        else:
            print("✗ Error: Failed to generate animation")
            
            # Save error to JSON if requested
            if args.output_json:
                result = {
                    'success': False,
                    'error': 'Animation generation failed'
                }
                with open(args.output_json, 'w') as f:
                    json.dump(result, f, indent=2)
            
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        
        # Save error to JSON if requested
        if args.output_json:
            result = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            with open(args.output_json, 'w') as f:
                json.dump(result, f, indent=2)
        
        sys.exit(1)

if __name__ == "__main__":
    main()