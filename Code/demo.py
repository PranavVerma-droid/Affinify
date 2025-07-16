#!/usr/bin/env python3
"""
Demo script for Affinify project
Demonstrates core functionality with minimal dependencies
"""

import os
import sys
import pandas as pd #type: ignore
import numpy as np
from pathlib import Path

print("🧬 Affinify Demo - AI-Powered Protein-Ligand Binding Affinity Predictor")
print("=" * 70)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_dependencies():
    """Check available dependencies"""
    print("Checking available dependencies...")
    
    deps = {
        'pandas': False,
        'numpy': False,
        'scikit-learn': False,
        'matplotlib': False,
        'streamlit': False
    }
    
    for dep in deps:
        try:
            __import__(dep)
            deps[dep] = True
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
    
    return deps

def demo_data_processing():
    """Demonstrate data processing capabilities"""
    print("\n" + "=" * 40)
    print("📊 Data Processing Demo")
    print("=" * 40)
    
    try:
        from data_processing.data_collector import DataCollector #type: ignore
        from data_processing.feature_extractor import MolecularFeatureExtractor #type: ignore
        
        # Create sample data
        print("Creating sample dataset...")
        collector = DataCollector()
        sample_df = collector.create_sample_dataset(100)
        
        print(f"Sample dataset shape: {sample_df.shape}")
        print("Sample columns:", list(sample_df.columns))
        print("\nFirst few rows:")
        print(sample_df.head())
        
        # Extract features
        print("\nExtracting molecular features...")
        extractor = MolecularFeatureExtractor()
        features, target = extractor.prepare_features(sample_df)
        
        print(f"Features shape: {features.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Sample features: {list(features.columns[:5])}")
        
        return features, target
        
    except Exception as e:
        print(f"Data processing demo failed: {e}")
        return None, None

def demo_model_training(features, target):
    """Demonstrate model training"""
    print("\n" + "=" * 40)
    print("🤖 Model Training Demo")
    print("=" * 40)
    
    if features is None or target is None:
        print("No data available for training")
        return None
    
    try:
        from models.ml_models import RandomForestModel #type: ignore
        from sklearn.model_selection import train_test_split #type: ignore
        from sklearn.metrics import r2_score, mean_squared_error #type: ignore
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train model
        print("Training Random Forest model...")
        model = RandomForestModel(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Model performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        # Show feature importance
        if hasattr(model, 'get_feature_importance'):
            importance_df = model.get_feature_importance()
            print(f"\nTop 5 important features:")
            for _, row in importance_df.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return model
        
    except Exception as e:
        print(f"Model training demo failed: {e}")
        return None

def demo_prediction(model, features):
    """Demonstrate prediction capabilities"""
    print("\n" + "=" * 40)
    print("🔬 Prediction Demo")
    print("=" * 40)
    
    if model is None or features is None:
        print("No trained model available for prediction")
        return
    
    try:
        # Make predictions on sample data
        sample_features = features.head(5)
        predictions = model.predict(sample_features)
        
        print("Sample predictions:")
        for i, pred in enumerate(predictions):
            print(f"  Sample {i+1}: {pred:.3f} (pKd/pKi/pIC50)")
        
        # Interpret predictions
        print("\nInterpretation:")
        for i, pred in enumerate(predictions):
            if pred > 7:
                strength = "Strong binding"
            elif pred > 5:
                strength = "Moderate binding"
            else:
                strength = "Weak binding"
            print(f"  Sample {i+1}: {strength}")
        
    except Exception as e:
        print(f"Prediction demo failed: {e}")

def demo_visualization():
    """Demonstrate visualization capabilities"""
    print("\n" + "=" * 40)
    print("📈 Visualization Demo")
    print("=" * 40)
    
    try:
        import matplotlib.pyplot as plt #type: ignore
        
        # Create sample data
        np.random.seed(42)
        actual = np.random.normal(6, 2, 100)
        predicted = actual + np.random.normal(0, 0.5, 100)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.scatter(actual, predicted, alpha=0.6)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel('Actual Binding Affinity')
        plt.ylabel('Predicted Binding Affinity')
        plt.title('Predictions vs Actual (Demo)')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('demo_prediction_plot.png', dpi=150, bbox_inches='tight')
        print("Sample prediction plot saved as 'demo_prediction_plot.png'")
        plt.close()
        
        print("Visualization demo completed successfully!")
        
    except Exception as e:
        print(f"Visualization demo failed: {e}")

def main():
    """Main demo function"""
    # Check dependencies
    deps = check_dependencies()
    
    if not deps['pandas'] or not deps['numpy']:
        print("\n❌ Critical dependencies missing. Please install pandas and numpy.")
        return
    
    # Run demos
    features, target = demo_data_processing()
    model = demo_model_training(features, target)
    demo_prediction(model, features)
    demo_visualization()
    
    print("\n" + "=" * 70)
    print("🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run full data download: python scripts/download_data.py")
    print("2. Train complete models: python scripts/train_models.py")
    print("3. Launch web app: streamlit run app/main.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()