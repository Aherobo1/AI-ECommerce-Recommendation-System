#!/usr/bin/env python3
"""
Environment Setup Script

Ensures all dependencies are installed and the environment is properly configured
for the ManaKnight AI E-Commerce Recommendation System.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_requirements():
    """Check and install required packages."""
    print("🔧 Checking and installing required packages...")
    
    # Essential packages for the improvements
    essential_packages = [
        "python-dotenv",
        "numpy",
        "pandas", 
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "Pillow"
    ]
    
    installed = []
    failed = []
    
    for package in essential_packages:
        print(f"  Installing {package}...")
        if install_package(package):
            installed.append(package)
            print(f"    ✅ {package} installed successfully")
        else:
            failed.append(package)
            print(f"    ❌ Failed to install {package}")
    
    print(f"\n📊 Installation Summary:")
    print(f"  ✅ Installed: {len(installed)}")
    print(f"  ❌ Failed: {len(failed)}")
    
    if failed:
        print(f"  Failed packages: {failed}")
        print("  You may need to install these manually")
    
    return len(failed) == 0

def install_from_requirements():
    """Install packages from requirements.txt if available."""
    requirements_file = Path("requirements.txt")
    
    if requirements_file.exists():
        print("📋 Installing from requirements.txt...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("  ✅ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Some packages from requirements.txt failed to install: {e}")
            return False
    else:
        print("  ⚠️  requirements.txt not found, installing essential packages only")
        return check_and_install_requirements()

def create_directories():
    """Create necessary directories."""
    print("📁 Creating necessary directories...")
    
    directories = [
        "data",
        "data/evaluation_results",
        "data/benchmark_results",
        "logs",
        "models",
        "static/images/evaluation",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created/verified: {directory}")

def create_env_file():
    """Create a sample .env file if it doesn't exist."""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("🔧 Creating sample .env file...")
        
        env_content = """# ManaKnight AI E-Commerce Recommendation System Configuration

# Flask Configuration
FLASK_ENV=development
SECRET_KEY=dev-secret-key-change-in-production
HOST=0.0.0.0
PORT=5000

# Database Configuration
DATABASE_PATH=data/ecommerce.db
DB_TIMEOUT=30
DB_MAX_CONNECTIONS=10

# Model Configuration
MODELS_DIR=models
CNN_MODEL_PATH=models/cnn_product_classifier.h5
VECTOR_MODEL_PATH=models/product_vectors.pkl
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.001
VALIDATION_SPLIT=0.2

# Data Configuration
DATA_DIR=data
SCRAPED_IMAGES_DIR=data/scraped_images
LOGS_DIR=logs
MIN_IMAGES_PER_CLASS=10
MAX_FEATURES_TFIDF=5000

# API Configuration
MAX_FILE_SIZE=16777216
UPLOAD_FOLDER=static/uploads
ALLOWED_EXTENSIONS=jpg,jpeg,png,gif,bmp,tiff

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE=logs/app.log
LOG_MAX_FILE_SIZE=10485760
LOG_BACKUP_COUNT=5

# Vector Database Configuration (Optional)
# PINECONE_API_KEY=your_pinecone_api_key
# PINECONE_ENVIRONMENT=your_pinecone_environment
# PINECONE_INDEX_NAME=product-recommendations
# VECTOR_DIMENSION=384
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("  ✅ Sample .env file created")
        print("  📝 You can customize the configuration in .env file")
    else:
        print("  ✅ .env file already exists")

def verify_installation():
    """Verify that the installation was successful."""
    print("🔍 Verifying installation...")
    
    try:
        # Test configuration import
        from config import get_config
        config = get_config()
        print("  ✅ Configuration system working")
        
        # Test logging import
        from utils.logging_config import get_logger
        logger = get_logger('setup_test')
        print("  ✅ Logging system working")
        
        # Test base model import
        from services.ml_models.base_model import BaseModel, PredictionResult
        print("  ✅ Base model interface working")
        
        # Test preprocessing import
        from services.data_processing.preprocessing_pipeline import ImagePreprocessor
        print("  ✅ Preprocessing pipeline working")
        
        print("  🎉 All core components verified successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 MANAKNIGHT AI E-COMMERCE RECOMMENDATION SYSTEM")
    print("🔧 Environment Setup Script")
    print("=" * 60)
    print()
    
    # Step 1: Create directories
    create_directories()
    print()
    
    # Step 2: Create .env file
    create_env_file()
    print()
    
    # Step 3: Install dependencies
    success = install_from_requirements()
    print()
    
    # Step 4: Verify installation
    if success:
        verification_success = verify_installation()
        print()
        
        if verification_success:
            print("🎉 SETUP COMPLETED SUCCESSFULLY!")
            print("✅ Your environment is ready for the interview!")
            print()
            print("📋 Next steps:")
            print("  1. Run: python test_integration.py")
            print("  2. Open: notebooks/comprehensive_model_evaluation.ipynb")
            print("  3. Review: IMPROVEMENTS_SUMMARY.md")
            print()
            print("🚀 You're ready to showcase your improvements!")
            return True
        else:
            print("⚠️  Setup completed but verification failed.")
            print("   Some components may not work correctly.")
            return False
    else:
        print("❌ Setup failed due to dependency installation issues.")
        print("   Please install dependencies manually and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
