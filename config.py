"""
Configuration Management for ManaKnight AI E-Commerce Recommendation System

This module provides centralized configuration management with environment-specific
settings, replacing hardcoded values throughout the codebase.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

# Try to load environment variables with fallback
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Using environment variables only.")


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_path: str = field(default_factory=lambda: os.getenv('DATABASE_PATH', 'data/ecommerce.db'))
    connection_timeout: int = field(default_factory=lambda: int(os.getenv('DB_TIMEOUT', '30')))
    max_connections: int = field(default_factory=lambda: int(os.getenv('DB_MAX_CONNECTIONS', '10')))


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    models_dir: str = field(default_factory=lambda: os.getenv('MODELS_DIR', 'models'))
    cnn_model_path: str = field(default_factory=lambda: os.getenv('CNN_MODEL_PATH', 'models/cnn_product_classifier.h5'))
    vector_model_path: str = field(default_factory=lambda: os.getenv('VECTOR_MODEL_PATH', 'models/product_vectors.pkl'))
    
    # CNN Model Parameters
    image_size: tuple = (224, 224)
    batch_size: int = field(default_factory=lambda: int(os.getenv('BATCH_SIZE', '32')))
    epochs: int = field(default_factory=lambda: int(os.getenv('EPOCHS', '50')))
    learning_rate: float = field(default_factory=lambda: float(os.getenv('LEARNING_RATE', '0.001')))
    validation_split: float = field(default_factory=lambda: float(os.getenv('VALIDATION_SPLIT', '0.2')))
    
    # Model Classes
    class_names: list = field(default_factory=lambda: [
        'antique_car', 'kitchen', 't-shirt', 'computer', 'teapot',
        'electronics', 'clothing', 'home_garden', 'automotive', 'office'
    ])


@dataclass
class APIConfig:
    """API and Flask application configuration."""
    secret_key: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'))
    max_file_size: int = field(default_factory=lambda: int(os.getenv('MAX_FILE_SIZE', '16777216')))  # 16MB
    upload_folder: str = field(default_factory=lambda: os.getenv('UPLOAD_FOLDER', 'static/uploads'))
    allowed_extensions: set = field(default_factory=lambda: set(os.getenv('ALLOWED_EXTENSIONS', 'jpg,jpeg,png,gif,bmp,tiff').split(',')))
    
    # Server Configuration
    host: str = field(default_factory=lambda: os.getenv('HOST', '0.0.0.0'))
    port: int = field(default_factory=lambda: int(os.getenv('PORT', '5000')))
    debug: bool = field(default_factory=lambda: os.getenv('FLASK_ENV', 'production') == 'development')


@dataclass
class DataConfig:
    """Data processing and storage configuration."""
    data_dir: str = field(default_factory=lambda: os.getenv('DATA_DIR', 'data'))
    scraped_images_dir: str = field(default_factory=lambda: os.getenv('SCRAPED_IMAGES_DIR', 'data/scraped_images'))
    logs_dir: str = field(default_factory=lambda: os.getenv('LOGS_DIR', 'logs'))
    
    # Data Processing Parameters
    min_images_per_class: int = field(default_factory=lambda: int(os.getenv('MIN_IMAGES_PER_CLASS', '10')))
    max_features_tfidf: int = field(default_factory=lambda: int(os.getenv('MAX_FEATURES_TFIDF', '5000')))


@dataclass
class VectorDBConfig:
    """Vector database configuration."""
    pinecone_api_key: Optional[str] = field(default_factory=lambda: os.getenv('PINECONE_API_KEY'))
    pinecone_environment: Optional[str] = field(default_factory=lambda: os.getenv('PINECONE_ENVIRONMENT'))
    index_name: str = field(default_factory=lambda: os.getenv('PINECONE_INDEX_NAME', 'product-recommendations'))
    dimension: int = field(default_factory=lambda: int(os.getenv('VECTOR_DIMENSION', '384')))


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    format: str = field(default_factory=lambda: os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    file_path: str = field(default_factory=lambda: os.getenv('LOG_FILE', 'logs/app.log'))
    max_file_size: int = field(default_factory=lambda: int(os.getenv('LOG_MAX_FILE_SIZE', '10485760')))  # 10MB
    backup_count: int = field(default_factory=lambda: int(os.getenv('LOG_BACKUP_COUNT', '5')))


@dataclass
class AppConfig:
    """Main application configuration container."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.data_dir,
            self.data.scraped_images_dir,
            self.data.logs_dir,
            self.model.models_dir,
            self.api.upload_folder,
            Path(self.logging.file_path).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information for debugging."""
        return {
            'environment': os.getenv('FLASK_ENV', 'production'),
            'debug_mode': self.api.debug,
            'database_path': self.database.db_path,
            'models_directory': self.model.models_dir,
            'upload_directory': self.api.upload_folder,
            'log_level': self.logging.level
        }


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def reload_config():
    """Reload configuration from environment variables."""
    global config
    if DOTENV_AVAILABLE:
        load_dotenv(override=True)
    config = AppConfig()
    return config


if __name__ == "__main__":
    # Test configuration
    print("Configuration loaded successfully!")
    print(f"Environment info: {config.get_env_info()}")
