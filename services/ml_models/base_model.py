"""
Base Model Interface and Abstract Classes

Defines common interfaces for all ML models to ensure consistency
and enable easy swapping of implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from datetime import datetime

from config import get_config
from utils.logging_config import get_logger, monitor_performance


@dataclass
class ModelMetadata:
    """Metadata for model tracking and versioning."""
    name: str
    version: str
    created_at: datetime
    model_type: str
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    parameters: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    description: Optional[str] = None


@dataclass
class PredictionResult:
    """Standardized prediction result format."""
    prediction: Any
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None


class ModelInterface(ABC):
    """Abstract interface that all models must implement."""
    
    @abstractmethod
    def predict(self, input_data: Any) -> PredictionResult:
        """Make a prediction on input data."""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load model from file."""
        pass
    
    @abstractmethod
    def save_model(self, model_path: str) -> bool:
        """Save model to file."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelMetadata:
        """Get model metadata and information."""
        pass


class BaseModel(ModelInterface):
    """Base class providing common functionality for all models."""
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        """Initialize base model."""
        self.config = get_config()
        self.logger = get_logger(f"models.{model_name}")
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.metadata = ModelMetadata(
            name=model_name,
            version=model_version,
            created_at=datetime.now(),
            model_type=self.__class__.__name__
        )
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for predictions."""
        return self._is_loaded and self.model is not None
    
    @monitor_performance("model_prediction")
    def predict(self, input_data: Any) -> PredictionResult:
        """Make a prediction with performance monitoring."""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} is not loaded")
        
        import time
        start_time = time.time()
        
        try:
            result = self._predict_impl(input_data)
            processing_time = time.time() - start_time
            
            # Log prediction
            from utils.logging_config import log_model_prediction
            log_model_prediction(
                self.model_name,
                type(input_data).__name__,
                result.confidence,
                str(result.prediction)[:100]  # Truncate long predictions
            )
            
            result.processing_time = processing_time
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    @abstractmethod
    def _predict_impl(self, input_data: Any) -> PredictionResult:
        """Implementation-specific prediction logic."""
        pass
    
    def load_model(self, model_path: str) -> bool:
        """Load model with error handling and logging."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            success = self._load_model_impl(str(model_path))
            if success:
                self._is_loaded = True
                self.logger.info(f"Model {self.model_name} loaded successfully from {model_path}")
                
                # Load metadata if available
                metadata_path = model_path.with_suffix('.metadata.json')
                if metadata_path.exists():
                    self._load_metadata(str(metadata_path))
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    @abstractmethod
    def _load_model_impl(self, model_path: str) -> bool:
        """Implementation-specific model loading logic."""
        pass
    
    def save_model(self, model_path: str) -> bool:
        """Save model with metadata."""
        try:
            if not self.is_loaded:
                self.logger.error("Cannot save unloaded model")
                return False
            
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            success = self._save_model_impl(str(model_path))
            
            if success:
                # Save metadata
                metadata_path = model_path.with_suffix('.metadata.json')
                self._save_metadata(str(metadata_path))
                self.logger.info(f"Model {self.model_name} saved to {model_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    @abstractmethod
    def _save_model_impl(self, model_path: str) -> bool:
        """Implementation-specific model saving logic."""
        pass
    
    def get_model_info(self) -> ModelMetadata:
        """Get model metadata."""
        return self.metadata
    
    def update_metadata(self, **kwargs):
        """Update model metadata."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
    
    def _load_metadata(self, metadata_path: str):
        """Load metadata from JSON file."""
        try:
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Update metadata fields
            for key, value in metadata_dict.items():
                if hasattr(self.metadata, key):
                    if key == 'created_at':
                        value = datetime.fromisoformat(value)
                    setattr(self.metadata, key, value)
                    
        except Exception as e:
            self.logger.warning(f"Could not load metadata: {e}")
    
    def _save_metadata(self, metadata_path: str):
        """Save metadata to JSON file."""
        try:
            metadata_dict = {
                'name': self.metadata.name,
                'version': self.metadata.version,
                'created_at': self.metadata.created_at.isoformat(),
                'model_type': self.metadata.model_type,
                'input_shape': self.metadata.input_shape,
                'output_shape': self.metadata.output_shape,
                'parameters': self.metadata.parameters,
                'performance_metrics': self.metadata.performance_metrics,
                'description': self.metadata.description
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not save metadata: {e}")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data format."""
        # Override in subclasses for specific validation
        return input_data is not None
    
    def preprocess_input(self, input_data: Any) -> Any:
        """Preprocess input data before prediction."""
        # Override in subclasses for specific preprocessing
        return input_data
    
    def postprocess_output(self, raw_output: Any) -> Any:
        """Postprocess model output."""
        # Override in subclasses for specific postprocessing
        return raw_output
