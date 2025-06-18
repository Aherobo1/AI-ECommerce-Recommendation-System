"""
Machine Learning Models Package

This package contains all ML model implementations including:
- CNN models for image classification
- Recommendation engines
- Vector databases and embeddings
"""

from .base_model import BaseModel, ModelInterface
from .cnn_classifier import CNNClassifier

# Import existing recommendation engine with fallback
try:
    from ..recommendation import RecommendationEngine
except ImportError:
    RecommendationEngine = None

__all__ = [
    'BaseModel',
    'ModelInterface',
    'CNNClassifier',
    'RecommendationEngine'
]
