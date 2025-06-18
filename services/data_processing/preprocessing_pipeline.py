"""
Data Preprocessing Pipeline

Standalone, reusable components for data preprocessing with clear
input/output contracts and modular design.
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from abc import ABC, abstractmethod
import cv2
from PIL import Image
import json

from config import get_config
from utils.logging_config import get_logger, monitor_performance


class PreprocessorInterface(ABC):
    """Abstract interface for all preprocessors."""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get preprocessor configuration."""
        pass


class ImagePreprocessor(PreprocessorInterface):
    """Image preprocessing component with configurable transformations."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 augment: bool = False,
                 preserve_aspect_ratio: bool = False):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (width, height)
            normalize: Whether to normalize pixel values to [0, 1]
            augment: Whether to apply data augmentation
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
        """
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.logger = get_logger('preprocessing.image')
        
        # Augmentation parameters
        self.augmentation_params = {
            'rotation_range': 15,
            'brightness_range': 0.2,
            'zoom_range': 0.1,
            'horizontal_flip': True
        }
    
    @monitor_performance("image_preprocessing")
    def process(self, data: Any) -> Optional[np.ndarray]:
        """Process image data."""
        try:
            # Load image
            image = self._load_image(data)
            if image is None:
                return None
            
            # Resize image
            image = self._resize_image(image)
            
            # Apply augmentation if enabled
            if self.augment:
                image = self._apply_augmentation(image)
            
            # Normalize if enabled
            if self.normalize:
                image = self._normalize_image(image)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return None
    
    def _load_image(self, data: Any) -> Optional[np.ndarray]:
        """Load image from various input types."""
        try:
            if isinstance(data, str):
                # File path
                return self._load_from_path(data)
            elif isinstance(data, np.ndarray):
                # NumPy array
                return data
            elif hasattr(data, 'read'):
                # File-like object
                return self._load_from_file_object(data)
            elif isinstance(data, bytes):
                # Raw bytes
                return self._load_from_bytes(data)
            else:
                self.logger.error(f"Unsupported input type: {type(data)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None
    
    def _load_from_path(self, path: str) -> Optional[np.ndarray]:
        """Load image from file path."""
        if not os.path.exists(path):
            self.logger.error(f"Image file not found: {path}")
            return None
        
        try:
            # Try OpenCV first
            if 'cv2' in globals():
                image = cv2.imread(path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Fallback to PIL
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
                
        except Exception as e:
            self.logger.error(f"Error loading image from {path}: {e}")
            return None
    
    def _load_from_file_object(self, file_obj) -> Optional[np.ndarray]:
        """Load image from file-like object."""
        try:
            with Image.open(file_obj) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
        except Exception as e:
            self.logger.error(f"Error loading image from file object: {e}")
            return None
    
    def _load_from_bytes(self, data: bytes) -> Optional[np.ndarray]:
        """Load image from raw bytes."""
        try:
            # Try OpenCV first
            if 'cv2' in globals():
                nparr = np.frombuffer(data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Fallback to PIL
            from io import BytesIO
            with Image.open(BytesIO(data)) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
                
        except Exception as e:
            self.logger.error(f"Error loading image from bytes: {e}")
            return None
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        if image.shape[:2] == self.target_size[::-1]:  # Height, Width vs Width, Height
            return image
        
        if self.preserve_aspect_ratio:
            return self._resize_with_aspect_ratio(image)
        else:
            return self._resize_direct(image)
    
    def _resize_direct(self, image: np.ndarray) -> np.ndarray:
        """Direct resize without preserving aspect ratio."""
        if 'cv2' in globals():
            return cv2.resize(image, self.target_size)
        else:
            # Fallback using PIL
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(self.target_size)
            return np.array(pil_image)
    
    def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """Resize image while preserving aspect ratio."""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        if 'cv2' in globals():
            resized = cv2.resize(image, (new_w, new_h))
        else:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((new_w, new_h))
            resized = np.array(pil_image)
        
        # Pad to target size
        padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation transformations."""
        import random
        
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-self.augmentation_params['rotation_range'], 
                                 self.augmentation_params['rotation_range'])
            image = self._rotate_image(image, angle)
        
        # Random brightness adjustment
        if random.random() < 0.5:
            factor = random.uniform(1 - self.augmentation_params['brightness_range'],
                                  1 + self.augmentation_params['brightness_range'])
            image = np.clip(image * factor, 0, 255).astype(image.dtype)
        
        # Random horizontal flip
        if self.augmentation_params['horizontal_flip'] and random.random() < 0.5:
            image = np.fliplr(image)
        
        return image
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        if 'cv2' in globals():
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, matrix, (w, h))
        else:
            # Fallback using PIL
            pil_image = Image.fromarray(image)
            rotated = pil_image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
            return np.array(rotated)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values."""
        return image.astype(np.float32) / 255.0
    
    def get_config(self) -> Dict[str, Any]:
        """Get preprocessor configuration."""
        return {
            'target_size': self.target_size,
            'normalize': self.normalize,
            'augment': self.augment,
            'preserve_aspect_ratio': self.preserve_aspect_ratio,
            'augmentation_params': self.augmentation_params
        }


class TextPreprocessor(PreprocessorInterface):
    """Text preprocessing component for NLP tasks."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_stopwords: bool = False,
                 max_length: Optional[int] = None):
        """
        Initialize text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_stopwords: Remove common stopwords
            max_length: Maximum text length (truncate if longer)
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.max_length = max_length
        self.logger = get_logger('preprocessing.text')
    
    @monitor_performance("text_preprocessing")
    def process(self, data: str) -> str:
        """Process text data."""
        try:
            text = str(data)
            
            if self.lowercase:
                text = text.lower()
            
            if self.remove_punctuation:
                import string
                text = text.translate(str.maketrans('', '', string.punctuation))
            
            if self.remove_stopwords:
                text = self._remove_stopwords(text)
            
            if self.max_length:
                text = text[:self.max_length]
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return ""
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove common English stopwords."""
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their'
        }
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        return ' '.join(filtered_words)
    
    def get_config(self) -> Dict[str, Any]:
        """Get preprocessor configuration."""
        return {
            'lowercase': self.lowercase,
            'remove_punctuation': self.remove_punctuation,
            'remove_stopwords': self.remove_stopwords,
            'max_length': self.max_length
        }


class PreprocessingPipeline:
    """Pipeline that chains multiple preprocessors together."""

    def __init__(self, preprocessors: List[PreprocessorInterface]):
        """
        Initialize preprocessing pipeline.

        Args:
            preprocessors: List of preprocessors to apply in order
        """
        self.preprocessors = preprocessors
        self.logger = get_logger('preprocessing.pipeline')

    @monitor_performance("preprocessing_pipeline")
    def process(self, data: Any) -> Any:
        """Process data through all preprocessors in sequence."""
        try:
            result = data
            for i, preprocessor in enumerate(self.preprocessors):
                result = preprocessor.process(result)
                if result is None:
                    self.logger.error(f"Preprocessor {i} ({type(preprocessor).__name__}) returned None")
                    return None

            return result

        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {e}")
            return None

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for all preprocessors in the pipeline."""
        return {
            f"preprocessor_{i}": {
                'type': type(preprocessor).__name__,
                'config': preprocessor.get_config()
            }
            for i, preprocessor in enumerate(self.preprocessors)
        }
