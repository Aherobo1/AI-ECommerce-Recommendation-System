"""
Data Processing Package

This package contains all data processing components including:
- Data cleaning and preprocessing
- OCR services
- Web scraping utilities
- Vector database operations
"""

from .preprocessing_pipeline import ImagePreprocessor, TextPreprocessor, PreprocessingPipeline

# Import existing services with fallback
try:
    from ..data_cleaning import DataCleaner
except ImportError:
    DataCleaner = None

try:
    from ..ocr_service import OCRService as OCRProcessor
except ImportError:
    OCRProcessor = None

try:
    from ..scraper import WebScraper
except ImportError:
    WebScraper = None

try:
    from ..vector_db import VectorDatabase
except ImportError:
    VectorDatabase = None

__all__ = [
    'ImagePreprocessor',
    'TextPreprocessor',
    'PreprocessingPipeline',
    'DataCleaner',
    'OCRProcessor',
    'WebScraper',
    'VectorDatabase'
]
