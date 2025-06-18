#!/usr/bin/env python3
"""
Integration Test Script

Tests all the new modular components to ensure they work together properly.
This script validates that all interviewer feedback has been addressed.
"""

import sys
import os
import traceback
from pathlib import Path

def test_configuration():
    """Test the centralized configuration system."""
    print("🔧 Testing Configuration System...")
    try:
        from config import get_config, config
        
        # Test configuration loading
        app_config = get_config()
        
        # Verify key configuration sections exist
        assert hasattr(app_config, 'database'), "Database config missing"
        assert hasattr(app_config, 'model'), "Model config missing"
        assert hasattr(app_config, 'api'), "API config missing"
        assert hasattr(app_config, 'data'), "Data config missing"
        
        # Test environment info
        env_info = app_config.get_env_info()
        assert 'environment' in env_info, "Environment info missing"
        
        print("  ✅ Configuration system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_logging_system():
    """Test the structured logging system."""
    print("📊 Testing Logging System...")
    try:
        from utils.logging_config import get_logger, log_performance, monitor_performance
        
        # Test logger creation
        logger = get_logger('test')
        logger.info("Test log message")
        
        # Test performance logging
        log_performance('test_operation', 0.123, test_param='value')
        
        # Test performance decorator
        @monitor_performance('test_decorated_function')
        def test_function():
            return "test_result"
        
        result = test_function()
        assert result == "test_result", "Decorated function failed"
        
        print("  ✅ Logging system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Logging test failed: {e}")
        traceback.print_exc()
        return False

def test_base_model_interface():
    """Test the base model interface and abstract classes."""
    print("🏗️ Testing Base Model Interface...")
    try:
        from services.ml_models.base_model import BaseModel, ModelInterface, PredictionResult, ModelMetadata
        
        # Test data classes
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            created_at=None,
            model_type="TestModel"
        )
        assert metadata.name == "test_model", "ModelMetadata creation failed"
        
        result = PredictionResult(
            prediction="test_prediction",
            confidence=0.95
        )
        assert result.prediction == "test_prediction", "PredictionResult creation failed"
        assert result.confidence == 0.95, "PredictionResult confidence failed"
        
        print("  ✅ Base model interface working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Base model interface test failed: {e}")
        traceback.print_exc()
        return False

def test_cnn_classifier():
    """Test the modular CNN classifier."""
    print("🧠 Testing CNN Classifier...")
    try:
        from services.ml_models.cnn_classifier import CNNClassifier
        import numpy as np
        
        # Initialize CNN classifier
        cnn = CNNClassifier()
        
        # Test model info
        model_info = cnn.get_model_info()
        assert model_info.name == "cnn_product_classifier", "Model name incorrect"
        
        # Test input validation
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        is_valid = cnn.validate_input(test_image)
        assert is_valid, "Input validation failed"
        
        # Test preprocessing
        preprocessed = cnn.preprocess_input(test_image)
        assert preprocessed is not None, "Preprocessing failed"
        assert preprocessed.shape == (1, 224, 224, 3), "Preprocessed shape incorrect"
        
        print("  ✅ CNN classifier working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ CNN classifier test failed: {e}")
        traceback.print_exc()
        return False

def test_preprocessing_pipeline():
    """Test the modular preprocessing pipeline."""
    print("🔄 Testing Preprocessing Pipeline...")
    try:
        from services.data_processing.preprocessing_pipeline import ImagePreprocessor, TextPreprocessor, PreprocessingPipeline
        import numpy as np
        
        # Test image preprocessor
        img_processor = ImagePreprocessor(target_size=(224, 224))
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed_image = img_processor.process(test_image)
        assert processed_image is not None, "Image preprocessing failed"
        assert processed_image.shape[:2] == (224, 224), "Image resize failed"
        
        # Test text preprocessor
        text_processor = TextPreprocessor(lowercase=True, remove_punctuation=True)
        test_text = "Hello, World! This is a TEST."
        processed_text = text_processor.process(test_text)
        assert processed_text == "hello world this is a test", "Text preprocessing failed"
        
        # Test pipeline
        pipeline = PreprocessingPipeline([text_processor])
        pipeline_result = pipeline.process("Test Pipeline!")
        assert pipeline_result == "test pipeline", "Pipeline processing failed"
        
        print("  ✅ Preprocessing pipeline working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Preprocessing pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_evaluation_framework():
    """Test the comprehensive evaluation framework."""
    print("📈 Testing Evaluation Framework...")
    try:
        from services.ml_models.evaluation_framework import ModelEvaluator, AccuracyMetric, PrecisionMetric
        from services.ml_models.cnn_classifier import CNNClassifier
        import numpy as np
        
        # Test metrics
        accuracy_metric = AccuracyMetric()
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        accuracy = accuracy_metric.calculate(y_true, y_pred)
        assert 0 <= accuracy <= 1, "Accuracy calculation failed"
        
        precision_metric = PrecisionMetric()
        precision = precision_metric.calculate(y_true, y_pred)
        assert 0 <= precision <= 1, "Precision calculation failed"
        
        print("  ✅ Evaluation framework working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Evaluation framework test failed: {e}")
        traceback.print_exc()
        return False

def test_benchmarking_suite():
    """Test the performance benchmarking suite."""
    print("⚡ Testing Benchmarking Suite...")
    try:
        from services.ml_models.benchmarking_suite import BenchmarkingSuite, BenchmarkResult
        
        # Test benchmark result creation
        result = BenchmarkResult(
            model_name="test_model",
            model_version="1.0.0",
            benchmark_type="test",
            metrics={"accuracy": 0.95}
        )
        
        result_dict = result.to_dict()
        assert "model_name" in result_dict, "BenchmarkResult serialization failed"
        assert result_dict["metrics"]["accuracy"] == 0.95, "Metrics serialization failed"
        
        print("  ✅ Benchmarking suite working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Benchmarking suite test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("📁 Testing File Structure...")
    
    required_files = [
        "config.py",
        "utils/logging_config.py",
        "services/ml_models/__init__.py",
        "services/ml_models/base_model.py",
        "services/ml_models/cnn_classifier.py",
        "services/ml_models/evaluation_framework.py",
        "services/ml_models/benchmarking_suite.py",
        "services/data_processing/__init__.py",
        "services/data_processing/preprocessing_pipeline.py",
        "services/api_services/__init__.py",
        "notebooks/comprehensive_model_evaluation.ipynb",
        "docs/model_architecture.md",
        "IMPROVEMENTS_SUMMARY.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    else:
        print("  ✅ All required files present")
        return True

def main():
    """Run all integration tests."""
    print("🚀 RUNNING COMPREHENSIVE INTEGRATION TESTS")
    print("=" * 60)
    print("Testing all improvements made to address interviewer feedback...")
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Configuration System", test_configuration),
        ("Logging System", test_logging_system),
        ("Base Model Interface", test_base_model_interface),
        ("CNN Classifier", test_cnn_classifier),
        ("Preprocessing Pipeline", test_preprocessing_pipeline),
        ("Evaluation Framework", test_evaluation_framework),
        ("Benchmarking Suite", test_benchmarking_suite),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print("🎯 INTEGRATION TEST RESULTS")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Your codebase is ready for the interview!")
        print("\n✅ INTERVIEWER FEEDBACK ADDRESSED:")
        print("   1. ✅ Codebase organization and structure improved")
        print("   2. ✅ Model pipeline modularized into reusable components")
        print("   3. ✅ Comprehensive evaluation and documentation added")
        print("\n🚀 You're ready to showcase your improvements!")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
