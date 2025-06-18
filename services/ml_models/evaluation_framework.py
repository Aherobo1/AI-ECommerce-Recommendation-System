"""
Model Evaluation Framework

Comprehensive evaluation system that can be used across different models
with standardized metrics, visualizations, and reporting.
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from abc import ABC, abstractmethod
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from .base_model import BaseModel, PredictionResult
from config import get_config
from utils.logging_config import get_logger, monitor_performance

# Try to import sklearn
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        mean_squared_error, mean_absolute_error, r2_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Calculate the metric value."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass


class AccuracyMetric(EvaluationMetric):
    """Accuracy metric for classification tasks."""
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Calculate accuracy score."""
        if SKLEARN_AVAILABLE:
            return accuracy_score(y_true, y_pred)
        else:
            return float(np.mean(y_true == y_pred))
    
    def get_name(self) -> str:
        return "accuracy"


class PrecisionMetric(EvaluationMetric):
    """Precision metric for classification tasks."""
    
    def __init__(self, average: str = 'weighted'):
        self.average = average
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Calculate precision score."""
        if SKLEARN_AVAILABLE:
            return precision_score(y_true, y_pred, average=self.average, zero_division=0)
        else:
            # Simple binary precision calculation
            if len(np.unique(y_true)) == 2:
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                return tp / (tp + fp) if (tp + fp) > 0 else 0.0
            return 0.0
    
    def get_name(self) -> str:
        return f"precision_{self.average}"


class RecallMetric(EvaluationMetric):
    """Recall metric for classification tasks."""
    
    def __init__(self, average: str = 'weighted'):
        self.average = average
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Calculate recall score."""
        if SKLEARN_AVAILABLE:
            return recall_score(y_true, y_pred, average=self.average, zero_division=0)
        else:
            # Simple binary recall calculation
            if len(np.unique(y_true)) == 2:
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                return tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 0.0
    
    def get_name(self) -> str:
        return f"recall_{self.average}"


class F1Metric(EvaluationMetric):
    """F1 score metric for classification tasks."""
    
    def __init__(self, average: str = 'weighted'):
        self.average = average
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Calculate F1 score."""
        if SKLEARN_AVAILABLE:
            return f1_score(y_true, y_pred, average=self.average, zero_division=0)
        else:
            # Calculate F1 from precision and recall
            precision_metric = PrecisionMetric(self.average)
            recall_metric = RecallMetric(self.average)
            
            precision = precision_metric.calculate(y_true, y_pred)
            recall = recall_metric.calculate(y_true, y_pred)
            
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)
    
    def get_name(self) -> str:
        return f"f1_{self.average}"


class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, model: BaseModel, task_type: str = 'classification'):
        """
        Initialize model evaluator.
        
        Args:
            model: Model to evaluate
            task_type: Type of task ('classification' or 'regression')
        """
        self.model = model
        self.task_type = task_type
        self.config = get_config()
        self.logger = get_logger('evaluation')
        
        # Initialize metrics based on task type
        self.metrics = self._get_default_metrics()
        
        # Results storage
        self.evaluation_results = {}
        self.predictions_cache = {}
    
    def _get_default_metrics(self) -> List[EvaluationMetric]:
        """Get default metrics for the task type."""
        if self.task_type == 'classification':
            return [
                AccuracyMetric(),
                PrecisionMetric('weighted'),
                RecallMetric('weighted'),
                F1Metric('weighted'),
                PrecisionMetric('macro'),
                RecallMetric('macro'),
                F1Metric('macro')
            ]
        else:  # regression
            return []  # Would add regression metrics here
    
    def add_metric(self, metric: EvaluationMetric):
        """Add a custom metric to the evaluation."""
        self.metrics.append(metric)
    
    @monitor_performance("model_evaluation")
    def evaluate(self, 
                 test_data: List[Any], 
                 test_labels: List[Any],
                 batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: List of test inputs
            test_labels: List of true labels
            batch_size: Batch size for predictions
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            self.logger.info(f"Starting evaluation on {len(test_data)} samples")
            
            # Get predictions
            predictions = self._get_predictions(test_data, batch_size)
            
            # Convert labels to numpy arrays
            y_true = np.array(test_labels)
            y_pred = np.array([pred.prediction for pred in predictions])
            
            # Calculate metrics
            metrics_results = {}
            for metric in self.metrics:
                try:
                    score = metric.calculate(y_true, y_pred)
                    metrics_results[metric.get_name()] = score
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {metric.get_name()}: {e}")
                    metrics_results[metric.get_name()] = 0.0
            
            # Generate additional analysis
            analysis = self._generate_analysis(y_true, y_pred, predictions)
            
            # Combine results
            results = {
                'metrics': metrics_results,
                'analysis': analysis,
                'metadata': {
                    'model_name': self.model.model_name,
                    'model_version': self.model.model_version,
                    'task_type': self.task_type,
                    'num_samples': len(test_data),
                    'evaluation_date': datetime.now().isoformat()
                }
            }
            
            self.evaluation_results = results
            self.logger.info("Evaluation completed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
    
    def _get_predictions(self, test_data: List[Any], batch_size: int) -> List[PredictionResult]:
        """Get model predictions for test data."""
        predictions = []
        
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            
            for sample in batch:
                try:
                    pred = self.model.predict(sample)
                    predictions.append(pred)
                except Exception as e:
                    self.logger.warning(f"Prediction failed for sample {i}: {e}")
                    # Create dummy prediction
                    predictions.append(PredictionResult(
                        prediction="unknown",
                        confidence=0.0,
                        metadata={'error': str(e)}
                    ))
        
        return predictions
    
    def _generate_analysis(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Generate additional analysis of the results."""
        analysis = {}
        
        try:
            # Confidence distribution
            confidences = [pred.confidence for pred in predictions]
            analysis['confidence_stats'] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences))
            }
            
            # Class distribution (for classification)
            if self.task_type == 'classification':
                unique_true, counts_true = np.unique(y_true, return_counts=True)
                unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
                
                analysis['class_distribution'] = {
                    'true': dict(zip(unique_true.tolist(), counts_true.tolist())),
                    'predicted': dict(zip(unique_pred.tolist(), counts_pred.tolist()))
                }
                
                # Confusion matrix
                if SKLEARN_AVAILABLE:
                    cm = confusion_matrix(y_true, y_pred)
                    analysis['confusion_matrix'] = cm.tolist()
            
            # Error analysis
            errors = []
            for i, (true_label, pred) in enumerate(zip(y_true, predictions)):
                if true_label != pred.prediction:
                    errors.append({
                        'sample_index': i,
                        'true_label': true_label,
                        'predicted_label': pred.prediction,
                        'confidence': pred.confidence
                    })
            
            analysis['error_analysis'] = {
                'num_errors': len(errors),
                'error_rate': len(errors) / len(y_true),
                'sample_errors': errors[:10]  # First 10 errors
            }
            
        except Exception as e:
            self.logger.warning(f"Analysis generation failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        report_lines = []
        results = self.evaluation_results
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Model information
        metadata = results['metadata']
        report_lines.append("Model Information:")
        report_lines.append(f"  Name: {metadata['model_name']}")
        report_lines.append(f"  Version: {metadata['model_version']}")
        report_lines.append(f"  Task Type: {metadata['task_type']}")
        report_lines.append(f"  Evaluation Date: {metadata['evaluation_date']}")
        report_lines.append(f"  Test Samples: {metadata['num_samples']}")
        report_lines.append("")
        
        # Metrics
        report_lines.append("Performance Metrics:")
        metrics = results['metrics']
        for metric_name, score in metrics.items():
            report_lines.append(f"  {metric_name}: {score:.4f}")
        report_lines.append("")
        
        # Analysis
        if 'analysis' in results:
            analysis = results['analysis']
            
            # Confidence statistics
            if 'confidence_stats' in analysis:
                conf_stats = analysis['confidence_stats']
                report_lines.append("Confidence Statistics:")
                report_lines.append(f"  Mean: {conf_stats['mean']:.4f}")
                report_lines.append(f"  Std: {conf_stats['std']:.4f}")
                report_lines.append(f"  Range: [{conf_stats['min']:.4f}, {conf_stats['max']:.4f}]")
                report_lines.append("")
            
            # Error analysis
            if 'error_analysis' in analysis:
                error_analysis = analysis['error_analysis']
                report_lines.append("Error Analysis:")
                report_lines.append(f"  Total Errors: {error_analysis['num_errors']}")
                report_lines.append(f"  Error Rate: {error_analysis['error_rate']:.4f}")
                report_lines.append("")
        
        # Summary
        report_lines.append("Summary:")
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            if acc > 0.9:
                report_lines.append("  ✅ Excellent performance (>90% accuracy)")
            elif acc > 0.8:
                report_lines.append("  ✅ Good performance (>80% accuracy)")
            elif acc > 0.7:
                report_lines.append("  ⚠️  Fair performance (>70% accuracy)")
            else:
                report_lines.append("  ❌ Poor performance (<70% accuracy)")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text
    
    def save_results(self, output_path: str):
        """Save evaluation results to JSON file."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results to save")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation results saved to {output_path}")
