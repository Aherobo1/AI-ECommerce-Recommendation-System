"""
Performance Benchmarking Suite

Automated benchmarking system that tracks model performance over time
and compares different model versions with comprehensive metrics.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from .base_model import BaseModel, PredictionResult
from .evaluation_framework import ModelEvaluator
from config import get_config
from utils.logging_config import get_logger, monitor_performance


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, 
                 model_name: str,
                 model_version: str,
                 benchmark_type: str,
                 metrics: Dict[str, float],
                 metadata: Dict[str, Any] = None):
        self.model_name = model_name
        self.model_version = model_version
        self.benchmark_type = benchmark_type
        self.metrics = metrics
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'benchmark_type': self.benchmark_type,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class PerformanceBenchmark:
    """Performance benchmarking for model inference speed and resource usage."""
    
    def __init__(self):
        self.logger = get_logger('benchmarking.performance')
    
    @monitor_performance("performance_benchmark")
    def run_inference_benchmark(self, 
                               model: BaseModel, 
                               test_data: List[Any],
                               batch_sizes: List[int] = None,
                               num_runs: int = 5) -> BenchmarkResult:
        """
        Benchmark model inference performance.
        
        Args:
            model: Model to benchmark
            test_data: Test data for inference
            batch_sizes: List of batch sizes to test
            num_runs: Number of runs for averaging
            
        Returns:
            BenchmarkResult with performance metrics
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]
        
        self.logger.info(f"Running inference benchmark for {model.model_name}")
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(test_data):
                continue
                
            self.logger.info(f"Testing batch size: {batch_size}")
            
            # Prepare batch
            batch_data = test_data[:batch_size]
            
            # Warm-up runs
            for _ in range(2):
                try:
                    for sample in batch_data[:min(2, len(batch_data))]:
                        model.predict(sample)
                except Exception as e:
                    self.logger.warning(f"Warm-up failed: {e}")
            
            # Benchmark runs
            run_times = []
            for run in range(num_runs):
                start_time = time.time()
                
                successful_predictions = 0
                for sample in batch_data:
                    try:
                        model.predict(sample)
                        successful_predictions += 1
                    except Exception as e:
                        self.logger.warning(f"Prediction failed: {e}")
                
                end_time = time.time()
                
                if successful_predictions > 0:
                    run_times.append(end_time - start_time)
            
            if run_times:
                avg_time = np.mean(run_times)
                std_time = np.std(run_times)
                throughput = batch_size / avg_time
                
                results[f'batch_{batch_size}'] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'throughput': throughput,
                    'successful_predictions': successful_predictions
                }
        
        # Calculate overall metrics
        if results:
            all_throughputs = [r['throughput'] for r in results.values()]
            all_times = [r['avg_time'] for r in results.values()]
            
            overall_metrics = {
                'max_throughput': max(all_throughputs),
                'min_latency': min(all_times),
                'avg_throughput': np.mean(all_throughputs),
                'avg_latency': np.mean(all_times),
                'batch_results': results
            }
        else:
            overall_metrics = {'error': 'No successful benchmark runs'}
        
        return BenchmarkResult(
            model_name=model.model_name,
            model_version=model.model_version,
            benchmark_type='inference_performance',
            metrics=overall_metrics,
            metadata={
                'test_samples': len(test_data),
                'batch_sizes_tested': batch_sizes,
                'num_runs': num_runs
            }
        )
    
    def run_memory_benchmark(self, model: BaseModel) -> BenchmarkResult:
        """
        Benchmark model memory usage.
        
        Args:
            model: Model to benchmark
            
        Returns:
            BenchmarkResult with memory metrics
        """
        self.logger.info(f"Running memory benchmark for {model.model_name}")
        
        try:
            import psutil
            import gc
            
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Force garbage collection
            gc.collect()
            
            # Get model info
            model_info = model.get_model_info()
            
            # Estimate model size
            model_size_mb = 0
            if hasattr(model, 'model') and model.model is not None:
                try:
                    # Try to get TensorFlow model size
                    if hasattr(model.model, 'count_params'):
                        params = model.model.count_params()
                        model_size_mb = params * 4 / 1024 / 1024  # Assume float32
                except:
                    pass
            
            # Get current memory after model loading
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            metrics = {
                'initial_memory_mb': initial_memory,
                'current_memory_mb': current_memory,
                'model_memory_mb': current_memory - initial_memory,
                'estimated_model_size_mb': model_size_mb,
                'memory_efficiency': model_size_mb / (current_memory - initial_memory) if (current_memory - initial_memory) > 0 else 0
            }
            
        except ImportError:
            self.logger.warning("psutil not available, using basic memory estimation")
            metrics = {
                'estimated_model_size_mb': 10.0,  # Default estimate
                'memory_efficiency': 0.5
            }
        except Exception as e:
            self.logger.error(f"Memory benchmark failed: {e}")
            metrics = {'error': str(e)}
        
        return BenchmarkResult(
            model_name=model.model_name,
            model_version=model.model_version,
            benchmark_type='memory_usage',
            metrics=metrics
        )


class AccuracyBenchmark:
    """Accuracy benchmarking for model performance tracking."""
    
    def __init__(self):
        self.logger = get_logger('benchmarking.accuracy')
    
    def run_accuracy_benchmark(self, 
                             model: BaseModel,
                             test_data: List[Any],
                             test_labels: List[Any]) -> BenchmarkResult:
        """
        Benchmark model accuracy performance.
        
        Args:
            model: Model to benchmark
            test_data: Test data
            test_labels: True labels
            
        Returns:
            BenchmarkResult with accuracy metrics
        """
        self.logger.info(f"Running accuracy benchmark for {model.model_name}")
        
        # Use evaluation framework
        evaluator = ModelEvaluator(model, task_type='classification')
        evaluation_results = evaluator.evaluate(test_data, test_labels)
        
        # Extract key metrics
        metrics = evaluation_results['metrics']
        analysis = evaluation_results.get('analysis', {})
        
        # Add confidence analysis
        if 'confidence_stats' in analysis:
            conf_stats = analysis['confidence_stats']
            metrics.update({
                'avg_confidence': conf_stats['mean'],
                'confidence_std': conf_stats['std'],
                'min_confidence': conf_stats['min'],
                'max_confidence': conf_stats['max']
            })
        
        # Add error analysis
        if 'error_analysis' in analysis:
            error_analysis = analysis['error_analysis']
            metrics.update({
                'error_rate': error_analysis['error_rate'],
                'num_errors': error_analysis['num_errors']
            })
        
        return BenchmarkResult(
            model_name=model.model_name,
            model_version=model.model_version,
            benchmark_type='accuracy_performance',
            metrics=metrics,
            metadata={
                'test_samples': len(test_data),
                'evaluation_results': evaluation_results
            }
        )


class BenchmarkingSuite:
    """Comprehensive benchmarking suite for model performance tracking."""
    
    def __init__(self, results_dir: str = None):
        """
        Initialize benchmarking suite.
        
        Args:
            results_dir: Directory to store benchmark results
        """
        self.config = get_config()
        self.logger = get_logger('benchmarking.suite')
        
        self.results_dir = Path(results_dir) if results_dir else Path('data/benchmark_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_benchmark = PerformanceBenchmark()
        self.accuracy_benchmark = AccuracyBenchmark()
        
        self.results_history = []
        self._load_history()
    
    def run_comprehensive_benchmark(self, 
                                  model: BaseModel,
                                  test_data: List[Any],
                                  test_labels: List[Any] = None) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive benchmark including performance and accuracy tests.
        
        Args:
            model: Model to benchmark
            test_data: Test data
            test_labels: True labels (optional, for accuracy benchmark)
            
        Returns:
            Dictionary of benchmark results
        """
        self.logger.info(f"Running comprehensive benchmark for {model.model_name}")
        
        results = {}
        
        # Performance benchmark
        try:
            perf_result = self.performance_benchmark.run_inference_benchmark(
                model, test_data[:50]  # Use subset for performance testing
            )
            results['performance'] = perf_result
            self.logger.info("Performance benchmark completed")
        except Exception as e:
            self.logger.error(f"Performance benchmark failed: {e}")
        
        # Memory benchmark
        try:
            memory_result = self.performance_benchmark.run_memory_benchmark(model)
            results['memory'] = memory_result
            self.logger.info("Memory benchmark completed")
        except Exception as e:
            self.logger.error(f"Memory benchmark failed: {e}")
        
        # Accuracy benchmark (if labels provided)
        if test_labels:
            try:
                acc_result = self.accuracy_benchmark.run_accuracy_benchmark(
                    model, test_data, test_labels
                )
                results['accuracy'] = acc_result
                self.logger.info("Accuracy benchmark completed")
            except Exception as e:
                self.logger.error(f"Accuracy benchmark failed: {e}")
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, BenchmarkResult]):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual results
        for benchmark_type, result in results.items():
            filename = f"{result.model_name}_{result.model_version}_{benchmark_type}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        
        # Add to history
        self.results_history.extend(results.values())
        
        # Save consolidated history
        history_file = self.results_dir / 'benchmark_history.json'
        with open(history_file, 'w') as f:
            json.dump([r.to_dict() for r in self.results_history], f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {self.results_dir}")
    
    def _load_history(self):
        """Load benchmark history from file."""
        history_file = self.results_dir / 'benchmark_history.json'
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                for item in history_data:
                    result = BenchmarkResult(
                        model_name=item['model_name'],
                        model_version=item['model_version'],
                        benchmark_type=item['benchmark_type'],
                        metrics=item['metrics'],
                        metadata=item.get('metadata', {})
                    )
                    result.timestamp = datetime.fromisoformat(item['timestamp'])
                    self.results_history.append(result)
                
                self.logger.info(f"Loaded {len(self.results_history)} historical benchmark results")
                
            except Exception as e:
                self.logger.warning(f"Failed to load benchmark history: {e}")
    
    def generate_comparison_report(self, model_names: List[str] = None) -> str:
        """
        Generate comparison report across models and versions.
        
        Args:
            model_names: List of model names to compare (None for all)
            
        Returns:
            Formatted comparison report
        """
        if not self.results_history:
            return "No benchmark results available for comparison."
        
        # Filter results
        filtered_results = self.results_history
        if model_names:
            filtered_results = [r for r in filtered_results if r.model_name in model_names]
        
        if not filtered_results:
            return "No matching benchmark results found."
        
        # Group by model and benchmark type
        grouped_results = {}
        for result in filtered_results:
            key = (result.model_name, result.benchmark_type)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Generate report
        report_lines = []
        report_lines.append("BENCHMARK COMPARISON REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for (model_name, benchmark_type), results in grouped_results.items():
            report_lines.append(f"Model: {model_name} | Benchmark: {benchmark_type}")
            report_lines.append("-" * 40)
            
            # Sort by timestamp (most recent first)
            results.sort(key=lambda x: x.timestamp, reverse=True)
            
            for result in results[:5]:  # Show last 5 results
                report_lines.append(f"  Version: {result.model_version}")
                report_lines.append(f"  Date: {result.timestamp.strftime('%Y-%m-%d %H:%M')}")
                
                # Show key metrics
                for metric_name, value in result.metrics.items():
                    if isinstance(value, (int, float)) and not metric_name.startswith('batch_'):
                        report_lines.append(f"    {metric_name}: {value:.4f}")
                
                report_lines.append("")
        
        return "\n".join(report_lines)
