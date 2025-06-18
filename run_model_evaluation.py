#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

This script runs the complete model evaluation pipeline and generates
all necessary artifacts for interviewer review.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append('.')

def setup_directories():
    """Create necessary directories for evaluation results."""
    dirs = [
        'static/images/evaluation',
        'data/evaluation_results',
        'logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def create_synthetic_test_data(class_names, samples_per_class=20):
    """Create synthetic test data for evaluation."""
    print(f"🔄 Creating synthetic test data...")
    
    def create_synthetic_image(class_name, image_size=(224, 224, 3)):
        """Create a synthetic image with class-specific characteristics."""
        np.random.seed(hash(class_name) % 1000)  # Consistent seed per class
        image = np.random.randint(0, 256, image_size, dtype=np.uint8)
        
        # Add class-specific patterns
        if 'computer' in class_name or 'electronics' in class_name:
            image[50:150, 50:150] = [100, 100, 150]  # Blue rectangle
        elif 'clothing' in class_name or 't-shirt' in class_name:
            image[::2, ::2] = [200, 150, 100]  # Fabric pattern
        elif 'kitchen' in class_name or 'teapot' in class_name:
            center = (image_size[0]//2, image_size[1]//2)
            y, x = np.ogrid[:image_size[0], :image_size[1]]
            mask = (x - center[0])**2 + (y - center[1])**2 <= 50**2
            image[mask] = [150, 100, 50]  # Brown circle
        elif 'antique' in class_name:
            image = (image * 0.8 + 50).astype(np.uint8)  # Vintage look
        
        return image
    
    test_images = []
    test_labels = []
    
    for class_name in class_names:
        for i in range(samples_per_class):
            image = create_synthetic_image(class_name)
            test_images.append(image)
            test_labels.append(class_name)
    
    print(f"✅ Created {len(test_images)} test samples")
    print(f"📊 Classes: {len(set(test_labels))}")
    print(f"📊 Samples per class: {samples_per_class}")
    
    return test_images, test_labels

def run_basic_evaluation(model, test_images, test_labels):
    """Run basic model evaluation."""
    print("🔄 Running basic model evaluation...")
    
    predictions = []
    confidences = []
    processing_times = []
    
    for i, (image, true_label) in enumerate(zip(test_images, test_labels)):
        try:
            result = model.predict(image)
            predictions.append(result.prediction)
            confidences.append(result.confidence)
            processing_times.append(result.processing_time or 0.0)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(test_images)} samples")
                
        except Exception as e:
            print(f"  ⚠️ Error processing sample {i}: {e}")
            predictions.append("unknown")
            confidences.append(0.0)
            processing_times.append(0.0)
    
    # Calculate metrics
    correct_predictions = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
    accuracy = correct_predictions / len(test_labels)
    avg_confidence = np.mean(confidences)
    avg_processing_time = np.mean(processing_times)
    
    metrics = {
        'accuracy': accuracy,
        'total_samples': len(test_labels),
        'correct_predictions': correct_predictions,
        'average_confidence': avg_confidence,
        'average_processing_time': avg_processing_time,
        'predictions': predictions,
        'confidences': confidences,
        'processing_times': processing_times
    }
    
    print(f"✅ Evaluation completed!")
    print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 Average Confidence: {avg_confidence:.4f}")
    print(f"📊 Average Processing Time: {avg_processing_time:.4f}s")
    
    return metrics

def generate_visualizations(metrics, test_labels, class_names):
    """Generate evaluation visualizations."""
    print("🎨 Generating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    predictions = metrics['predictions']
    cm = confusion_matrix(test_labels, predictions, labels=class_names)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Model Performance', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('static/images/evaluation/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved")
    
    # Performance metrics visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # Accuracy bar chart
    axes[0, 0].bar(['Overall Accuracy'], [metrics['accuracy']], color='skyblue', alpha=0.8)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_ylabel('Accuracy Score')
    for i, v in enumerate([metrics['accuracy']]):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Confidence distribution
    axes[0, 1].hist(metrics['confidences'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Confidence Score Distribution')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(metrics['average_confidence'], color='red', linestyle='--', 
                       label=f'Mean: {metrics["average_confidence"]:.3f}')
    axes[0, 1].legend()
    
    # Processing time distribution
    axes[1, 0].hist(metrics['processing_times'], bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Processing Time Distribution')
    axes[1, 0].set_xlabel('Processing Time (seconds)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(metrics['average_processing_time'], color='red', linestyle='--',
                       label=f'Mean: {metrics["average_processing_time"]:.4f}s')
    axes[1, 0].legend()
    
    # Class-wise accuracy
    class_accuracy = {}
    for class_name in class_names:
        class_indices = [i for i, label in enumerate(test_labels) if label == class_name]
        class_predictions = [predictions[i] for i in class_indices]
        class_correct = sum(1 for pred in class_predictions if pred == class_name)
        class_accuracy[class_name] = class_correct / len(class_indices) if class_indices else 0
    
    class_names_short = [name[:10] + '...' if len(name) > 10 else name for name in class_names]
    axes[1, 1].bar(class_names_short, list(class_accuracy.values()), color='gold', alpha=0.8)
    axes[1, 1].set_title('Per-Class Accuracy')
    axes[1, 1].set_xlabel('Classes')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('static/images/evaluation/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Performance metrics visualization saved")
    
    return class_accuracy

def save_evaluation_report(metrics, class_accuracy, class_names):
    """Save comprehensive evaluation report."""
    print("📝 Generating evaluation report...")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""
# Model Evaluation Report
Generated: {timestamp}

## Executive Summary
- **Overall Accuracy**: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
- **Total Samples Evaluated**: {metrics['total_samples']}
- **Correct Predictions**: {metrics['correct_predictions']}
- **Average Confidence**: {metrics['average_confidence']:.4f}
- **Average Processing Time**: {metrics['average_processing_time']:.4f} seconds

## Detailed Performance Metrics

### Overall Performance
- Accuracy: {metrics['accuracy']:.4f}
- Error Rate: {1-metrics['accuracy']:.4f}
- Confidence Statistics:
  - Mean: {metrics['average_confidence']:.4f}
  - Min: {min(metrics['confidences']):.4f}
  - Max: {max(metrics['confidences']):.4f}
  - Std: {np.std(metrics['confidences']):.4f}

### Processing Performance
- Average Processing Time: {metrics['average_processing_time']:.4f}s
- Total Processing Time: {sum(metrics['processing_times']):.4f}s
- Throughput: {len(metrics['processing_times'])/sum(metrics['processing_times']):.2f} images/second

### Per-Class Performance
"""
    
    for class_name in class_names:
        accuracy = class_accuracy.get(class_name, 0)
        report += f"- {class_name}: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
    
    report += f"""

## Model Architecture
- Model Type: CNN Classifier
- Input Shape: (224, 224, 3)
- Number of Classes: {len(class_names)}
- Classes: {', '.join(class_names)}

## Evaluation Methodology
- Test Data: Synthetic images with class-specific patterns
- Samples per Class: 20
- Total Test Samples: {metrics['total_samples']}
- Evaluation Date: {timestamp}

## Conclusions
The model demonstrates {'good' if metrics['accuracy'] > 0.7 else 'moderate' if metrics['accuracy'] > 0.5 else 'limited'} performance with an overall accuracy of {metrics['accuracy']*100:.1f}%.
Processing time is efficient at {metrics['average_processing_time']:.4f} seconds per image.

## Recommendations
1. {'Consider fine-tuning with real-world data' if metrics['accuracy'] < 0.9 else 'Model performance is excellent'}
2. Monitor confidence scores for prediction reliability
3. Optimize processing pipeline for production deployment
"""
    
    # Save report
    report_file = f"data/evaluation_results/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Save metrics as JSON
    metrics_file = f"data/evaluation_results/evaluation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    metrics_data = {
        'timestamp': timestamp,
        'overall_metrics': {
            'accuracy': metrics['accuracy'],
            'total_samples': metrics['total_samples'],
            'correct_predictions': metrics['correct_predictions'],
            'average_confidence': metrics['average_confidence'],
            'average_processing_time': metrics['average_processing_time']
        },
        'class_accuracy': class_accuracy,
        'class_names': class_names
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"✅ Report saved: {report_file}")
    print(f"✅ Metrics saved: {metrics_file}")
    
    return report_file, metrics_file

def main():
    """Main evaluation function."""
    print("🚀 Starting Comprehensive Model Evaluation")
    print("=" * 50)
    
    try:
        # Setup
        setup_directories()
        
        # Import model components
        from services.ml_models.cnn_classifier import CNNClassifier
        print("✅ Model components imported")
        
        # Initialize model
        cnn_model = CNNClassifier()
        print("✅ CNN Model initialized")
        
        # Load or create model
        model_path = 'models/cnn_product_classifier.h5'
        if os.path.exists(model_path):
            success = cnn_model.load_model(model_path)
            print(f"✅ Model loaded: {success}")
        else:
            print("⚠️ Model file not found, using demo model")
        
        # Get class names
        class_names = cnn_model.class_names
        print(f"✅ Model classes: {class_names}")
        
        # Create test data
        test_images, test_labels = create_synthetic_test_data(class_names)
        
        # Run evaluation
        metrics = run_basic_evaluation(cnn_model, test_images, test_labels)
        
        # Generate visualizations
        class_accuracy = generate_visualizations(metrics, test_labels, class_names)
        
        # Save report
        report_file, metrics_file = save_evaluation_report(metrics, class_accuracy, class_names)
        
        print("\n🎉 EVALUATION COMPLETE!")
        print("=" * 30)
        print(f"📊 Overall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"📁 Results saved in: data/evaluation_results/")
        print(f"🖼️ Visualizations saved in: static/images/evaluation/")
        print(f"📝 Report: {report_file}")
        print(f"📊 Metrics: {metrics_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
