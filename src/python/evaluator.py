import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class Evaluator:
    """Model evaluator"""
    
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.classes = dataset.classes
    
    def evaluate(self, test_loader, verbose=True) -> Dict:
        """Evaluate model on test set"""
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        start_time = time.time()
        
        # Set model to evaluation mode
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        from framework import Tensor
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            if HAS_CPP_BACKEND:
                import custom_dl_framework as cpp
                loss = cpp.crossEntropyLoss(outputs._tensor, labels._tensor)
                loss_value = loss.numpy()[0]
            else:
                # Manual loss computation
                probs = self._softmax(outputs.data)
                loss_value = -np.sum(labels.data * np.log(probs + 1e-8)) / outputs.shape[0]
            
            # Get predictions
            predictions = np.argmax(outputs.data, axis=1)
            true_labels = np.argmax(labels.data, axis=1)
            
            # Update metrics
            batch_size = images.shape[0]
            total_loss += loss_value * batch_size
            total_correct += (predictions == true_labels).sum()
            total_samples += batch_size
            
            # Store for detailed analysis
            all_predictions.extend(predictions)
            all_labels.extend(true_labels)
            all_probabilities.extend(outputs.data)
            
            if verbose and batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx}/{len(test_loader)}")
        
        evaluation_time = time.time() - start_time
        
        # Compute metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        # Compute additional metrics
        metrics = self._compute_metrics(np.array(all_predictions), 
                                       np.array(all_labels))
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"  Test Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.2%}")
        print(f"  Evaluation Time: {evaluation_time:.2f} seconds")
        print(f"  Total Samples: {total_samples}")
        
        # Print per-class metrics
        print(f"\nPer-class Metrics:")
        for i, class_name in enumerate(self.classes):
            class_mask = np.array(all_labels) == i
            if class_mask.any():
                class_acc = (np.array(all_predictions)[class_mask] == i).mean()
                print(f"  {class_name}: {class_acc:.2%}")
        
        # Create comprehensive report
        report = {
            'test_loss': float(avg_loss),
            'test_accuracy': float(accuracy),
            'evaluation_time': float(evaluation_time),
            'total_samples': int(total_samples),
            'metrics': metrics,
            'confusion_matrix': self._compute_confusion_matrix(all_predictions, all_labels),
            'classification_report': classification_report(all_labels, all_predictions, 
                                                         target_names=self.classes, 
                                                         output_dict=True)
        }
        
        # Save report
        self._save_report(report)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(all_predictions, all_labels)
        
        return report
    
    def _softmax(self, x):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _compute_metrics(self, predictions, labels):
        """Compute additional evaluation metrics"""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.classes):
            class_mask = labels == i
            if class_mask.any():
                class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                    labels[class_mask], predictions[class_mask], average='binary'
                )
                per_class_metrics[class_name] = {
                    'precision': float(class_precision),
                    'recall': float(class_recall),
                    'f1': float(class_f1),
                    'support': int(class_mask.sum())
                }
        
        return {
            'weighted_precision': float(precision),
            'weighted_recall': float(recall),
            'weighted_f1': float(f1),
            'per_class': per_class_metrics
        }
    
    def _compute_confusion_matrix(self, predictions, labels):
        """Compute confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        return cm.tolist()
    
    def _plot_confusion_matrix(self, predictions, labels):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save plot
        plot_file = Path("./reports/confusion_matrix.png")
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {plot_file}")
    
    def _save_report(self, report):
        """Save evaluation report"""
        report_file = Path("./reports/evaluation_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save as text
        text_file = Path("./reports/evaluation_summary.txt")
        with open(text_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Test Loss: {report['test_loss']:.4f}\n")
            f.write(f"Test Accuracy: {report['test_accuracy']:.2%}\n")
            f.write(f"Evaluation Time: {report['evaluation_time']:.2f} seconds\n")
            f.write(f"Total Samples: {report['total_samples']}\n\n")
            
            f.write("Weighted Metrics:\n")
            f.write(f"  Precision: {report['metrics']['weighted_precision']:.4f}\n")
            f.write(f"  Recall: {report['metrics']['weighted_recall']:.4f}\n")
            f.write(f"  F1-Score: {report['metrics']['weighted_f1']:.4f}\n\n")
            
            f.write("Per-class Metrics:\n")
            for class_name, metrics in report['metrics']['per_class'].items():
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall: {metrics['recall']:.4f}\n")
                f.write(f"    F1-Score: {metrics['f1']:.4f}\n")
                f.write(f"    Support: {metrics['support']}\n")
        
        print(f"Evaluation report saved to {report_file}")
        print(f"Evaluation summary saved to {text_file}")
    
    def analyze_predictions(self, test_loader, num_samples=10):
        """Analyze individual predictions"""
        print("\n" + "="*60)
        print("Prediction Analysis")
        print("="*60)
        
        self.model.eval()
        
        # Get a batch of data
        images, labels = next(iter(test_loader))
        
        # Make predictions
        outputs = self.model(images)
        predictions = np.argmax(outputs.data, axis=1)
        probabilities = self._softmax(outputs.data)
        true_labels = np.argmax(labels.data, axis=1)
        
        # Analyze first few samples
        print(f"\nAnalyzing first {min(num_samples, len(images))} samples:\n")
        
        for i in range(min(num_samples, len(images))):
            pred_class = self.classes[predictions[i]]
            true_class = self.classes[true_labels[i]]
            confidence = probabilities[i, predictions[i]]
            
            print(f"Sample {i+1}:")
            print(f"  True: {true_class}")
            print(f"  Predicted: {pred_class}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Correct: {'✓' if predictions[i] == true_labels[i] else '✗'}")
            print()