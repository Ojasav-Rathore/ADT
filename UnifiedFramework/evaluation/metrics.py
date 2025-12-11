"""
Evaluation metrics for backdoor defense
Includes: ACC (Clean Accuracy), ASR (Attack Success Rate), TPR vs FPR (ROC)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


class BackdoorMetrics:
    """Metrics for evaluating backdoor defense"""
    
    def __init__(self):
        self.clean_acc_list = []
        self.asr_list = []
        self.all_predictions = []
        self.all_labels = []
        self.all_confidences = []
    
    def calculate_acc(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate clean accuracy (ACC)
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
        
        Returns:
            Accuracy percentage
        """
        correct = np.sum(predictions == labels)
        total = len(labels)
        acc = (correct / total) * 100
        self.clean_acc_list.append(acc)
        return acc
    
    def calculate_asr(self, predictions: np.ndarray, target_label: int) -> float:
        """
        Calculate attack success rate (ASR)
        
        Args:
            predictions: Model predictions on poisoned samples
            target_label: Target label of backdoor attack
        
        Returns:
            ASR percentage
        """
        correct = np.sum(predictions == target_label)
        total = len(predictions)
        asr = (correct / total) * 100
        self.asr_list.append(asr)
        return asr
    
    def calculate_metrics_batch(self, model: nn.Module, dataloader,
                               target_label: int, is_poisoned: bool = False) -> Dict[str, float]:
        """
        Calculate all metrics on a batch
        
        Args:
            model: Neural network model
            dataloader: Data loader
            target_label: Target label for ASR
            is_poisoned: Whether data is poisoned
        
        Returns:
            Dictionary of metrics
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_confidence = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.cuda()
                outputs = model(images)
                confidence, predictions = torch.max(outputs, 1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_confidence.extend(confidence.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_confidence = np.array(all_confidence)
        
        metrics = {}
        
        if is_poisoned:
            metrics['ASR'] = self.calculate_asr(all_preds, target_label)
        else:
            metrics['ACC'] = self.calculate_acc(all_preds, all_labels)
        
        # Store for ROC calculation
        self.all_predictions.append(all_preds)
        self.all_labels.append(all_labels)
        self.all_confidences.append(all_confidence)
        
        return metrics
    
    def calculate_roc_metrics(self, clean_confidences: np.ndarray, 
                             poisoned_confidences: np.ndarray) -> Dict[str, any]:
        """
        Calculate ROC metrics (TPR vs FPR)
        
        Args:
            clean_confidences: Confidence scores for clean samples
            poisoned_confidences: Confidence scores for poisoned samples
        
        Returns:
            Dictionary containing FPR, TPR, and AUC
        """
        # Create binary labels: 0 for clean, 1 for poisoned
        true_labels = np.concatenate([
            np.zeros(len(clean_confidences)),
            np.ones(len(poisoned_confidences))
        ])
        
        confidence_scores = np.concatenate([
            clean_confidences,
            poisoned_confidences
        ])
        
        fpr, tpr, thresholds = roc_curve(true_labels, confidence_scores)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        summary = {}
        
        if self.clean_acc_list:
            summary['mean_ACC'] = np.mean(self.clean_acc_list)
            summary['std_ACC'] = np.std(self.clean_acc_list)
        
        if self.asr_list:
            summary['mean_ASR'] = np.mean(self.asr_list)
            summary['std_ASR'] = np.std(self.asr_list)
        
        return summary


class AdversarialDetection:
    """Adversarial detection metrics for defense"""
    
    @staticmethod
    def detect_backdoor_via_activation(model: nn.Module, clean_loader, 
                                       poisoned_loader, threshold: float = None) -> Dict:
        """
        Detect backdoor using activation patterns
        
        Args:
            model: Model to analyze
            clean_loader: Clean data loader
            poisoned_loader: Poisoned data loader
            threshold: Detection threshold
        
        Returns:
            Detection metrics
        """
        # Extract activations
        clean_acts = AdversarialDetection._extract_activations(model, clean_loader)
        poison_acts = AdversarialDetection._extract_activations(model, poisoned_loader)
        
        # Calculate statistical differences
        clean_mean = np.mean(clean_acts, axis=0)
        poison_mean = np.mean(poison_acts, axis=0)
        
        # Use Euclidean distance as metric
        activation_diff = np.linalg.norm(clean_mean - poison_mean)
        
        # Calculate detection rate
        clean_dists = np.linalg.norm(clean_acts - poison_mean, axis=1)
        poison_dists = np.linalg.norm(poison_acts - poison_mean, axis=1)
        
        if threshold is None:
            threshold = (np.mean(clean_dists) + np.mean(poison_dists)) / 2
        
        clean_detected = np.sum(clean_dists > threshold) / len(clean_dists)
        poison_detected = np.sum(poison_dists > threshold) / len(poison_dists)
        
        return {
            'activation_distance': activation_diff,
            'threshold': threshold,
            'false_positive_rate': clean_detected,
            'true_positive_rate': poison_detected,
            'detection_rate': poison_detected
        }
    
    @staticmethod
    def _extract_activations(model: nn.Module, dataloader) -> np.ndarray:
        """Extract activations from penultimate layer"""
        model.eval()
        activations = []
        
        # Register hook to capture activations
        def hook(module, input, output):
            activations.append(output.detach().cpu().numpy())
        
        # Hook into the layer before classification
        if hasattr(model, 'fc'):
            model.fc.register_forward_hook(hook)
        elif hasattr(model, 'classifier'):
            model.classifier.register_forward_hook(hook)
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.cuda()
                _ = model(images)
        
        return np.concatenate(activations, axis=0)


def plot_roc_curve(roc_metrics: Dict, save_path: str = None):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(roc_metrics['fpr'], roc_metrics['tpr'], 
             label=f"ROC (AUC = {roc_metrics['auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Backdoor Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_comparison_metrics(metrics_dict: Dict[str, Dict], save_path: str = None):
    """Plot comparison of metrics across defense methods"""
    methods = list(metrics_dict.keys())
    acc_values = [metrics_dict[m].get('ACC', 0) for m in methods]
    asr_values = [metrics_dict[m].get('ASR', 100) for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ACC comparison
    ax1.bar(methods, acc_values, color='steelblue')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Clean Accuracy (ACC) Comparison')
    ax1.set_ylim([0, 100])
    
    # ASR comparison
    ax2.bar(methods, asr_values, color='coral')
    ax2.set_ylabel('Attack Success Rate (%)')
    ax2.set_title('Attack Success Rate (ASR) Comparison')
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
