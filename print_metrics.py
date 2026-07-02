import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    precision_score,
    recall_score
)
from collections import defaultdict

def compute_metrics_from_batched_outputs(data, positive_label='spoof'):
    """
    Compute metrics from batched model outputs
    
    Args:
        data: List of batches, each containing 'id', 'ground_truth', 'text' (predictions)
        positive_label: The label to consider as positive for binary metrics (default: 'spoof')
    
    Returns:
        Dictionary with all metrics
    """
    # Collect all predictions and ground truths
    all_ids = []
    all_ground_truth = []
    all_predictions = []
    all_losses = []
    
    for batch in data:
        # Extract data from each batch
        ids = batch.get('id', [])
        ground_truth = batch.get('ground_truth', [])
        predictions = batch.get('text', [])
        losses = batch.get('loss', [])
        
        all_ids.extend(ids)
        all_ground_truth.extend(ground_truth)
        all_predictions.extend(predictions)
        
        if losses:
            if isinstance(losses, list):
                all_losses.extend(losses)
            else:
                all_losses.append(losses)
    
    # Convert to numpy arrays
    y_true = np.array(all_ground_truth)
    y_pred = np.array(all_predictions)
    
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Compute metrics
    metrics = {}
    
    # Basic metrics
    metrics['total_samples'] = len(y_true)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', labels=classes)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', labels=classes)
    metrics['classes'] = classes.tolist()
    
    # F1 per class
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=classes)
    for i, cls in enumerate(classes):
        metrics[f'f1_{cls}'] = f1_per_class[i]
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics['confusion_matrix'] = cm.tolist()
    
    # For binary classification
    if len(classes) == 2:
        # Determine which class is positive
        if positive_label in classes:
            pos_label = positive_label
        else:
            # If positive_label not in classes, use the second class as positive
            pos_label = classes[1]
        
        # Get indices for positive and negative classes
        pos_idx = list(classes).index(pos_label)
        neg_idx = 1 - pos_idx
        
        # Extract TP, FP, TN, FN
        tp = cm[pos_idx][pos_idx]
        fp = cm[neg_idx][pos_idx]
        tn = cm[neg_idx][neg_idx]
        fn = cm[pos_idx][neg_idx]
        
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
        metrics['positive_label'] = pos_label
        metrics['negative_label'] = classes[neg_idx]
        
        # Additional binary metrics with explicit pos_label
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics['f1_binary'] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true == pos_label, y_pred == pos_label)
        
        # Calculate specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, 
        target_names=classes, 
        output_dict=True,
        zero_division=0
    )
    
    # Loss statistics if available
    if all_losses:
        metrics['avg_loss'] = float(np.mean(all_losses))
        metrics['std_loss'] = float(np.std(all_losses))
        metrics['min_loss'] = float(np.min(all_losses))
        metrics['max_loss'] = float(np.max(all_losses))
    
    # Error analysis
    errors = y_true != y_pred
    metrics['error_count'] = int(np.sum(errors))
    metrics['error_rate'] = float(np.mean(errors))
    
    # Detailed error analysis
    error_indices = np.where(errors)[0]
    metrics['error_indices'] = error_indices.tolist()
    
    # Show some examples of errors
    error_samples = []
    for idx in error_indices[:10]:  # Show first 10 errors
        error_samples.append({
            'id': all_ids[idx] if idx < len(all_ids) else f'sample_{idx}',
            'true': y_true[idx],
            'predicted': y_pred[idx]
        })
    metrics['error_samples'] = error_samples
    
    return metrics, y_true, y_pred, all_ids

def print_metrics(metrics, verbose=True):
    """Print metrics in a readable format"""
    print("="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Samples: {metrics['total_samples']}")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"   Error Rate: {metrics['error_rate']:.4f}")
    print(f"   Error Count: {metrics['error_count']}")
    
    print(f"\n🎯 F1 Scores:")
    print(f"   Macro F1: {metrics['f1_macro']:.4f}")
    print(f"   Weighted F1: {metrics['f1_weighted']:.4f}")
    
    for cls in metrics['classes']:
        print(f"   F1 ({cls}): {metrics[f'f1_{cls}']:.4f}")
    
    if 'precision' in metrics:
        print(f"\n🔍 Binary Classification Metrics (Positive: {metrics['positive_label']}):")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"   Specificity: {metrics['specificity']:.4f}")
        print(f"   F1 Score: {metrics['f1_binary']:.4f}")
        print(f"   MCC: {metrics['mcc']:.4f}")
        
        print(f"\n   Confusion Matrix:")
        print(f"   TP: {metrics['true_positives']}")
        print(f"   FP: {metrics['false_positives']}")
        print(f"   TN: {metrics['true_negatives']}")
        print(f"   FN: {metrics['false_negatives']}")
    
    if 'avg_loss' in metrics:
        print(f"\n📉 Loss Statistics:")
        print(f"   Average Loss: {metrics['avg_loss']:.6f}")
        print(f"   Std Loss: {metrics['std_loss']:.6f}")
        print(f"   Min Loss: {metrics['min_loss']:.6f}")
        print(f"   Max Loss: {metrics['max_loss']:.6f}")
    
    # Confusion matrix visualization
    if 'confusion_matrix' in metrics:
        print(f"\n📊 Confusion Matrix:")
        cm = metrics['confusion_matrix']
        classes = metrics['classes']
        
        # Print header
        header = " " * 12
        for cls in classes:
            header += f"{cls:>10}"
        print(header)
        
        # Print rows
        for i, cls in enumerate(classes):
            row = f"{cls:>10}:"
            for j in range(len(classes)):
                row += f"{cm[i][j]:>10}"
            print(row)
    
    print(f"\n❌ Error Samples (first 5):")
    for i, error in enumerate(metrics['error_samples'][:5]):
        print(f"   {i+1}. ID: {error['id']}")
        print(f"      True: {error['true']} → Predicted: {error['predicted']}")
    
    print("="*60)

def compute_detailed_metrics_by_category(data, category_key='task', positive_label='spoof'):
    """
    Compute metrics broken down by category (e.g., by task, by dataset, etc.)
    """
    # Group predictions by category
    category_data = defaultdict(lambda: {'y_true': [], 'y_pred': [], 'ids': []})
    
    for batch in data:
        categories = batch.get(category_key, ['unknown'] * len(batch.get('id', [])))
        ground_truth = batch.get('ground_truth', [])
        predictions = batch.get('text', [])
        ids = batch.get('id', [])
        
        for i, category in enumerate(categories):
            if i < len(ground_truth) and i < len(predictions):
                category_data[category]['y_true'].append(ground_truth[i])
                category_data[category]['y_pred'].append(predictions[i])
                if i < len(ids):
                    category_data[category]['ids'].append(ids[i])
    
    # Compute metrics for each category
    results = {}
    for category, data in category_data.items():
        y_true = np.array(data['y_true'])
        y_pred = np.array(data['y_pred'])
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        results[category] = {
            'samples': len(y_true),
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', labels=classes, zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', labels=classes, zero_division=0),
        }
        
        # Add binary metrics if applicable
        if len(classes) == 2:
            # Determine positive label
            if positive_label in classes:
                pos_label = positive_label
            else:
                pos_label = classes[1]
            
            results[category]['f1_binary'] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            results[category]['precision'] = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            results[category]['recall'] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    
    return results

def save_metrics_to_file(metrics, output_path='metrics_results.json'):
    """Save metrics to a JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    metrics_copy = metrics.copy()
    for key, value in metrics_copy.items():
        if isinstance(value, np.ndarray):
            metrics_copy[key] = value.tolist()
        elif isinstance(value, np.float32) or isinstance(value, np.float64):
            metrics_copy[key] = float(value)
        elif isinstance(value, np.int32) or isinstance(value, np.int64):
            metrics_copy[key] = int(value)
        elif isinstance(value, dict):
            # Handle nested dictionaries
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    value[sub_key] = sub_value.tolist()
                elif isinstance(sub_value, (np.float32, np.float64)):
                    value[sub_key] = float(sub_value)
                elif isinstance(sub_value, (np.int32, np.int64)):
                    value[sub_key] = int(sub_value)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_copy, f, indent=2)
    print(f"\n✅ Metrics saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Example data structure
    # Load your actual data
    with open('outputs/202606242045/eval_test_epoch_0.json', 'r') as f:
        data = json.load(f)
    
    
    # Compute metrics with 'spoof' as positive label
    metrics, y_true, y_pred, all_ids = compute_metrics_from_batched_outputs(data, positive_label='spoof')
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics
    save_metrics_to_file(metrics, 'evaluation_metrics.json')
    
    # Compute per-task metrics
    print("\n" + "="*60)
    print("PER-TASK METRICS")
    print("="*60)
    
    per_task_metrics = compute_detailed_metrics_by_category(data, 'task', positive_label='spoof')
    for task, task_metrics in per_task_metrics.items():
        print(f"\n📌 Task: {task}")
        print(f"   Samples: {task_metrics['samples']}")
        print(f"   Accuracy: {task_metrics['accuracy']:.4f}")
        print(f"   Balanced Accuracy: {task_metrics['balanced_accuracy']:.4f}")
        print(f"   F1 Macro: {task_metrics['f1_macro']:.4f}")
        if 'f1_binary' in task_metrics:
            print(f"   F1 Binary: {task_metrics['f1_binary']:.4f}")
            print(f"   Precision: {task_metrics['precision']:.4f}")
            print(f"   Recall: {task_metrics['recall']:.4f}")