import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path

class ModelEvaluator:
    """
    Class for evaluating trained earthquake prediction models.
    """
    
    def __init__(
        self, 
        model, 
        data_loader, 
        device=None, 
        class_names=None,
        output_dir="./evaluation_results"
    ):
        """
        Initialize the evaluator with a model and data loader.
        
        Args:
            model: The trained PyTorch model to evaluate
            data_loader: PyTorch DataLoader for evaluation data
            device: Device to run evaluation on ('cuda' or 'cpu')
            class_names: Names of the magnitude classes for plots
            output_dir: Directory to save evaluation plots and results
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.class_names = class_names or [
            'Low (< 1.5)', 
            'Moderate (1.5–2.5)', 
            'High (2.5–3.5)', 
            'Very High (> 3.5)'
        ]
        self.output_dir = output_dir
        self.output_dir  = Path(output_dir)          # ← make it a Path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = type(model).__name__
        self.emb_pickle  = (
            self.output_dir / f"{self.model_name}_sample_embeddings.pkl"
        )
        # Determine if model uses tabular features
        self.use_tabular_features = hasattr(model, 'use_tabular_features') and model.use_tabular_features
        if self.use_tabular_features:
            print("Model uses tabular features. Evaluation will include them.")
        

        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_predictions_and_embeddings(self):
        self.model.eval()
        all_probs, all_preds, all_labels = [], [], []
        records = []

        with torch.no_grad():
            for batch_data in tqdm(self.data_loader, desc="Evaluating"):

                if self.use_tabular_features:
                    windows, tab_feats, metas, label = batch_data
                    windows, tab_feats, label = (
                        windows.to(self.device),
                        tab_feats.to(self.device),
                        label.to(self.device),
                    )
                    logits, emb = self.model(windows, tab_feats, return_embedding=True)
                else:
                    windows, metas, label = batch_data
                    windows, label = windows.to(self.device), label.to(self.device)
                    logits, emb = self.model(windows, return_embedding=True)

                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

                emb_np = emb.detach().cpu().numpy()          # [B, H]
                for i, meta in enumerate(metas):             # metas aligns with batch
                    records.append({
                        "station":      meta["station"],              # ← NEW
                        "period_start": meta["period_start"],
                        "period_end":   meta["period_end"],
                        "label":        meta["label"],                # ← NEW
                        "embedding":    emb_np[i] 
                                        })

        return (np.array(all_probs),
                np.array(all_preds),
                np.array(all_labels),
                records)

        
    def collect_predictions(self):
        """
        Collect predictions and ground truth labels from the model.
        Now handles models that use tabular features.
        
        Returns:
            Tuple of (all_probs, all_preds, all_labels)
        """
        self.model.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.data_loader, desc="Evaluating"):
                # Handle different batch structures based on tabular features usage  
                if self.use_tabular_features:
                    # Unpack structured batch data with tabular features
                    windows_tensor, tabular_features, label = batch_data
                    windows_tensor = windows_tensor.to(self.device)
                    tabular_features = tabular_features.to(self.device)
                    label = label.to(self.device)
                    
                    # Forward pass with tabular features
                    outputs = self.model(windows_tensor, tabular_features)
                else:
                    # Standard batch structure
                    windows_tensor, label = batch_data
                    windows_tensor = windows_tensor.to(self.device)
                    label = label.to(self.device)
                    
                    # Forward pass without tabular features
                    outputs = self.model(windows_tensor)
                
                # Calculate probabilities, predictions
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Collect results
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        
        return np.array(all_probs), np.array(all_preds), np.array(all_labels)
    
    def compute_confusion_matrix(self, predictions=None, labels=None, normalize=False):
        """
        Compute and visualize confusion matrix.
        
        Args:
            predictions: Optional precomputed predictions
            labels: Optional precomputed labels
            normalize: Whether to normalize the confusion matrix
        
        Returns:
            Confusion matrix
        """
        if predictions is None or labels is None:
            _, predictions, labels = self.collect_predictions()
        
        conf_matrix = confusion_matrix(labels, predictions)
        
        if normalize:
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Visualization
        plt.figure(figsize=(10, 8))
        fmt = '.2f' if normalize else 'd'
        sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'{self.output_dir}/confusion_matrix_{self.model_name}.png')
        plt.show()
        
        return conf_matrix
    
    def compute_classification_report(self, predictions=None, labels=None):
        """
        Generate and print classification report.
        
        Args:
            predictions: Optional precomputed predictions
            labels: Optional precomputed labels
            
        Returns:
            Classification report dictionary
        """
        if predictions is None or labels is None:
            _, predictions, labels = self.collect_predictions()
        
        report = classification_report(
            labels,
            predictions,
            labels=range(len(self.class_names)),
            target_names=self.class_names,
            zero_division=0,
            output_dict=True
        )
        
        # Print text report
        print("\nClassification Report:")
        print(classification_report(
            labels,
            predictions,
            labels=range(len(self.class_names)),
            target_names=self.class_names,
            zero_division=0
        ))
        
        # Calculate overall accuracy
        accuracy = accuracy_score(labels, predictions)
        print(f"Overall Accuracy: {accuracy:.4f}")
        report['accuracy'] = accuracy
        
        return report
    
    def plot_roc_curves(self, probabilities=None, labels=None):
        """
        Plot ROC curves for each class.
        
        Args:
            probabilities: Optional precomputed class probabilities
            labels: Optional precomputed ground truth labels
            
        Returns:
            Dictionary of AUC values for each class
        """
        if probabilities is None or labels is None:
            probabilities, _, labels = self.collect_predictions()
        
        # Binarize the labels for OvR AUC calculation
        n_classes = len(self.class_names)
        labels_binarized = label_binarize(labels, classes=range(n_classes))
        
        # Calculate ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        colors = ['blue', 'orange', 'green', 'red']
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels_binarized[:, i], probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                     label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves by Class - {self.model_name}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'{self.output_dir}/roc_curves_{self.model_name}.png')
        plt.show()
        
        return roc_auc
    
    def perform_feature_importance_analysis(self):
        """
        Analyze the importance of tabular features if the model uses them.
        This is a placeholder for feature importance analysis.
        You would need to implement a proper feature importance analysis method
        based on your specific model type.
        """
        if not self.use_tabular_features:
            print("Model does not use tabular features. Skipping feature importance analysis.")
            return None
        
        print("\nFeature Importance Analysis:")
        print("Note: This is a placeholder for feature importance analysis.")
        print("For a proper analysis, you could implement methods like:")
        print("  - Permutation importance")
        print("  - SHAP values")
        print("  - Integrated gradients")
        print("  - Feature ablation studies")
        
        # This is where you would implement your feature importance analysis
        # For example, if using a model that exposes feature importances:
        # importances = self.model.tabular_projection.weight.detach().cpu().numpy()
        
        return None
    
    
    def evaluate(self, save_results=True):
        """
        Run full evaluation suite and optionally save results.
        Now includes feature importance analysis for tabular features.
        
        Args:
            save_results: Whether to save results to output directory
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"Starting comprehensive evaluation of {self.model_name}...")
        print(f"Model {'uses' if self.use_tabular_features else 'does not use'} tabular features")
        
        # Collect predictions once for all metrics
        probs, preds, labels, records = self.collect_predictions_and_embeddings()
        
        # Compute all metrics
        conf_matrix = self.compute_confusion_matrix(preds, labels)
        report = self.compute_classification_report(preds, labels)
        roc_auc = self.plot_roc_curves(probs, labels)
        
        # Feature importance analysis if using tabular features
        feature_importance = None
        if self.use_tabular_features:
            feature_importance = self.perform_feature_importance_analysis()
        
        # Prepare results summary
        results = {
            "model_name": self.model_name,
            "uses_tabular_features": self.use_tabular_features,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": report,
            "roc_auc": roc_auc,
            "accuracy": report['accuracy']
        }
        
        if self.use_tabular_features and feature_importance is not None:
            results["feature_importance"] = feature_importance
        
        if save_results:
            import json
            with open(f'{self.output_dir}/evaluation_results_{self.model_name}.json', 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"Evaluation results saved to {self.output_dir}/evaluation_results_{self.model_name}.json")
        
        if records:
            df = pd.DataFrame.from_records(records)
            df.to_pickle(self.emb_pickle, protocol=4)
            print(f"Saved {len(df):,} sample-level embeddings → {self.emb_pickle.resolve()}")
        
        return results
    
    def evaluate_model_on_class(self, class_idx):
        """
        Evaluate model performance on a specific class.
        
        Args:
            class_idx: Index of the class to evaluate
            
        Returns:
            Dictionary with class-specific metrics
        """
        _, preds, labels = self.collect_predictions()
        
        # Get predictions for this specific class
        class_mask = (labels == class_idx)
        if not any(class_mask):
            print(f"No samples found for class {self.class_names[class_idx]}")
            return None
        
        # Calculate class-specific metrics
        class_preds = preds[class_mask]
        class_labels = labels[class_mask]
        
        class_accuracy = np.mean(class_preds == class_labels)
        
        print(f"\nEvaluation for class: {self.class_names[class_idx]}")
        print(f"Samples: {sum(class_mask)}")
        print(f"Accuracy: {class_accuracy:.4f}")
        
        return {
            "class_name": self.class_names[class_idx],
            "samples": int(sum(class_mask)),
            "accuracy": float(class_accuracy)
        }