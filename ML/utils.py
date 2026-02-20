import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, y_proba=None, class_labels=None):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }
    
    print("\n" + "="*50)
    print("         AVALIAÇÃO DO MODELO")
    print("="*50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print("="*50)
    
    if class_labels:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))
        
        cm = confusion_matrix(y_true, y_pred)
        return results, cm
    
    return results, None


def plot_confusion_matrix(cm, class_labels, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Matriz de Confusão")
    plt.ylabel("Real")
    plt.xlabel("Predito")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nMatriz de confusão salva em: {save_path}")
    
    plt.close()


def get_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        print("\nTop Features Importantes:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        return indices, importances[indices]
    return None, None
