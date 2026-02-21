#!/usr/bin/env python3
"""
PHORCE Decision Tree Analysis Script

This script uses decision trees to discover optimal feature ranges that
characterize domain-specific compound subsets (soil, water, crop related).
It extracts interpretable rules and feature importance rankings.

The analysis configuration is defined in the preprocessing_config.json file
under the "decision_tree_analysis" section.

Usage:
    python scripts/analyze_decision_trees.py
    python scripts/analyze_decision_trees.py --config custom_config.json
    python scripts/analyze_decision_trees.py --input labeled_data.csv
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Set matplotlib backend for non-interactive environments
plt.switch_backend('Agg')

# Set global font settings: Arial font, size 20
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20


@dataclass
class ModelResult:
    """Container for decision tree model results."""
    domain_name: str
    model: Any
    features_used: List[str]
    metrics: Dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    rules_text: str
    feature_importances: Dict[str, float]
    extracted_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PCAResult:
    """Container for PCA analysis results."""
    n_components: int
    explained_variance_ratio: List[float]
    cumulative_variance: List[float]
    loadings: Dict[str, List[float]]
    coordinates: Optional[np.ndarray] = None


@dataclass
class SimilarityResult:
    """Container for similarity analysis results."""
    domain_name: str
    n_samples: int
    mean_cosine_similarity: float
    std_cosine_similarity: float
    mean_euclidean_similarity: float
    std_euclidean_similarity: float
    intra_group_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Container for the full analysis report."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_compounds: int = 0
    models: Dict[str, Dict] = field(default_factory=dict)
    feature_statistics: Dict[str, Dict] = field(default_factory=dict)
    pca_results: Dict[str, Any] = field(default_factory=dict)
    similarity_results: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class DecisionTreeAnalyzer:
    """Main class for decision tree-based feature range discovery with PCA and similarity analysis."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config.get("decision_tree_analysis", {})
        self.logger = logger
        self.report = AnalysisReport()
        self.results: Dict[str, ModelResult] = {}
        self.pca_results: Dict[str, PCAResult] = {}
        self.similarity_results: Dict[str, SimilarityResult] = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load labeled data from CSV file."""
        self.logger.info(f"Loading data from {filepath}")
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise
            
        self.report.total_compounds = len(df)
        self.logger.info(f"Loaded {len(df)} compounds with {len(df.columns)} columns")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Prepare feature matrix from dataframe."""
        feature_config = self.config.get("features", {})
        
        numeric_features = feature_config.get("numeric", [
            "mw", "xlogp", "polararea", "complexity", "rotbonds",
            "hbonddonor", "hbondacc", "heavycnt", "charge"
        ])
        categorical_features = feature_config.get("categorical", [])
        
        # Filter to existing columns
        existing_numeric = [f for f in numeric_features if f in df.columns]
        existing_categorical = [f for f in categorical_features if f in df.columns]
        
        missing = set(numeric_features) - set(existing_numeric)
        if missing:
            self.logger.warning(f"Some numeric features not found: {missing}")
        
        missing_cat = set(categorical_features) - set(existing_categorical)
        if missing_cat:
            self.logger.warning(f"Some categorical features not found: {missing_cat}")
        
        all_features = existing_numeric + existing_categorical
        
        if not all_features:
            raise ValueError("No valid features found for analysis")
        
        self.logger.info(f"Using {len(existing_numeric)} numeric and {len(existing_categorical)} categorical features")
        
        return df[all_features], existing_numeric, existing_categorical
    
    def create_preprocessor(self, numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
        """Create sklearn preprocessing pipeline."""
        transformers = []
        
        if numeric_features:
            # No imputer needed - rows with missing values are dropped before analysis
            numeric_transformer = Pipeline([
                ("passthrough", "passthrough")
            ])
            transformers.append(("num", numeric_transformer, numeric_features))
        
        if categorical_features:
            categorical_transformer = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat", categorical_transformer, categorical_features))
        
        return ColumnTransformer(transformers, remainder='drop')
    
    def compute_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Compute ROC-AUC score handling edge cases."""
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                return roc_auc_score(y_true, y_proba[:, 1])
            elif y_proba.ndim == 2 and y_proba.shape[1] > 2:
                return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            else:
                return roc_auc_score(y_true, y_proba)
        except Exception:
            return np.nan
    
    def extract_feature_ranges(self, tree_model, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract feature threshold ranges from decision tree."""
        ranges = {}
        
        tree = tree_model.tree_
        
        for feature_idx, threshold in zip(tree.feature, tree.threshold):
            if feature_idx >= 0 and feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
                
                if feature_name not in ranges:
                    ranges[feature_name] = {
                        "thresholds": [],
                        "min_threshold": float('inf'),
                        "max_threshold": float('-inf')
                    }
                
                if threshold != -2.0:  # -2.0 indicates leaf node
                    ranges[feature_name]["thresholds"].append(float(threshold))
                    ranges[feature_name]["min_threshold"] = min(
                        ranges[feature_name]["min_threshold"], float(threshold)
                    )
                    ranges[feature_name]["max_threshold"] = max(
                        ranges[feature_name]["max_threshold"], float(threshold)
                    )
        
        # Clean up infinity values
        for feature_name in ranges:
            if ranges[feature_name]["min_threshold"] == float('inf'):
                ranges[feature_name]["min_threshold"] = None
            if ranges[feature_name]["max_threshold"] == float('-inf'):
                ranges[feature_name]["max_threshold"] = None
            
            # Calculate suggested range
            thresholds = ranges[feature_name]["thresholds"]
            if thresholds:
                ranges[feature_name]["suggested_range"] = [
                    round(min(thresholds), 4),
                    round(max(thresholds), 4)
                ]
        
        return ranges
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        domain_name: str,
        preprocessor: ColumnTransformer
    ) -> ModelResult:
        """Train a decision tree model for a specific domain."""
        self.logger.info(f"Training decision tree for domain: {domain_name}")
        
        model_settings = self.config.get("model_settings", {})
        random_seed = self.config.get("random_seed", 42)
        test_size = self.config.get("test_size", 0.3)
        cv_folds = self.config.get("cv_folds", 5)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state=random_seed
        )
        
        # Create pipeline
        tree = DecisionTreeClassifier(
            random_state=random_seed,
            class_weight=model_settings.get("class_weight", "balanced")
        )
        
        pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", tree)
        ])
        
        # Parameter grid for GridSearchCV
        param_grid = {
            "clf__max_depth": model_settings.get("max_depth_range", [2, 3, 4, 5]),
            "clf__min_samples_leaf": model_settings.get("min_samples_leaf_range", [5, 10, 20])
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        scoring = model_settings.get("scoring_metric", "f1_macro")
        
        gs = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=True)
        gs.fit(X_train, y_train)
        
        best_model = gs.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else np.zeros((len(y_pred), 2))
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        metrics = {
            "accuracy": float(report.get("accuracy", 0)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "precision_class_1": float(report.get("1", {}).get("precision", 0)),
            "recall_class_1": float(report.get("1", {}).get("recall", 0)),
            "f1_class_1": float(report.get("1", {}).get("f1-score", 0)),
            "support_class_0": int(report.get("0", {}).get("support", 0)),
            "support_class_1": int(report.get("1", {}).get("support", 0)),
            "roc_auc": float(self.compute_roc_auc(y_test, y_proba)),
            "cv_mean_score": float(gs.cv_results_['mean_test_score'][gs.best_index_]),
            "cv_std_score": float(gs.cv_results_['std_test_score'][gs.best_index_])
        }
        
        # Try to compute average precision
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                metrics["avg_precision"] = float(average_precision_score(y_test, y_proba[:, 1]))
        except Exception:
            metrics["avg_precision"] = np.nan
        
        # Feature names after preprocessing
        feature_names = list(best_model.named_steps["prep"].get_feature_names_out())
        
        # Feature importances
        importances = best_model.named_steps["clf"].feature_importances_
        feature_importance_dict = {
            name: float(imp) for name, imp in zip(feature_names, importances)
        }
        # Sort by importance
        feature_importance_dict = dict(sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        ))
        
        # Extract decision rules
        rule_config = self.config.get("rule_extraction", {})
        max_depth = rule_config.get("max_display_depth", 8)
        
        rules_text = export_text(
            best_model.named_steps["clf"],
            feature_names=feature_names,
            max_depth=max_depth
        )
        
        # Extract feature ranges from tree
        extracted_ranges = {}
        if rule_config.get("extract_feature_ranges", True):
            extracted_ranges = self.extract_feature_ranges(
                best_model.named_steps["clf"],
                feature_names
            )
        
        self.logger.info(f"  Best params: {gs.best_params_}")
        self.logger.info(f"  Macro F1: {metrics['macro_f1']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
        
        return ModelResult(
            domain_name=domain_name,
            model=best_model,
            features_used=list(X.columns),
            metrics=metrics,
            y_true=y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.array(y_test),
            y_pred=y_pred,
            y_proba=y_proba,
            rules_text=rules_text,
            feature_importances=feature_importance_dict,
            extracted_ranges=extracted_ranges,
            best_params={k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        )
    
    def plot_feature_importance(self, result: ModelResult, output_dir: Path):
        """Plot feature importance bar chart."""
        viz_config = self.config.get("visualization", {})
        if not viz_config.get("plot_feature_importance", True):
            return
        
        figsize = tuple(viz_config.get("figsize", [10, 8]))
        dpi = viz_config.get("dpi", 150)
        
        # Get top features
        importances = result.feature_importances
        top_n = min(15, len(importances))
        
        features = list(importances.keys())[:top_n]
        values = list(importances.values())[:top_n]
        
        # Clean feature names (remove num__ prefix)
        clean_names = [f.replace("num__", "").replace("cat__", "") for f in features]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
        bars = ax.barh(range(len(features)), values, color=colors)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(clean_names)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Feature Importance: {result.domain_name}")
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        output_path = output_dir / f"feature_importance_{result.domain_name}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  Saved feature importance plot: {output_path}")
    
    def plot_confusion_matrix(self, result: ModelResult, output_dir: Path):
        """Plot confusion matrix heatmap."""
        viz_config = self.config.get("visualization", {})
        if not viz_config.get("plot_confusion_matrix", True):
            return
        
        dpi = viz_config.get("dpi", 150)
        
        cm = confusion_matrix(result.y_true, result.y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not ' + result.domain_name, result.domain_name],
            yticklabels=['Not ' + result.domain_name, result.domain_name],
            ax=ax
        )
        
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix: {result.domain_name}")
        
        plt.tight_layout()
        
        output_path = output_dir / f"confusion_matrix_{result.domain_name}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  Saved confusion matrix: {output_path}")
    
    def plot_roc_curve(self, result: ModelResult, output_dir: Path):
        """Plot ROC curve."""
        viz_config = self.config.get("visualization", {})
        if not viz_config.get("plot_roc_curve", True):
            return
        
        dpi = viz_config.get("dpi", 150)
        
        if result.y_proba.ndim < 2 or result.y_proba.shape[1] < 2:
            self.logger.warning(f"Cannot plot ROC curve for {result.domain_name}: insufficient probability data")
            return
        
        try:
            fpr, tpr, _ = roc_curve(result.y_true, result.y_proba[:, 1])
            roc_auc = result.metrics.get("roc_auc", 0)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve: {result.domain_name}')
            ax.legend(loc="lower right")
            
            plt.tight_layout()
            
            output_path = output_dir / f"roc_curve_{result.domain_name}.png"
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"  Saved ROC curve: {output_path}")
        except Exception as e:
            self.logger.warning(f"Could not plot ROC curve for {result.domain_name}: {e}")
    
    def plot_decision_tree(self, result: ModelResult, output_dir: Path):
        """Plot the decision tree structure."""
        viz_config = self.config.get("visualization", {})
        dpi = viz_config.get("dpi", 150)
        
        tree_model = result.model.named_steps["clf"]
        feature_names = list(result.model.named_steps["prep"].get_feature_names_out())
        
        # Clean feature names
        clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
        
        # Determine figure size based on tree depth
        depth = tree_model.get_depth()
        fig_width = max(20, depth * 4)
        fig_height = max(10, depth * 2)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        plot_tree(
            tree_model,
            feature_names=clean_names,
            class_names=['No', 'Yes'],
            filled=True,
            rounded=True,
            ax=ax,
            fontsize=8
        )
        
        ax.set_title(f"Decision Tree: {result.domain_name}", fontsize=14)
        
        output_path = output_dir / f"decision_tree_{result.domain_name}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  Saved decision tree plot: {output_path}")
    
    def save_rules(self, result: ModelResult, output_dir: Path):
        """Save extracted rules to text file."""
        rule_config = self.config.get("rule_extraction", {})
        if not rule_config.get("save_rules_to_file", True):
            return
        
        output_path = output_dir / f"rules_{result.domain_name}.txt"
        
        with open(output_path, 'w') as f:
            f.write(f"Decision Tree Rules for: {result.domain_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Best Parameters: {result.best_params}\n\n")
            f.write("Model Performance:\n")
            f.write(f"  Macro F1: {result.metrics['macro_f1']:.4f}\n")
            f.write(f"  ROC-AUC: {result.metrics['roc_auc']:.4f}\n")
            f.write(f"  Precision (class 1): {result.metrics['precision_class_1']:.4f}\n")
            f.write(f"  Recall (class 1): {result.metrics['recall_class_1']:.4f}\n\n")
            f.write("=" * 60 + "\n")
            f.write("DECISION RULES:\n")
            f.write("=" * 60 + "\n\n")
            f.write(result.rules_text)
            f.write("\n\n")
            f.write("=" * 60 + "\n")
            f.write("FEATURE IMPORTANCES:\n")
            f.write("=" * 60 + "\n\n")
            for feat, imp in result.feature_importances.items():
                clean_name = feat.replace("num__", "").replace("cat__", "")
                f.write(f"  {clean_name:30s}: {imp:.4f}\n")
            
            if result.extracted_ranges:
                f.write("\n\n")
                f.write("=" * 60 + "\n")
                f.write("EXTRACTED FEATURE THRESHOLDS:\n")
                f.write("=" * 60 + "\n\n")
                for feat, ranges in result.extracted_ranges.items():
                    clean_name = feat.replace("num__", "").replace("cat__", "")
                    if ranges.get("suggested_range"):
                        f.write(f"  {clean_name}: {ranges['suggested_range']}\n")
        
        self.logger.info(f"  Saved rules to: {output_path}")
    
    def generate_visualizations(self, output_dir: Path, df: pd.DataFrame = None):
        """Generate all visualizations for trained models, PCA, and similarity analysis."""
        viz_config = self.config.get("visualization", {})
        
        if not viz_config.get("enabled", True):
            self.logger.info("Visualization is disabled in configuration")
            return
        
        self.logger.info("Generating visualizations...")
        
        # Decision tree visualizations
        for domain_name, result in self.results.items():
            self.logger.info(f"  Generating plots for {domain_name}...")
            self.plot_feature_importance(result, output_dir)
            self.plot_confusion_matrix(result, output_dir)
            self.plot_roc_curve(result, output_dir)
            self.plot_decision_tree(result, output_dir)
            self.save_rules(result, output_dir)
        
        # PCA visualizations
        if self.pca_results and 'overall' in self.pca_results:
            self.logger.info("  Generating PCA plots...")
            pca_result = self.pca_results['overall']
            self.plot_pca_variance(pca_result, output_dir)
            self.plot_pca_loadings(pca_result, output_dir)
            
            if df is not None and hasattr(self, '_X_numeric') and hasattr(self, '_existing_domains'):
                self.plot_pca_scatter(df, self._X_numeric, pca_result, self._existing_domains, output_dir)
        
        # Similarity visualizations
        if self.similarity_results:
            self.logger.info("  Generating similarity plots...")
            self.plot_similarity_comparison(output_dir)
            
            if df is not None and hasattr(self, '_cos_sim') and self._cos_sim is not None and hasattr(self, '_sample_indices'):
                for domain in self._existing_domains:
                    self.plot_similarity_heatmap(df, domain, self._cos_sim, self._sample_indices, output_dir)
    
    # =========================================================================
    # PCA Analysis Methods
    # =========================================================================
    
    def perform_pca(self, X: pd.DataFrame, n_components: Optional[int] = None) -> PCAResult:
        """
        Perform PCA on the feature matrix.
        
        Args:
            X: Feature DataFrame
            n_components: Number of components (default: min(n_features, 10))
            
        Returns:
            PCAResult with explained variance and loadings
        """
        pca_config = self.config.get("pca_analysis", {})
        
        if n_components is None:
            n_components = min(pca_config.get("n_components", 5), len(X.columns), 10)
        
        self.logger.info(f"Performing PCA with {n_components} components...")
        
        # Handle missing values
        X_clean = X.dropna()
        if len(X_clean) < 10:
            self.logger.warning("Insufficient data for PCA after dropping NaN values")
            return None
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca_coords = pca.fit_transform(X_scaled)
        
        # Calculate loadings (correlation between original features and components)
        loadings = {}
        for i, col in enumerate(X_clean.columns):
            loadings[col] = [float(pca.components_[j, i]) for j in range(n_components)]
        
        result = PCAResult(
            n_components=n_components,
            explained_variance_ratio=[float(v) for v in pca.explained_variance_ratio_],
            cumulative_variance=[float(v) for v in np.cumsum(pca.explained_variance_ratio_)],
            loadings=loadings,
            coordinates=pca_coords
        )
        
        self.logger.info(f"  Explained variance (first 3): {result.explained_variance_ratio[:3]}")
        self.logger.info(f"  Cumulative variance: {result.cumulative_variance[-1]:.3f}")
        
        return result
    
    def plot_pca_variance(self, pca_result: PCAResult, output_dir: Path):
        """Plot PCA explained variance (scree plot)."""
        viz_config = self.config.get("visualization", {})
        if not viz_config.get("plot_pca", True):
            return
        
        dpi = viz_config.get("dpi", 150)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scree plot
        ax1 = axes[0]
        components = range(1, len(pca_result.explained_variance_ratio) + 1)
        ax1.bar(components, pca_result.explained_variance_ratio, alpha=0.7, color='steelblue', label='Individual')
        ax1.plot(components, pca_result.cumulative_variance, 'ro-', label='Cumulative')
        ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='80% threshold')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Scree Plot')
        ax1.legend(loc='center right')
        ax1.set_xticks(components)
        
        # Cumulative variance
        ax2 = axes[1]
        ax2.plot(components, pca_result.cumulative_variance, 'b-o', linewidth=2, markersize=8)
        ax2.fill_between(components, pca_result.cumulative_variance, alpha=0.3)
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        ax2.set_xticks(components)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        output_path = output_dir / "pca_variance_explained.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  Saved PCA variance plot: {output_path}")
    
    def plot_pca_scatter(self, df: pd.DataFrame, X: pd.DataFrame, pca_result: PCAResult, 
                         domains: List[str], output_dir: Path):
        """Plot PCA scatter plots colored by domain membership."""
        viz_config = self.config.get("visualization", {})
        if not viz_config.get("plot_pca", True):
            return
        
        dpi = viz_config.get("dpi", 150)
        
        # Get clean indices (matching PCA coordinates)
        X_clean = X.dropna()
        clean_indices = X_clean.index
        
        # Create scatter plots for each domain
        for domain in domains:
            if domain not in df.columns:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get domain labels for clean data
            domain_labels = df.loc[clean_indices, domain].values
            
            # Custom colors
            colors = ['#1A85FF','#D41159']
            
            # Sort data so True values (1) are plotted last (on top)
            sort_idx = np.argsort(domain_labels)  # 0s first, then 1s
            coords_sorted = pca_result.coordinates[sort_idx]
            labels_sorted = domain_labels[sort_idx]
            
            scatter = ax.scatter(
                coords_sorted[:, 1],
                coords_sorted[:, 0],
                c=labels_sorted,
                cmap=plt.cm.colors.ListedColormap(colors),
                alpha=0.33,
                s=10,
                edgecolors='none'
            )
            
            ax.set_xlabel(f'PC1 ({pca_result.explained_variance_ratio[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({pca_result.explained_variance_ratio[1]:.1%} variance)')
            ax.set_title(f'PCA Projection: {domain.replace("_", " ").title()}')
            
            # Legend with True/False labels
            legend_labels = ['False', 'True']
            handles = [plt.scatter([], [], c=colors[i], s=50, label=legend_labels[i]) for i in range(2)]
            ax.legend(handles=handles, loc='upper right', title=domain.replace('_', ' ').title())
            
            plt.tight_layout()
            
            output_path = output_dir / f"pca_scatter_{domain}.png"
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"  Saved PCA scatter plot: {output_path}")
    
    def plot_pca_loadings(self, pca_result: PCAResult, output_dir: Path):
        """Plot PCA feature loadings heatmap."""
        viz_config = self.config.get("visualization", {})
        if not viz_config.get("plot_pca", True):
            return
        
        dpi = viz_config.get("dpi", 150)
        
        # Create loadings matrix
        features = list(pca_result.loadings.keys())
        n_components = len(pca_result.explained_variance_ratio)
        loadings_matrix = np.array([pca_result.loadings[f] for f in features])
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
        
        sns.heatmap(
            loadings_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            xticklabels=[f'PC{i+1}' for i in range(n_components)],
            yticklabels=features,
            ax=ax,
            cbar_kws={'label': 'Loading'}
        )
        
        ax.set_title('PCA Feature Loadings')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        output_path = output_dir / "pca_loadings_heatmap.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  Saved PCA loadings heatmap: {output_path}")
    
    # =========================================================================
    # Cosine Similarity Analysis Methods  
    # =========================================================================
    
    def compute_similarity_matrix(self, X: pd.DataFrame, max_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute cosine and euclidean similarity matrices.
        For large datasets, samples are taken to avoid memory issues.
        
        Args:
            X: Feature DataFrame
            max_samples: Maximum number of samples for similarity computation
            
        Returns:
            Tuple of (cosine_similarity_matrix, euclidean_similarity_matrix, sample_indices)
        """
        # Handle missing values
        X_clean = X.dropna()
        if len(X_clean) < 2:
            return None, None, None
        
        # Sample if dataset is too large
        sample_indices = X_clean.index.to_numpy()
        if len(X_clean) > max_samples:
            self.logger.info(f"  Sampling {max_samples} of {len(X_clean)} compounds for similarity analysis")
            np.random.seed(42)
            sample_idx = np.random.choice(len(X_clean), max_samples, replace=False)
            sample_indices = X_clean.index[sample_idx].to_numpy()
            X_clean = X_clean.iloc[sample_idx]
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Compute similarities
        cos_sim = cosine_similarity(X_scaled)
        
        # Convert euclidean distance to similarity (1 / (1 + distance))
        euc_dist = euclidean_distances(X_scaled)
        euc_sim = 1 / (1 + euc_dist)
        
        return cos_sim, euc_sim, sample_indices
    
    def compute_intra_group_similarity(
        self, 
        df: pd.DataFrame, 
        X: pd.DataFrame, 
        domain: str,
        cos_sim: np.ndarray,
        euc_sim: np.ndarray,
        sample_indices: np.ndarray
    ) -> SimilarityResult:
        """
        Compute average intra-group similarity for a domain.
        
        Args:
            df: Full DataFrame with domain labels
            X: Feature DataFrame
            domain: Domain column name
            cos_sim: Cosine similarity matrix
            euc_sim: Euclidean similarity matrix
            sample_indices: Indices of samples used in similarity matrix
            
        Returns:
            SimilarityResult with statistics
        """
        # Get domain labels for sampled data
        domain_labels = df.loc[sample_indices, domain].values
        positive_mask = domain_labels == 1
        positive_indices = np.where(positive_mask)[0]
        
        n_positive = len(positive_indices)
        
        if n_positive < 2:
            self.logger.warning(f"Insufficient samples in {domain} for similarity analysis")
            return SimilarityResult(
                domain_name=domain,
                n_samples=n_positive,
                mean_cosine_similarity=np.nan,
                std_cosine_similarity=np.nan,
                mean_euclidean_similarity=np.nan,
                std_euclidean_similarity=np.nan
            )
        
        # Extract sub-matrix for positive samples
        cos_sub = cos_sim[np.ix_(positive_indices, positive_indices)]
        euc_sub = euc_sim[np.ix_(positive_indices, positive_indices)]
        
        # Calculate mean similarity (excluding diagonal)
        n = cos_sub.shape[0]
        mask = ~np.eye(n, dtype=bool)
        
        cos_values = cos_sub[mask]
        euc_values = euc_sub[mask]
        
        result = SimilarityResult(
            domain_name=domain,
            n_samples=n_positive,
            mean_cosine_similarity=float(np.mean(cos_values)),
            std_cosine_similarity=float(np.std(cos_values)),
            mean_euclidean_similarity=float(np.mean(euc_values)),
            std_euclidean_similarity=float(np.std(euc_values)),
            intra_group_stats={
                'min_cosine': float(np.min(cos_values)),
                'max_cosine': float(np.max(cos_values)),
                'median_cosine': float(np.median(cos_values)),
                'min_euclidean': float(np.min(euc_values)),
                'max_euclidean': float(np.max(euc_values)),
                'median_euclidean': float(np.median(euc_values))
            }
        )
        
        self.logger.info(f"  {domain}: Cosine sim = {result.mean_cosine_similarity:.3f} ± {result.std_cosine_similarity:.3f}")
        self.logger.info(f"  {domain}: Euclidean sim = {result.mean_euclidean_similarity:.3f} ± {result.std_euclidean_similarity:.3f}")
        
        return result
    
    def plot_similarity_heatmap(
        self, 
        df: pd.DataFrame,
        domain: str,
        cos_sim: np.ndarray,
        sample_indices: np.ndarray,
        output_dir: Path,
        max_samples: int = 500
    ):
        """Plot cosine similarity heatmap for a domain."""
        viz_config = self.config.get("visualization", {})
        if not viz_config.get("plot_similarity", True):
            return
        
        dpi = viz_config.get("dpi", 150)
        
        # Get domain labels for sampled data
        domain_labels = df.loc[sample_indices, domain].values
        positive_indices = np.where(domain_labels == 1)[0]
        
        if len(positive_indices) < 2:
            return
        
        # Subsample if too large
        if len(positive_indices) > max_samples:
            np.random.seed(42)
            positive_indices = np.random.choice(positive_indices, max_samples, replace=False)
            positive_indices = np.sort(positive_indices)
        
        # Extract sub-matrix
        cos_sub = cos_sim[np.ix_(positive_indices, positive_indices)]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cos_sub,
            cmap='Blues',
            vmin=0,
            vmax=1,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Cosine Similarity'},
            ax=ax
        )
        
        ax.set_title(f'Cosine Similarity Matrix: {domain.replace("_", " ").title()} (n={len(positive_indices)})')
        ax.set_xlabel('Compound Index')
        ax.set_ylabel('Compound Index')
        
        plt.tight_layout()
        
        output_path = output_dir / f"similarity_heatmap_{domain}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  Saved similarity heatmap: {output_path}")
    
    def plot_similarity_comparison(self, output_dir: Path):
        """Plot comparison of intra-group similarities across domains."""
        viz_config = self.config.get("visualization", {})
        if not viz_config.get("plot_similarity", True):
            return
        
        if not self.similarity_results:
            return
        
        dpi = viz_config.get("dpi", 150)
        
        domains = list(self.similarity_results.keys())
        cos_means = [self.similarity_results[d].mean_cosine_similarity for d in domains]
        cos_stds = [self.similarity_results[d].std_cosine_similarity for d in domains]
        euc_means = [self.similarity_results[d].mean_euclidean_similarity for d in domains]
        euc_stds = [self.similarity_results[d].std_euclidean_similarity for d in domains]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        x = np.arange(len(domains))
        width = 0.6
        
        # Cosine similarity
        ax1 = axes[0]
        bars1 = ax1.bar(x, cos_means, width, yerr=cos_stds, capsize=5, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Mean Cosine Similarity')
        ax1.set_title('Intra-Group Cosine Similarity by Domain')
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.replace('_', '\n') for d in domains])
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars1, cos_means):
            if not np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Euclidean similarity
        ax2 = axes[1]
        bars2 = ax2.bar(x, euc_means, width, yerr=euc_stds, capsize=5, color='darkorange', alpha=0.8)
        ax2.set_ylabel('Mean Euclidean Similarity')
        ax2.set_title('Intra-Group Euclidean Similarity by Domain')
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.replace('_', '\n') for d in domains])
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars2, euc_means):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        output_path = output_dir / "similarity_comparison.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  Saved similarity comparison plot: {output_path}")
    
    # =========================================================================
    # Feature Statistics
    # =========================================================================
    
    def compute_feature_statistics(self, df: pd.DataFrame, X: pd.DataFrame, domains: List[str]) -> Dict:
        """Compute descriptive statistics for features by domain."""
        stats = {}
        
        for feature in X.columns:
            feature_data = pd.to_numeric(df[feature], errors='coerce')
            
            stats[feature] = {
                "overall": {
                    "mean": float(feature_data.mean()) if not feature_data.isna().all() else None,
                    "std": float(feature_data.std()) if not feature_data.isna().all() else None,
                    "min": float(feature_data.min()) if not feature_data.isna().all() else None,
                    "max": float(feature_data.max()) if not feature_data.isna().all() else None,
                    "median": float(feature_data.median()) if not feature_data.isna().all() else None
                }
            }
            
            for domain in domains:
                if domain in df.columns:
                    domain_data = feature_data[df[domain] == 1]
                    if len(domain_data) > 0 and not domain_data.isna().all():
                        stats[feature][domain] = {
                            "mean": float(domain_data.mean()),
                            "std": float(domain_data.std()) if len(domain_data) > 1 else 0,
                            "min": float(domain_data.min()),
                            "max": float(domain_data.max()),
                            "median": float(domain_data.median()),
                            "count": int(domain_data.notna().sum())
                        }
        
        return stats
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, ModelResult]:
        """Run the full decision tree analysis pipeline with PCA and similarity analysis."""
        self.logger.info("Starting decision tree analysis...")
        
        # Prepare features
        X, numeric_features, categorical_features = self.prepare_features(df)
        
        # Get target domains
        target_domains = self.config.get("target_domains", ["soil_related", "water_related", "crop_related"])
        existing_domains = [d for d in target_domains if d in df.columns]
        
        if not existing_domains:
            raise ValueError(f"No target domains found in data. Expected: {target_domains}")
        
        # Drop rows with missing values in feature columns
        all_features = list(X.columns)
        original_count = len(df)
        missing_mask = X.isna().any(axis=1)
        n_missing = missing_mask.sum()
        
        if n_missing > 0:
            self.logger.info(f"Removing {n_missing:,} rows with missing values ({n_missing/original_count*100:.2f}%)")
            df = df[~missing_mask].copy()
            X = X[~missing_mask].copy()
            self.logger.info(f"Remaining rows: {len(df):,}")
        
        self.logger.info(f"Analyzing {len(existing_domains)} domains: {existing_domains}")
        
        # =====================================================================
        # PCA Analysis
        # =====================================================================
        pca_config = self.config.get("pca_analysis", {})
        if pca_config.get("enabled", True):
            self.logger.info("\n" + "="*60)
            self.logger.info("PERFORMING PCA ANALYSIS")
            self.logger.info("="*60)
            
            pca_result = self.perform_pca(X[numeric_features])
            if pca_result:
                self.pca_results['overall'] = pca_result
                self.report.pca_results = {
                    'n_components': pca_result.n_components,
                    'explained_variance_ratio': pca_result.explained_variance_ratio,
                    'cumulative_variance': pca_result.cumulative_variance,
                    'loadings': pca_result.loadings
                }
        
        # =====================================================================
        # Similarity Analysis
        # =====================================================================
        sim_config = self.config.get("similarity_analysis", {})
        sample_indices = None
        cos_sim = None
        euc_sim = None
        
        if sim_config.get("enabled", True):
            self.logger.info("\n" + "="*60)
            self.logger.info("PERFORMING SIMILARITY ANALYSIS")
            self.logger.info("="*60)
            
            # Compute similarity matrices once (with sampling for large datasets)
            max_samples = sim_config.get("max_samples", 5000)
            if max_samples is None:
                max_samples = 5000  # Handle null from JSON config
            cos_sim, euc_sim, sample_indices = self.compute_similarity_matrix(X[numeric_features], max_samples)
            
            if cos_sim is not None:
                for domain in existing_domains:
                    if domain in df.columns:
                        sim_result = self.compute_intra_group_similarity(
                            df, X[numeric_features], domain, cos_sim, euc_sim, sample_indices
                        )
                        self.similarity_results[domain] = sim_result
                        self.report.similarity_results[domain] = {
                            'n_samples': sim_result.n_samples,
                            'mean_cosine_similarity': sim_result.mean_cosine_similarity,
                            'std_cosine_similarity': sim_result.std_cosine_similarity,
                            'mean_euclidean_similarity': sim_result.mean_euclidean_similarity,
                            'std_euclidean_similarity': sim_result.std_euclidean_similarity,
                            'intra_group_stats': sim_result.intra_group_stats
                        }
        
        # =====================================================================
        # Decision Tree Training
        # =====================================================================
        self.logger.info("\n" + "="*60)
        self.logger.info("TRAINING DECISION TREE MODELS")
        self.logger.info("="*60)
        
        for domain in existing_domains:
            y = df[domain]
            
            # Check class balance
            class_counts = y.value_counts()
            self.logger.info(f"\n{domain} class distribution: {dict(class_counts)}")
            
            if len(class_counts) < 2:
                self.logger.warning(f"Skipping {domain}: only one class present")
                continue
            
            if class_counts.min() < 10:
                self.logger.warning(f"Warning: {domain} has very few samples in minority class ({class_counts.min()})")
            
            # Create preprocessor
            preprocessor = self.create_preprocessor(numeric_features, categorical_features)
            
            # Train model
            result = self.train_model(X, y, domain, preprocessor)
            self.results[domain] = result
            
            # Add to report
            self.report.models[domain] = {
                "metrics": result.metrics,
                "best_params": result.best_params,
                "feature_importances": result.feature_importances,
                "extracted_ranges": result.extracted_ranges
            }
        
        # Compute feature statistics
        self.report.feature_statistics = self.compute_feature_statistics(df, X, existing_domains)
        
        # Store similarity matrices for visualization
        self._cos_sim = cos_sim
        self._euc_sim = euc_sim
        self._sample_indices = sample_indices
        self._X_numeric = X[numeric_features]
        self._existing_domains = existing_domains
        
        self.logger.info("\nDecision tree analysis complete")
        
        return self.results


def setup_logging(config: dict) -> logging.Logger:
    """Set up logging based on configuration."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("log_level", "INFO").upper())
    
    logger = logging.getLogger("phorce_decision_tree")
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PHORCE Decision Tree Analysis Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/analyze_decision_trees.py
    python scripts/analyze_decision_trees.py --config custom_config.json
    python scripts/analyze_decision_trees.py --input labeled_data.csv
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='preprocessing_config.json',
        help='Path to JSON configuration file (default: preprocessing_config.json)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input CSV file path (overrides config)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory path (overrides config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def print_summary(analyzer: DecisionTreeAnalyzer):
    """Print a formatted summary of the analysis results including PCA and similarity."""
    results = analyzer.results
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    # PCA Summary
    if analyzer.pca_results and 'overall' in analyzer.pca_results:
        print("\n" + "─" * 70)
        print("📈 PCA ANALYSIS")
        print("─" * 70)
        pca = analyzer.pca_results['overall']
        print(f"  Components: {pca.n_components}")
        print(f"  Total Variance Explained: {pca.cumulative_variance[-1]:.1%}")
        print(f"  Variance by Component:")
        for i, var in enumerate(pca.explained_variance_ratio[:5]):
            print(f"    PC{i+1}: {var:.1%}")
    
    # Similarity Summary
    if analyzer.similarity_results:
        print("\n" + "─" * 70)
        print("🔗 SIMILARITY ANALYSIS")
        print("─" * 70)
        for domain, sim in analyzer.similarity_results.items():
            print(f"\n  {domain}:")
            print(f"    Samples: {sim.n_samples}")
            if not np.isnan(sim.mean_cosine_similarity):
                print(f"    Cosine Similarity:    {sim.mean_cosine_similarity:.3f} ± {sim.std_cosine_similarity:.3f}")
                print(f"    Euclidean Similarity: {sim.mean_euclidean_similarity:.3f} ± {sim.std_euclidean_similarity:.3f}")
    
    # Decision Tree Summary
    print("\n" + "─" * 70)
    print("🌲 DECISION TREE MODELS")
    print("─" * 70)
    
    for domain, result in results.items():
        print(f"\n  📊 {domain.upper()}")
        print(f"  {'─' * 40}")
        print(f"    Best Parameters: {result.best_params}")
        print(f"    Macro F1:        {result.metrics['macro_f1']:.4f}")
        print(f"    ROC-AUC:         {result.metrics['roc_auc']:.4f}")
        print(f"    Precision:       {result.metrics['precision_class_1']:.4f}")
        print(f"    Recall:          {result.metrics['recall_class_1']:.4f}")
        print(f"    CV Score:        {result.metrics['cv_mean_score']:.4f} ± {result.metrics['cv_std_score']:.4f}")
        
        print(f"\n    Top Features:")
        for i, (feat, imp) in enumerate(list(result.feature_importances.items())[:5]):
            clean_name = feat.replace("num__", "").replace("cat__", "")
            print(f"      {i+1}. {clean_name:25s}: {imp:.4f}")
        
        if result.extracted_ranges:
            print(f"\n    Key Thresholds:")
            for feat, ranges in list(result.extracted_ranges.items())[:5]:
                if ranges.get("suggested_range"):
                    clean_name = feat.replace("num__", "").replace("cat__", "")
                    print(f"      {clean_name:25s}: {ranges['suggested_range']}")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point for the decision tree analysis script."""
    args = parse_arguments()
    
    # Determine the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Check if analysis is enabled
    analysis_config = config.get("decision_tree_analysis", {})
    if not analysis_config.get("enabled", True):
        print("Decision tree analysis is disabled in configuration. Exiting.")
        sys.exit(0)
    
    # Override verbose setting if specified
    if args.verbose:
        config.setdefault("logging", {})["log_level"] = "DEBUG"
    
    # Set up logging
    logger = setup_logging(config)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Determine input file
    input_file = args.input or analysis_config.get("input_file", "data/labeled/domain_labeled_compounds.csv")
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output directory
    output_dir_str = args.output or analysis_config.get("output_dir", "data/analysis")
    output_dir = Path(output_dir_str)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = DecisionTreeAnalyzer(config, logger)
    
    # Load and analyze data
    try:
        df = analyzer.load_data(str(input_path))
        results = analyzer.analyze(df)
        
        # Generate visualizations
        analyzer.generate_visualizations(output_dir, df)
        
        # Save report
        report_path = analysis_config.get("report_output", "data/analysis/decision_tree_report.json")
        report_full_path = Path(report_path)
        if not report_full_path.is_absolute():
            report_full_path = project_root / report_full_path
        
        report_full_path.parent.mkdir(parents=True, exist_ok=True)
        analyzer.report.save(str(report_full_path))
        logger.info(f"Analysis report saved to {report_full_path}")
        
        # Print summary
        print_summary(analyzer)
        
    except Exception as e:
        logger.error(f"Decision tree analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
