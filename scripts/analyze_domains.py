#!/usr/bin/env python3
"""
PHORCE Domain-Level XGBoost Analysis Script

This script performs XGBoost decision tree analysis on the main environmental domains:
- soil_related
- water_related  
- crop_related

Pipeline Stage: 2 (After domain labeling)
Input: data/labeled/domain_labeled_compounds.csv OR data/engineered/P1M_engineered.csv
Output: data/analysis/domains/

Usage:
    python scripts/analyze_domains.py
    python scripts/analyze_domains.py --config custom_config.json
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score,
    precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from xgboost import XGBClassifier

plt.switch_backend('Agg')

# Set global font settings for publication quality figures
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 28
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 36
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2


def format_label(text: str) -> str:
    """Format labels for publication: remove underscores and apply title case."""
    return text.replace("_", " ").title()


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("phorce_domain_analysis")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _to_json_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(item) for item in obj]
    return obj


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelResult:
    """Container for XGBoost model results."""
    domain_name: str
    model: Any
    features_used: List[str]
    metrics: Dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    X_test: pd.DataFrame  # Store test features for FP analysis
    rules_text: str
    feature_importances: Dict[str, float]
    extracted_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PCAResult:
    """Container for PCA results."""
    n_components: int
    explained_variance_ratio: List[float]
    cumulative_variance: List[float]
    loadings: Dict[str, List[float]]
    coordinates: Optional[np.ndarray] = None


@dataclass
class DomainAnalysisReport:
    """Report for domain-level analysis."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    input_file: str = ""
    output_dir: str = ""
    total_compounds: int = 0
    compounds_after_missing_removal: int = 0
    domains_analyzed: List[str] = field(default_factory=list)
    domain_counts: Dict[str, int] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    pca_results: Dict[str, Any] = field(default_factory=dict)
    similarity_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return _to_json_serializable({
            "timestamp": self.timestamp,
            "input_file": self.input_file,
            "output_dir": self.output_dir,
            "total_compounds": self.total_compounds,
            "compounds_after_missing_removal": self.compounds_after_missing_removal,
            "domains_analyzed": self.domains_analyzed,
            "domain_counts": self.domain_counts,
            "models": self.models,
            "pca_results": self.pca_results,
            "similarity_results": self.similarity_results
        })


# =============================================================================
# DOMAIN ANALYZER
# =============================================================================

class DomainAnalyzer:
    """XGBoost analyzer for environmental domains."""
    
    # Target domains for this analysis level
    TARGET_DOMAINS = ["soil_related", "water_related", "crop_related"]
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("phorce_domain_analysis")
        self.report = DomainAnalysisReport()
        self.results: Dict[str, ModelResult] = {}
        self.pca_result: Optional[PCAResult] = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature matrix."""
        analysis_config = self.config.get("domain_analysis", {})
        feature_config = analysis_config.get("features", {})
        
        numeric_features = feature_config.get("numeric", [
            "mw", "xlogp", "polararea", "hbonddonor", "hbondacc", 
            "complexity", "rotbonds", "charge"
        ])
        
        # Filter to existing columns
        existing = [f for f in numeric_features if f in df.columns]
        missing = set(numeric_features) - set(existing)
        if missing:
            self.logger.warning(f"Features not found: {missing}")
        
        if not existing:
            raise ValueError("No valid features found")
        
        self.logger.info(f"Using {len(existing)} numeric features: {existing}")
        return df[existing], existing
    
    def create_preprocessor(self, numeric_features: List[str]) -> ColumnTransformer:
        """Create preprocessing pipeline (passthrough since we drop NaN)."""
        return ColumnTransformer([
            ("num", "passthrough", numeric_features)
        ], remainder='drop')
    
    def extract_feature_ranges(self, model: XGBClassifier, feature_names: List[str], X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Extract feature threshold ranges based on feature importance from XGBoost."""
        ranges = {}
        
        # Get feature importances
        importance_dict = model.get_booster().get_score(importance_type='gain')
        
        # Map feature importance to actual feature names and calculate ranges from data
        for i, feature_name in enumerate(feature_names):
            # XGBoost uses f0, f1, etc. as feature names internally
            xgb_feature_name = f'f{i}'
            if xgb_feature_name in importance_dict or feature_name in importance_dict:
                if feature_name in X.columns:
                    col_data = X[feature_name].dropna()
                    if len(col_data) > 0:
                        ranges[feature_name] = {
                            "min": float(col_data.min()),
                            "max": float(col_data.max()),
                            "mean": float(col_data.mean()),
                            "std": float(col_data.std())
                        }
        
        return ranges
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, domain: str, 
                    preprocessor: ColumnTransformer) -> ModelResult:
        """Train an XGBoost model for a domain."""
        self.logger.info(f"Training XGBoost model for domain: {domain}")
        self.logger.info(f"Model settings: {self.config.get('domain_analysis', {}).get('model_settings', {})}")
        analysis_config = self.config.get("domain_analysis", {})
        model_settings = analysis_config.get("model_settings", {})
        
        # Split data
        test_size = analysis_config.get("test_size", 0.3)
        random_seed = analysis_config.get("random_seed", 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )
        
        # Calculate scale_pos_weight for imbalanced classes
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        # Create pipeline with XGBoost
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=random_seed,
                eval_metric='logloss',
                n_jobs=4  # Multi-threaded to improve training speed
            ))
        ])
        
        # Grid search with XGBoost parameters (reduced grid for memory efficiency)
        param_grid = {
            "clf__max_depth": model_settings.get("max_depth_range", [3, 5, 7]),
            "clf__n_estimators": model_settings.get("n_estimators_range", [50, 100]),
            "clf__learning_rate": model_settings.get("learning_rate_range", [0.1, 0.3])
        }
        
        cv = StratifiedKFold(n_splits=analysis_config.get("cv_folds", 5), shuffle=True, random_state=random_seed)
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv,
            scoring=model_settings.get("scoring_metric", "f1_macro"),
            n_jobs=1, refit=True  # Sequential to avoid disk space issues
        )
        
        grid_search.fit(X_train, y_train)
        
        # Predictions
        y_pred = grid_search.predict(X_test)
        y_proba = grid_search.predict_proba(X_test)
        
        # Metrics
        metrics = {
            "accuracy": float((y_pred == y_test).mean()),
            "f1_macro": float(f1_score(y_test, y_pred, average='macro')),
            "precision": float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
            "cv_mean": float(grid_search.cv_results_['mean_test_score'][grid_search.best_index_]),
            "cv_std": float(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
        }
        
        # ROC-AUC
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
        except:
            metrics["roc_auc"] = np.nan
        
        self.logger.info(f"  Best params: {grid_search.best_params_}")
        self.logger.info(f"  Macro F1: {metrics['f1_macro']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
        
        # Extract model info
        xgb_model = grid_search.best_estimator_.named_steps['clf']
        feature_names = list(X.columns)
        
        # Get feature importances
        importances = dict(zip(feature_names, xgb_model.feature_importances_))
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        # Generate rules text from feature importances
        rules_text = self._generate_rules_text(xgb_model, feature_names, importances)
        
        ranges = self.extract_feature_ranges(xgb_model, feature_names, X)
        
        best_params = {k.replace("clf__", ""): v for k, v in grid_search.best_params_.items()}
        
        return ModelResult(
            domain_name=domain,
            model=grid_search.best_estimator_,
            features_used=feature_names,
            metrics=metrics,
            y_true=y_test.values,
            y_pred=y_pred,
            y_proba=y_proba,
            X_test=X_test,  # Store test features for FP analysis
            rules_text=rules_text,
            feature_importances=importances,
            extracted_ranges=ranges,
            best_params=best_params
        )
    
    def _generate_rules_text(self, model: XGBClassifier, feature_names: List[str], 
                            importances: Dict[str, float]) -> str:
        """Generate a text summary of the XGBoost model."""
        lines = []
        lines.append("XGBoost Model Summary")
        lines.append("=" * 50)
        lines.append(f"\nNumber of estimators: {model.n_estimators}")
        lines.append(f"Max depth: {model.max_depth}")
        lines.append(f"Learning rate: {model.learning_rate}")
        lines.append(f"\nFeature Importances (sorted by importance):")
        lines.append("-" * 40)
        
        for feature, importance in importances.items():
            if importance > 0:
                lines.append(f"  {feature}: {importance:.4f}")
        
        return "\n".join(lines)
    
    def perform_pca(self, X: pd.DataFrame, n_components: int = 5) -> Optional[PCAResult]:
        """Perform PCA analysis."""
        X_clean = X.dropna()
        if len(X_clean) < 10:
            return None
        
        n_components = min(n_components, len(X.columns), len(X_clean))
        
        X_scaled = self.scaler.fit_transform(X_clean)
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(X_scaled)
        
        loadings = {col: pca.components_[:, i].tolist() for i, col in enumerate(X.columns)}
        
        self.logger.info(f"Performing PCA with {n_components} components...")
        self.logger.info(f"  Explained variance (first 3): {pca.explained_variance_ratio_[:3].tolist()}")
        
        return PCAResult(
            n_components=n_components,
            explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
            cumulative_variance=np.cumsum(pca.explained_variance_ratio_).tolist(),
            loadings=loadings,
            coordinates=coords
        )
    
    def compute_similarity(self, X: pd.DataFrame, df: pd.DataFrame, domain: str, 
                          max_samples: int = 5000) -> Dict[str, Any]:
        """Compute intra-group similarity for a domain."""
        # Work with aligned indices
        X_clean = X.dropna()
        df_aligned = df.loc[X_clean.index]
        
        if len(X_clean) < 2:
            return {}
        
        # Sample if needed
        if len(X_clean) > max_samples:
            np.random.seed(42)
            sample_indices = np.random.choice(X_clean.index.tolist(), max_samples, replace=False)
            X_sampled = X_clean.loc[sample_indices]
            df_sampled = df_aligned.loc[sample_indices]
        else:
            X_sampled = X_clean
            df_sampled = df_aligned
        
        X_scaled = self.scaler.fit_transform(X_sampled)
        
        # Get domain indices
        domain_mask = df_sampled[domain] == 1
        domain_indices = np.where(domain_mask)[0]
        
        if len(domain_indices) < 2:
            return {"n_samples": 0}
        
        # Compute similarity matrices
        cos_sim = cosine_similarity(X_scaled)
        euc_dist = euclidean_distances(X_scaled)
        euc_sim = 1 / (1 + euc_dist)
        
        # Intra-group similarity
        cos_group = cos_sim[np.ix_(domain_indices, domain_indices)]
        euc_group = euc_sim[np.ix_(domain_indices, domain_indices)]
        
        # Exclude diagonal
        n = len(domain_indices)
        mask = ~np.eye(n, dtype=bool)
        
        return {
            "n_samples": n,
            "cosine_mean": float(cos_group[mask].mean()),
            "cosine_std": float(cos_group[mask].std()),
            "euclidean_mean": float(euc_group[mask].mean()),
            "euclidean_std": float(euc_group[mask].std())
        }
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, ModelResult]:
        """Run the full analysis pipeline."""
        self.logger.info("Starting domain-level decision tree analysis...")
        
        # Prepare features
        X, feature_names = self.prepare_features(df)
        
        # Drop rows with missing values
        original_count = len(df)
        missing_mask = X.isna().any(axis=1)
        n_missing = missing_mask.sum()
        
        if n_missing > 0:
            self.logger.info(f"Removing {n_missing:,} rows with missing values ({n_missing/original_count*100:.2f}%)")
            df = df[~missing_mask].copy()
            X = X[~missing_mask].copy()
        
        self.report.total_compounds = original_count
        self.report.compounds_after_missing_removal = len(df)
        
        # Find existing domains
        existing_domains = [d for d in self.TARGET_DOMAINS if d in df.columns]
        if not existing_domains:
            raise ValueError(f"No target domains found. Expected: {self.TARGET_DOMAINS}")
        
        self.report.domains_analyzed = existing_domains
        self.report.domain_counts = {d: int(df[d].sum()) for d in existing_domains}
        
        self.logger.info(f"Analyzing {len(existing_domains)} domains: {existing_domains}")
        
        # PCA
        analysis_config = self.config.get("domain_analysis", {})
        pca_config = analysis_config.get("pca_analysis", {})
        if pca_config.get("enabled", True):
            self.logger.info("\n" + "=" * 60)
            self.logger.info("PERFORMING PCA ANALYSIS")
            self.logger.info("=" * 60)
            self.pca_result = self.perform_pca(X, pca_config.get("n_components", 5))
            if self.pca_result:
                self.report.pca_results = {
                    "n_components": self.pca_result.n_components,
                    "explained_variance_ratio": self.pca_result.explained_variance_ratio,
                    "cumulative_variance": self.pca_result.cumulative_variance
                }
        
        # Similarity
        sim_config = analysis_config.get("similarity_analysis", {})
        if sim_config.get("enabled", True):
            self.logger.info("\n" + "=" * 60)
            self.logger.info("PERFORMING SIMILARITY ANALYSIS")
            self.logger.info("=" * 60)
            for domain in tqdm(existing_domains, desc="Similarity analysis"):
                sim_result = self.compute_similarity(X, df, domain, sim_config.get("max_samples", 5000))
                self.report.similarity_results[domain] = sim_result
                if sim_result.get("n_samples", 0) > 0:
                    self.logger.info(f"  {domain}: Cosine sim = {sim_result['cosine_mean']:.3f} ± {sim_result['cosine_std']:.3f}")
        
        # Train models
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING XGBOOST MODELS")
        self.logger.info("=" * 60)
        
        for domain in tqdm(existing_domains, desc="Training models"):
            y = df[domain]
            class_counts = y.value_counts()
            self.logger.info(f"\n{domain} class distribution: {dict(class_counts)}")
            
            if len(class_counts) < 2 or class_counts.min() < 5:
                self.logger.warning(f"Skipping {domain}: insufficient samples")
                continue
            
            preprocessor = self.create_preprocessor(feature_names)
            result = self.train_model(X, y, domain, preprocessor)
            self.results[domain] = result
            
            self.report.models[domain] = {
                "metrics": result.metrics,
                "best_params": result.best_params,
                "feature_importances": result.feature_importances,
                "extracted_ranges": result.extracted_ranges
            }
        
        self.logger.info("\nDomain analysis complete")
        return self.results
    
    def generate_visualizations(self, df: pd.DataFrame, X: pd.DataFrame, output_dir: Path):
        """Generate all visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Generating visualizations...")
        
        for domain, result in tqdm(self.results.items(), desc="Generating plots"):
            self.logger.info(f"  Generating plots for {domain}...")
            
            # Feature importance
            self._plot_feature_importance(result, output_dir)
            
            # Confusion matrix (normalized and annotated)
            self._plot_confusion_matrix(result, output_dir)
            
            # ROC curve
            self._plot_roc_curve(result, output_dir)
            
            # Decision tree
            self._plot_decision_tree(result, output_dir)
            
            # Save rules
            self._save_rules(result, output_dir)
            
            # NEW: Threshold sweep analysis
            self._plot_threshold_sweep(result, output_dir)
            
            # NEW: Calibration curve
            self._plot_calibration_curve(result, output_dir)
            
            # NEW: False positive analysis
            self._analyze_false_positives(result, df, output_dir)
            
            # NEW: Feature attribution on FP cases
            self._plot_fp_feature_attribution(result, output_dir)
        
        # PCA plots
        if self.pca_result:
            self._plot_pca_variance(output_dir)
            self._plot_pca_loadings(output_dir)
            for domain in self.results.keys():
                self._plot_pca_scatter(df, domain, output_dir)
    
    def _plot_feature_importance(self, result: ModelResult, output_dir: Path):
        """Plot feature importance."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = [format_label(f) for f in result.feature_importances.keys()][:3]
        importances = list(result.feature_importances.values())[:3]
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontdict={'size': 40})
        
        ax.set_xlabel('Importance', fontdict={'size': 40})
        domain_display = format_label(result.domain_name)
        #ax.set_title(f'{domain_display}', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"feature_importance_{result.domain_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, result: ModelResult, output_dir: Path):
        """Plot confusion matrix with detailed annotations."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        cm = confusion_matrix(result.y_true, result.y_pred)
        
        # Calculate per-class metrics
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        # Create annotation labels with counts in scientific notation and percentages
        labels = np.array([
            [f'TN\n{tn:.2e}\n({tn/total*100:.1f}%)', f'FP\n{fp:.2e}\n({fp/total*100:.1f}%)'],
            [f'FN\n{fn:.2e}\n({fn/total*100:.1f}%)', f'TP\n{tp:.2e}\n({tp/total*100:.1f}%)']
        ])
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Predicted: No', 'Predicted: Yes'],
                yticklabels=['Actual: No', 'Actual: Yes'],
                annot_kws={'size': 38},
                vmin=0, vmax=250000)
        
        ax.set_xlabel('Predicted Label', fontweight='bold', fontdict={'size': 32})
        ax.set_ylabel('Actual Label', fontweight='bold', fontdict={'size': 32})
        domain_name_display = format_label(result.domain_name)
        ax.set_title(f'{domain_name_display}', fontweight='bold', fontdict={'size': 36})
        
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_{result.domain_name}.png", dpi=600, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, result: ModelResult, output_dir: Path):
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(result.y_true, result.y_proba[:, 1])
        auc = result.metrics.get("roc_auc", 0)
        
        ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        domain_display = format_label(result.domain_name)
        ax.set_title(f'{domain_display}', fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"roc_curve_{result.domain_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_decision_tree(self, result: ModelResult, output_dir: Path):
        """Plot XGBoost feature importance as a substitute for tree visualization.
        
        Note: XGBoost's plot_tree requires Graphviz which has dependency issues on Windows.
        Instead, we plot a detailed feature importance chart.
        """
        xgb_model = result.model.named_steps['clf']
        domain_name_display = format_label(result.domain_name)
        
        # Create a more detailed feature importance visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Gain-based importance
        importance_gain = xgb_model.get_booster().get_score(importance_type='gain')
        if importance_gain:
            features_gain = list(importance_gain.keys())
            values_gain = list(importance_gain.values())
            # Map f0, f1, etc. to actual feature names
            feature_map = {f'f{i}': name for i, name in enumerate(result.features_used)}
            features_gain = [format_label(feature_map.get(f, f)) for f in features_gain]
            
            y_pos = np.arange(len(features_gain))
            axes[0].barh(y_pos, values_gain, color='steelblue', edgecolor='black', linewidth=0.5)
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(features_gain)
            axes[0].invert_yaxis()
            axes[0].set_xlabel('Gain', fontweight='bold')
            axes[0].set_title(f'Feature Importance (Gain): {domain_name_display}', fontweight='bold')
        
        # Plot 2: Weight-based importance (number of times feature appears in trees)
        importance_weight = xgb_model.get_booster().get_score(importance_type='weight')
        if importance_weight:
            features_weight = list(importance_weight.keys())
            values_weight = list(importance_weight.values())
            features_weight = [format_label(feature_map.get(f, f)) for f in features_weight]
            
            y_pos = np.arange(len(features_weight))
            axes[1].barh(y_pos, values_weight, color='coral', edgecolor='black', linewidth=0.5)
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(features_weight)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Weight (Split Count)', fontweight='bold')
            axes[1].set_title(f'Feature Importance (Weight): {domain_name_display}', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"xgboost_importance_{result.domain_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save the tree structure as text
        try:
            tree_dump = xgb_model.get_booster().get_dump(dump_format='text')
            if tree_dump:
                with open(output_dir / f"tree_structure_{result.domain_name}.txt", 'w') as f:
                    f.write(f"XGBoost Tree Structure for {domain_name_display}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Number of trees: {len(tree_dump)}\n\n")
                    f.write("First Tree Structure:\n")
                    f.write("-" * 40 + "\n")
                    f.write(tree_dump[0])
        except Exception as e:
            self.logger.warning(f"Could not save tree structure: {e}")
        
        # Plot actual decision trees using XGBoost's plot_tree
        self._plot_xgboost_trees(result, output_dir)
    
    def _plot_xgboost_trees(self, result: ModelResult, output_dir: Path):
        """Plot the best XGBoost decision tree using custom matplotlib visualization.
        
        The first tree (tree 0) is typically the most informative as it captures
        the largest signal before subsequent trees fit the residuals.
        """
        import re
        
        xgb_model = result.model.named_steps['clf']
        domain_name_display = format_label(result.domain_name)
        
        # Create feature name mapping for display
        feature_map = {f'f{i}': format_label(name) for i, name in enumerate(result.features_used)}
        
        # Get tree dumps in text format
        try:
            tree_dumps = xgb_model.get_booster().get_dump(dump_format='text')
        except Exception as e:
            self.logger.warning(f"Could not get tree dumps for {result.domain_name}: {e}")
            return
        
        # Plot only the first (best) tree - it captures the most significant patterns
        if len(tree_dumps) == 0:
            self.logger.warning(f"No trees found for {result.domain_name}")
            return
        
        try:
            tree_text = tree_dumps[0]
            
            # Parse the tree structure
            nodes = self._parse_tree_text(tree_text, feature_map)
            
            if nodes:
                # Create visualization
                fig, ax = plt.subplots(figsize=(20, 14))
                self._draw_tree(ax, nodes, domain_name_display, tree_num=None)
                
                plt.tight_layout()
                plt.savefig(output_dir / f"decision_tree_{result.domain_name}.png", 
                           dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                self.logger.info(f"  Saved best decision tree for {result.domain_name}")
                
        except Exception as e:
            self.logger.warning(f"Could not plot tree for {result.domain_name}: {e}")
    
    def _parse_tree_text(self, tree_text: str, feature_map: dict) -> list:
        """Parse XGBoost tree text dump into node structure."""
        import re
        
        nodes = []
        lines = tree_text.strip().split('\n')
        
        for line in lines:
            # Match internal node: "0:[f0<0.5] yes=1,no=2,missing=1"
            internal_match = re.match(r'\s*(\d+):\[(f\d+)<([^\]]+)\]\s*yes=(\d+),no=(\d+)', line)
            # Match leaf node: "1:leaf=0.5"
            leaf_match = re.match(r'\s*(\d+):leaf=([+-]?\d*\.?\d+)', line)
            
            if internal_match:
                node_id = int(internal_match.group(1))
                feature = internal_match.group(2)
                threshold = float(internal_match.group(3))
                yes_child = int(internal_match.group(4))
                no_child = int(internal_match.group(5))
                
                feature_name = feature_map.get(feature, feature)
                
                nodes.append({
                    'id': node_id,
                    'type': 'internal',
                    'feature': feature_name,
                    'threshold': threshold,
                    'yes': yes_child,
                    'no': no_child,
                    'depth': line.count('\t')
                })
            elif leaf_match:
                node_id = int(leaf_match.group(1))
                value = float(leaf_match.group(2))
                
                nodes.append({
                    'id': node_id,
                    'type': 'leaf',
                    'value': value,
                    'depth': line.count('\t')
                })
        
        return nodes
    
    def _draw_tree(self, ax: plt.Axes, nodes: list, domain_name: str, tree_num: int):
        """Draw a publication-quality tree structure using matplotlib patches."""
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
        from matplotlib.collections import PatchCollection
        import matplotlib.patheffects as path_effects
        
        if not nodes:
            return
        
        # Build node lookup and determine positions
        node_lookup = {n['id']: n for n in nodes}
        
        # Calculate tree depth
        max_depth = max(n['depth'] for n in nodes) + 1
        
        # Assign positions using recursive layout
        positions = {}
        self._assign_positions(nodes, node_lookup, 0, 0, 1, 0, positions)
        
        # Normalize positions with better spacing
        if positions:
            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            x_range = max_x - min_x if max_x != min_x else 1
            y_range = max_y - min_y if max_y != min_y else 1
            
            # Scale to use more of the figure space
            margin = 0.08
            for node_id in positions:
                x, y = positions[node_id]
                positions[node_id] = (
                    margin + (1 - 2*margin) * (x - min_x) / x_range,
                    margin + (1 - 2*margin) * (1 - (y - min_y) / y_range)
                )
        
        # Publication-quality color scheme
        colors = {
            'split_fill': '#4A90D9',      # Professional blue for split nodes
            'split_edge': '#2C5282',       # Darker blue edge
            'leaf_pos_fill': '#48BB78',    # Green for positive leaves
            'leaf_pos_edge': '#276749',    # Darker green edge
            'leaf_neg_fill': '#F56565',    # Red for negative leaves
            'leaf_neg_edge': '#C53030',    # Darker red edge
            'yes_line': '#38A169',          # Green for yes branch
            'no_line': '#E53E3E',           # Red for no branch
            'text': '#1A202C',              # Dark text
        }
        
        # Node dimensions
        node_width = 0.12
        node_height = 0.06
        leaf_radius = 0.035
        
        # Draw edges with curved arrows
        for node in nodes:
            if node['type'] == 'internal':
                node_id = node['id']
                if node_id in positions:
                    x, y = positions[node_id]
                    
                    # Draw to yes child (left branch)
                    yes_id = node['yes']
                    if yes_id in positions:
                        yes_x, yes_y = positions[yes_id]
                        # Draw curved line
                        ax.annotate('', xy=(yes_x, yes_y + node_height/2 + 0.01), 
                                   xytext=(x, y - node_height/2 - 0.01),
                                   arrowprops=dict(arrowstyle='->', color=colors['yes_line'], 
                                                  lw=3, connectionstyle='arc3,rad=0.1'))
                        # Add "Yes" label with white outline for readability
                        mid_x = (x + yes_x) / 2 - 0.03
                        mid_y = (y + yes_y) / 2
                        txt = ax.text(mid_x, mid_y, 'Yes', fontsize=14, color=colors['yes_line'], 
                                     fontweight='bold', fontfamily='Arial', ha='center', va='center')
                        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
                    
                    # Draw to no child (right branch)
                    no_id = node['no']
                    if no_id in positions:
                        no_x, no_y = positions[no_id]
                        ax.annotate('', xy=(no_x, no_y + node_height/2 + 0.01), 
                                   xytext=(x, y - node_height/2 - 0.01),
                                   arrowprops=dict(arrowstyle='->', color=colors['no_line'], 
                                                  lw=3, connectionstyle='arc3,rad=-0.1'))
                        mid_x = (x + no_x) / 2 + 0.03
                        mid_y = (y + no_y) / 2
                        txt = ax.text(mid_x, mid_y, 'No', fontsize=14, color=colors['no_line'], 
                                     fontweight='bold', fontfamily='Arial', ha='center', va='center')
                        txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
        
        # Draw nodes on top
        for node in nodes:
            node_id = node['id']
            if node_id not in positions:
                continue
                
            x, y = positions[node_id]
            
            if node['type'] == 'internal':
                # Split node - rounded rectangle with gradient-like appearance
                rect = FancyBboxPatch(
                    (x - node_width/2, y - node_height/2), 
                    node_width, node_height,
                    boxstyle='round,pad=0.02,rounding_size=0.02',
                    facecolor=colors['split_fill'],
                    edgecolor=colors['split_edge'],
                    linewidth=3,
                    zorder=10
                )
                ax.add_patch(rect)
                
                # Add shadow effect
                shadow = FancyBboxPatch(
                    (x - node_width/2 + 0.003, y - node_height/2 - 0.003), 
                    node_width, node_height,
                    boxstyle='round,pad=0.02,rounding_size=0.02',
                    facecolor='gray',
                    alpha=0.3,
                    zorder=9
                )
                ax.add_patch(shadow)
                
                # Node text
                feature_text = node['feature']
                threshold_text = f"< {node['threshold']:.2f}"
                ax.text(x, y + 0.012, feature_text, ha='center', va='center', 
                       fontsize=13, fontweight='bold', color='white', fontfamily='Arial', zorder=11)
                ax.text(x, y - 0.015, threshold_text, ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white', fontfamily='Arial', zorder=11)
            else:
                # Leaf node - circle with value
                value = node['value']
                if value > 0:
                    fill_color = colors['leaf_pos_fill']
                    edge_color = colors['leaf_pos_edge']
                    label = 'Positive'
                else:
                    fill_color = colors['leaf_neg_fill']
                    edge_color = colors['leaf_neg_edge']
                    label = 'Negative'
                
                # Draw shadow
                shadow = Circle((x + 0.003, y - 0.003), leaf_radius, 
                               facecolor='gray', alpha=0.3, zorder=9)
                ax.add_patch(shadow)
                
                # Draw circle
                circle = Circle((x, y), leaf_radius, 
                               facecolor=fill_color, edgecolor=edge_color, 
                               linewidth=3, zorder=10)
                ax.add_patch(circle)
                
                # Value text
                ax.text(x, y, f'{value:.2f}', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white', fontfamily='Arial', zorder=11)
        
        # Set axis properties
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        title = f'XGBoost Decision Tree: {domain_name}'
        if tree_num:
            title = f'XGBoost Decision Tree {tree_num}: {domain_name}'
        ax.set_title(title, fontweight='bold', fontsize=24, pad=25, fontfamily='Arial')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['split_fill'], 
                   markersize=15, label='Split Node', markeredgecolor=colors['split_edge'], markeredgewidth=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['leaf_pos_fill'], 
                   markersize=12, label='Positive Leaf', markeredgecolor=colors['leaf_pos_edge'], markeredgewidth=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['leaf_neg_fill'], 
                   markersize=12, label='Negative Leaf', markeredgecolor=colors['leaf_neg_edge'], markeredgewidth=2),
            Line2D([0], [0], color=colors['yes_line'], lw=3, label='Yes Branch'),
            Line2D([0], [0], color=colors['no_line'], lw=3, label='No Branch'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
                 framealpha=0.95, edgecolor='gray', fancybox=True)
    
    def _assign_positions(self, nodes: list, node_lookup: dict, node_id: int, 
                          x: float, width: float, depth: int, positions: dict):
        """Recursively assign x,y positions to nodes."""
        if node_id not in node_lookup:
            return
        
        node = node_lookup[node_id]
        positions[node_id] = (x + width / 2, depth)
        
        if node['type'] == 'internal':
            # Assign children
            yes_id = node['yes']
            no_id = node['no']
            
            half_width = width / 2
            self._assign_positions(nodes, node_lookup, yes_id, x, half_width, depth + 1, positions)
            self._assign_positions(nodes, node_lookup, no_id, x + half_width, half_width, depth + 1, positions)

    def _save_rules(self, result: ModelResult, output_dir: Path):
        """Save XGBoost model summary to text file."""
        with open(output_dir / f"rules_{result.domain_name}.txt", 'w') as f:
            f.write(f"XGBoost Model Summary for {result.domain_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Parameters: {result.best_params}\n\n")
            f.write(result.rules_text)
            f.write("\n\nFeature Ranges (from training data):\n")
            for feature, ranges in result.extracted_ranges.items():
                f.write(f"  {feature}: [{ranges['min']:.3f}, {ranges['max']:.3f}] ")
                f.write(f"(mean: {ranges.get('mean', 0):.3f}, std: {ranges.get('std', 0):.3f})\n")
    
    def _plot_threshold_sweep(self, result: ModelResult, output_dir: Path):
        """Sweep decision thresholds and observe FP/FN/precision/recall behavior."""
        domain_name_display = format_label(result.domain_name)
        
        # Get probabilities for positive class
        y_proba_pos = result.y_proba[:, 1]
        y_true = result.y_true
        
        # Sweep thresholds
        thresholds = np.linspace(0.05, 0.95, 50)
        metrics_by_threshold = {
            'threshold': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'fp_count': [],
            'fn_count': [],
            'fp_rate': [],
            'tn_rate': []
        }
        
        for thresh in thresholds:
            y_pred_thresh = (y_proba_pos >= thresh).astype(int)
            
            tn = ((y_true == 0) & (y_pred_thresh == 0)).sum()
            fp = ((y_true == 0) & (y_pred_thresh == 1)).sum()
            fn = ((y_true == 1) & (y_pred_thresh == 0)).sum()
            tp = ((y_true == 1) & (y_pred_thresh == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics_by_threshold['threshold'].append(thresh)
            metrics_by_threshold['precision'].append(precision)
            metrics_by_threshold['recall'].append(recall)
            metrics_by_threshold['f1'].append(f1)
            metrics_by_threshold['fp_count'].append(fp)
            metrics_by_threshold['fn_count'].append(fn)
            metrics_by_threshold['fp_rate'].append(fp_rate)
            metrics_by_threshold['tn_rate'].append(tn_rate)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Precision, Recall, F1 vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(metrics_by_threshold['threshold'], metrics_by_threshold['precision'], 
                 'b-', label='Precision', linewidth=2.5)
        ax1.plot(metrics_by_threshold['threshold'], metrics_by_threshold['recall'], 
                 'r-', label='Recall', linewidth=2.5)
        ax1.plot(metrics_by_threshold['threshold'], metrics_by_threshold['f1'], 
                 'g-', label='F1 Score', linewidth=2.5)
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
        ax1.set_xlabel('Decision Threshold', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title(f'Precision/Recall/F1 vs Threshold: {domain_name_display}', fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Plot 2: FP and FN counts vs Threshold
        ax2 = axes[0, 1]
        ax2.plot(metrics_by_threshold['threshold'], metrics_by_threshold['fp_count'], 
                 'r-', label='False Positives', linewidth=2.5)
        ax2.plot(metrics_by_threshold['threshold'], metrics_by_threshold['fn_count'], 
                 'b-', label='False Negatives', linewidth=2.5)
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Decision Threshold', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title(f'FP/FN Counts vs Threshold: {domain_name_display}', fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Plot 3: FP Rate vs Threshold
        ax3 = axes[1, 0]
        ax3.plot(metrics_by_threshold['threshold'], metrics_by_threshold['fp_rate'], 
                 'r-', linewidth=2.5)
        ax3.fill_between(metrics_by_threshold['threshold'], metrics_by_threshold['fp_rate'], 
                         alpha=0.3, color='red')
        ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Decision Threshold', fontweight='bold')
        ax3.set_ylabel('False Positive Rate', fontweight='bold')
        ax3.set_title(f'FP Rate vs Threshold: {domain_name_display}', fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Plot 4: Find optimal thresholds
        ax4 = axes[1, 1]
        
        # Find optimal points
        best_f1_idx = np.argmax(metrics_by_threshold['f1'])
        best_f1_thresh = metrics_by_threshold['threshold'][best_f1_idx]
        
        # Find threshold where FP rate < 0.1
        low_fp_thresh = None
        for i, fpr in enumerate(metrics_by_threshold['fp_rate']):
            if fpr < 0.1:
                low_fp_thresh = metrics_by_threshold['threshold'][i]
                break
        
        # Trade-off visualization
        scatter = ax4.scatter(metrics_by_threshold['fp_rate'], metrics_by_threshold['recall'], 
                   c=metrics_by_threshold['threshold'], cmap='viridis', s=60, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('False Positive Rate', fontweight='bold')
        ax4.set_ylabel('Recall (True Positive Rate)', fontweight='bold')
        ax4.set_title(f'FPR vs Recall Trade-off: {domain_name_display}', fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Threshold', fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add text annotations
        textstr = f'Best F1 Threshold: {best_f1_thresh:.2f}\n'
        textstr += f'Best F1 Score: {metrics_by_threshold["f1"][best_f1_idx]:.3f}\n'
        if low_fp_thresh:
            textstr += f'Low FP (<10%) Threshold: {low_fp_thresh:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax4.text(0.95, 0.05, textstr, transform=ax4.transAxes, fontsize=11,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props, fontfamily='Arial')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"threshold_sweep_{result.domain_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save threshold analysis to text file
        with open(output_dir / f"threshold_analysis_{result.domain_name}.txt", 'w') as f:
            f.write(f"Threshold Analysis for {domain_name_display}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Best F1 Threshold: {best_f1_thresh:.3f}\n")
            f.write(f"  F1 at best threshold: {metrics_by_threshold['f1'][best_f1_idx]:.3f}\n")
            f.write(f"  Precision at best threshold: {metrics_by_threshold['precision'][best_f1_idx]:.3f}\n")
            f.write(f"  Recall at best threshold: {metrics_by_threshold['recall'][best_f1_idx]:.3f}\n")
            f.write(f"  FP count at best threshold: {metrics_by_threshold['fp_count'][best_f1_idx]}\n")
            f.write(f"  FN count at best threshold: {metrics_by_threshold['fn_count'][best_f1_idx]}\n\n")
            
            f.write("Default Threshold (0.5):\n")
            idx_05 = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 0.5))
            f.write(f"  F1: {metrics_by_threshold['f1'][idx_05]:.3f}\n")
            f.write(f"  Precision: {metrics_by_threshold['precision'][idx_05]:.3f}\n")
            f.write(f"  Recall: {metrics_by_threshold['recall'][idx_05]:.3f}\n")
            f.write(f"  FP count: {metrics_by_threshold['fp_count'][idx_05]}\n")
            f.write(f"  FN count: {metrics_by_threshold['fn_count'][idx_05]}\n")
    
    def _plot_calibration_curve(self, result: ModelResult, output_dir: Path):
        """Plot calibration curve to check if predicted probabilities are well-calibrated."""
        domain_name_display = format_label(result.domain_name)
        
        y_true = result.y_true
        y_proba_pos = result.y_proba[:, 1]
        
        # Compute calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_proba_pos, n_bins=10, strategy='uniform')
        except:
            self.logger.warning(f"Could not compute calibration curve for {result.domain_name}")
            return
        
        # Compute Brier score
        brier = brier_score_loss(y_true, y_proba_pos)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Calibration curve
        ax1 = axes[0]
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfectly Calibrated')
        ax1.plot(prob_pred, prob_true, 'bo-', linewidth=2, markersize=8, label=f'Model (Brier={brier:.3f})')
        ax1.set_xlabel('Mean Predicted Probability', fontweight='bold')
        ax1.set_ylabel('Fraction of Positives', fontweight='bold')
        ax1.set_title(f'Calibration Curve: {domain_name_display}', fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Plot 2: Histogram of predicted probabilities
        ax2 = axes[1]
        ax2.hist(y_proba_pos[y_true == 0], bins=20, alpha=0.6, label='Negative Class', color='blue', edgecolor='black', linewidth=0.5)
        ax2.hist(y_proba_pos[y_true == 1], bins=20, alpha=0.6, label='Positive Class', color='red', edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Predicted Probability', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title(f'Probability Distribution by Class: {domain_name_display}', fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"calibration_{result.domain_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add calibration info to metrics
        result.metrics['brier_score'] = float(brier)
    
    def _analyze_false_positives(self, result: ModelResult, df: pd.DataFrame, output_dir: Path):
        """Slice false positives by meaningful subgroups to understand FP behavior."""
        domain_name_display = format_label(result.domain_name)
        
        # Identify FP cases
        fp_mask = (result.y_true == 0) & (result.y_pred == 1)
        tn_mask = (result.y_true == 0) & (result.y_pred == 0)
        
        fp_count = fp_mask.sum()
        if fp_count == 0:
            self.logger.info(f"No false positives for {result.domain_name}")
            return
        
        # Get X_test for FP and TN cases
        X_test = result.X_test
        X_fp = X_test[fp_mask]
        X_tn = X_test[tn_mask]
        
        # Compare FP vs TN feature distributions
        feature_comparisons = {}
        for feature in result.features_used:
            if feature in X_fp.columns:
                fp_values = X_fp[feature].dropna()
                tn_values = X_tn[feature].dropna()
                
                if len(fp_values) > 0 and len(tn_values) > 0:
                    feature_comparisons[feature] = {
                        'fp_mean': float(fp_values.mean()),
                        'fp_std': float(fp_values.std()),
                        'tn_mean': float(tn_values.mean()),
                        'tn_std': float(tn_values.std()),
                        'diff_means': float(fp_values.mean() - tn_values.mean())
                    }
        
        # Sort by difference in means
        sorted_features = sorted(feature_comparisons.items(), 
                                 key=lambda x: abs(x[1]['diff_means']), reverse=True)
        
        # Visualize top features that differ between FP and TN
        top_n = min(6, len(sorted_features))
        if top_n > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (feature, stats) in enumerate(sorted_features[:top_n]):
                ax = axes[i]
                
                fp_data = X_fp[feature].dropna()
                tn_data = X_tn[feature].dropna()
                
                ax.hist(tn_data, bins=20, alpha=0.6, label=f'TN (n={len(tn_data)})', color='green', edgecolor='black', linewidth=0.5)
                ax.hist(fp_data, bins=20, alpha=0.6, label=f'FP (n={len(fp_data)})', color='red', edgecolor='black', linewidth=0.5)
                ax.set_xlabel(format_label(feature), fontweight='bold')
                ax.set_ylabel('Count', fontweight='bold')
                ax.set_title(f'{format_label(feature)}\n\u0394mean={stats["diff_means"]:.3f}', fontweight='bold')
                ax.legend(loc='best', fontsize=10, framealpha=0.9)
            
            # Hide unused subplots
            for j in range(top_n, len(axes)):
                axes[j].set_visible(False)
            
            plt.suptitle(f'FP vs TN Feature Distributions: {domain_name_display}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f"fp_analysis_{result.domain_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save FP analysis report
        with open(output_dir / f"fp_analysis_{result.domain_name}.txt", 'w', encoding='utf-8') as f:
            f.write(f"False Positive Analysis for {domain_name_display}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total False Positives: {fp_count}\n")
            f.write(f"Total True Negatives: {tn_mask.sum()}\n")
            f.write(f"FP Rate: {fp_count / (fp_count + tn_mask.sum()) * 100:.2f}%\n\n")
            
            f.write("Features that distinguish FP from TN (sorted by |diff_mean|):\n")
            f.write("-" * 60 + "\n")
            for feature, stats in sorted_features[:10]:
                f.write(f"\n{feature}:\n")
                f.write(f"  FP: mean={stats['fp_mean']:.3f}, std={stats['fp_std']:.3f}\n")
                f.write(f"  TN: mean={stats['tn_mean']:.3f}, std={stats['tn_std']:.3f}\n")
                f.write(f"  Difference: {stats['diff_means']:.3f}\n")
    
    def _plot_fp_feature_attribution(self, result: ModelResult, output_dir: Path):
        """Apply feature attribution analysis on FP cases only."""
        domain_name_display = format_label(result.domain_name)
        
        # Identify FP cases
        fp_mask = (result.y_true == 0) & (result.y_pred == 1)
        fp_count = fp_mask.sum()
        
        if fp_count == 0:
            return
        
        X_test = result.X_test
        X_fp = X_test[fp_mask]
        
        # Get the XGBoost model
        xgb_model = result.model.named_steps['clf']
        
        # Get preprocessed features for FP cases
        preprocessor = result.model.named_steps['preprocessor']
        X_fp_transformed = preprocessor.transform(X_fp)
        
        # Get feature contributions using XGBoost's predict with output_margin
        # We'll use feature importances weighted by the FP feature values
        feature_names = result.features_used
        
        # Calculate mean feature values for FP cases (normalized)
        fp_feature_means = X_fp.mean()
        
        # Weight by feature importance to get "contribution"
        feature_contributions = {}
        for feature in feature_names:
            if feature in result.feature_importances and feature in fp_feature_means.index:
                importance = result.feature_importances.get(feature, 0)
                mean_val = fp_feature_means[feature]
                # Contribution = importance * normalized deviation from overall mean
                overall_mean = X_test[feature].mean() if feature in X_test.columns else 0
                overall_std = X_test[feature].std() if feature in X_test.columns else 1
                if overall_std > 0:
                    z_score = (mean_val - overall_mean) / overall_std
                    feature_contributions[feature] = importance * z_score
                else:
                    feature_contributions[feature] = 0
        
        # Sort by absolute contribution
        sorted_contributions = sorted(feature_contributions.items(), 
                                      key=lambda x: abs(x[1]), reverse=True)
        
        # Visualize
        top_n = min(10, len(sorted_contributions))
        if top_n > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            features = [format_label(x[0]) for x in sorted_contributions[:top_n]]
            contributions = [x[1] for x in sorted_contributions[:top_n]]
            colors = ['red' if c > 0 else 'blue' for c in contributions]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, contributions, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            ax.set_xlabel('Contribution (Importance × Z-Score)', fontweight='bold')
            ax.set_title(f'Feature Attribution for False Positives: {domain_name_display}', fontweight='bold')
            ax.axvline(x=0, color='black', linewidth=1)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', edgecolor='black', label='Pushes Toward FP (Higher in FP)'),
                Patch(facecolor='blue', edgecolor='black', label='Pushes Toward TN (Lower in FP)')
            ]
            ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"fp_attribution_{result.domain_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_pca_variance(self, output_dir: Path):
        """Plot PCA variance explained."""
        if not self.pca_result:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n = self.pca_result.n_components
        x = range(1, n + 1)
        
        ax.bar(x, self.pca_result.explained_variance_ratio, alpha=0.7, label='Individual', color='steelblue', edgecolor='black', linewidth=0.5)
        ax.plot(x, self.pca_result.cumulative_variance, 'ro-', linewidth=2, markersize=8, label='Cumulative')
        ax.axhline(y=0.8, color='g', linestyle='--', linewidth=1.5, alpha=0.7, label='80% Threshold')
        ax.set_xlabel('Principal Component', fontweight='bold')
        ax.set_ylabel('Explained Variance Ratio', fontweight='bold')
        ax.set_title('Variance Explained', fontweight='bold')
        ax.legend(framealpha=0.9)
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / "pca_variance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_loadings(self, output_dir: Path):
        """Plot PCA loadings heatmap."""
        if not self.pca_result:
            return
        
        # Format feature names for display
        formatted_loadings = {format_label(k): v for k, v in self.pca_result.loadings.items()}
        loadings_df = pd.DataFrame(formatted_loadings).T
        loadings_df.columns = [f'PC{i+1}' for i in range(loadings_df.shape[1])]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
                    annot_kws={'size': 11, 'fontweight': 'bold'})
        ax.set_title('PCA Loadings', fontweight='bold')
        ax.set_xlabel('Principal Component', fontweight='bold')
        ax.set_ylabel('Feature', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "pca_loadings.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_scatter(self, df: pd.DataFrame, domain: str, output_dir: Path):
        """Plot PCA scatter colored by domain."""
        if not self.pca_result or self.pca_result.coordinates is None:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get domain labels for clean data
        X_clean = df[list(self.pca_result.loadings.keys())].dropna()
        domain_labels = df.loc[X_clean.index, domain].values
        
        # Sort so True values plot on top
        sort_idx = np.argsort(domain_labels)
        coords = self.pca_result.coordinates[sort_idx]
        labels = domain_labels[sort_idx]
        
        colors = ['#E8E8E8' if l == 0 else "#b41f1f" for l in labels]
        
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.5, s=15, edgecolors='none')
        ax.set_xlabel(f'PC1 ({self.pca_result.explained_variance_ratio[0]*100:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'PC2 ({self.pca_result.explained_variance_ratio[1]*100:.1f}%)', fontweight='bold')
        domain_display = format_label(domain)
        ax.set_title(f'{domain_display}', fontweight='bold')

        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E8E8E8', edgecolor='black', label='False'),
            Patch(facecolor='#1f77b4', edgecolor='black', label='True')
        ]
        #ax.legend(handles=legend_elements, title=domain_display, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"pca_scatter_{domain}.png", dpi=600, bbox_inches='tight')
        plt.close()

    def generate_full_report_text(self, df: pd.DataFrame, output_dir: Path):
        """Generate a comprehensive text file with all analysis data."""
        output_file = output_dir / "full_analysis_report.txt"
        self.logger.info(f"Generating comprehensive text report: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("PHORCE DOMAIN-LEVEL XGBOOST ANALYSIS - FULL REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {self.report.timestamp}\n")
            f.write(f"Input File: {self.report.input_file}\n")
            f.write(f"Output Directory: {self.report.output_dir}\n\n")
            
            # Dataset Overview
            f.write("-" * 80 + "\n")
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Total Compounds: {self.report.total_compounds:,}\n")
            f.write(f"Compounds After Missing Value Removal: {self.report.compounds_after_missing_removal:,}\n")
            f.write(f"Compounds Removed: {self.report.total_compounds - self.report.compounds_after_missing_removal:,}\n")
            f.write(f"Removal Percentage: {(self.report.total_compounds - self.report.compounds_after_missing_removal) / self.report.total_compounds * 100:.2f}%\n\n")
            
            f.write("Domains Analyzed:\n")
            for domain in self.report.domains_analyzed:
                count = self.report.domain_counts.get(domain, 0)
                pct = count / self.report.compounds_after_missing_removal * 100 if self.report.compounds_after_missing_removal > 0 else 0
                f.write(f"  - {domain}: {count:,} compounds ({pct:.2f}%)\n")
            f.write("\n")
            
            # PCA Results
            if self.report.pca_results:
                f.write("-" * 80 + "\n")
                f.write("PCA ANALYSIS RESULTS\n")
                f.write("-" * 80 + "\n\n")
                f.write(f"Number of Components: {self.report.pca_results.get('n_components', 'N/A')}\n\n")
                f.write("Explained Variance Ratio:\n")
                evr = self.report.pca_results.get('explained_variance_ratio', [])
                cum_var = self.report.pca_results.get('cumulative_variance', [])
                for i, (var, cum) in enumerate(zip(evr, cum_var), 1):
                    f.write(f"  PC{i}: {var*100:.2f}% (Cumulative: {cum*100:.2f}%)\n")
                f.write("\n")
                
                if self.pca_result and self.pca_result.loadings:
                    f.write("PCA Loadings:\n")
                    f.write("-" * 60 + "\n")
                    header = "Feature".ljust(20)
                    for i in range(min(5, len(evr))):
                        header += f"PC{i+1}".rjust(10)
                    f.write(header + "\n")
                    f.write("-" * 60 + "\n")
                    for feature, loadings in self.pca_result.loadings.items():
                        row = feature.ljust(20)
                        for loading in loadings[:5]:
                            row += f"{loading:10.4f}"
                        f.write(row + "\n")
                    f.write("\n")
            
            # Similarity Analysis Results
            if self.report.similarity_results:
                f.write("-" * 80 + "\n")
                f.write("SIMILARITY ANALYSIS RESULTS\n")
                f.write("-" * 80 + "\n\n")
                for domain, sim_data in self.report.similarity_results.items():
                    f.write(f"{domain}:\n")
                    if sim_data.get('n_samples', 0) > 0:
                        f.write(f"  Samples: {sim_data['n_samples']:,}\n")
                        f.write(f"  Cosine Similarity: {sim_data['cosine_mean']:.4f} ± {sim_data['cosine_std']:.4f}\n")
                        f.write(f"  Euclidean Similarity: {sim_data['euclidean_mean']:.4f} ± {sim_data['euclidean_std']:.4f}\n")
                    else:
                        f.write("  Insufficient samples for analysis\n")
                    f.write("\n")
            
            # Model Results for Each Domain
            for domain, result in self.results.items():
                f.write("=" * 80 + "\n")
                f.write(f"MODEL RESULTS: {domain.upper().replace('_', ' ')}\n")
                f.write("=" * 80 + "\n\n")
                
                # Best Parameters
                f.write("Best Hyperparameters:\n")
                for param, value in result.best_params.items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")
                
                # Performance Metrics
                f.write("Performance Metrics:\n")
                f.write("-" * 40 + "\n")
                metrics = result.metrics
                f.write(f"  Accuracy:     {metrics.get('accuracy', 0):.4f}\n")
                f.write(f"  F1 Macro:     {metrics.get('f1_macro', 0):.4f}\n")
                f.write(f"  Precision:    {metrics.get('precision', 0):.4f}\n")
                f.write(f"  Recall:       {metrics.get('recall', 0):.4f}\n")
                f.write(f"  ROC-AUC:      {metrics.get('roc_auc', 0):.4f}\n")
                f.write(f"  Brier Score:  {metrics.get('brier_score', 'N/A')}\n")
                f.write(f"  CV Mean:      {metrics.get('cv_mean', 0):.4f}\n")
                f.write(f"  CV Std:       {metrics.get('cv_std', 0):.4f}\n")
                f.write("\n")
                
                # Confusion Matrix
                from sklearn.metrics import confusion_matrix as cm_func
                cm = cm_func(result.y_true, result.y_pred)
                tn, fp, fn, tp = cm.ravel()
                total = tn + fp + fn + tp
                
                f.write("Confusion Matrix:\n")
                f.write("-" * 40 + "\n")
                f.write(f"                    Predicted\n")
                f.write(f"                  No        Yes\n")
                f.write(f"  Actual No   {tn:6} (TN) {fp:6} (FP)\n")
                f.write(f"  Actual Yes  {fn:6} (FN) {tp:6} (TP)\n")
                f.write(f"\n")
                f.write(f"  True Negative Rate (Specificity): {tn/(tn+fp)*100:.2f}%\n")
                f.write(f"  True Positive Rate (Recall):      {tp/(tp+fn)*100:.2f}%\n")
                f.write(f"  False Positive Rate:              {fp/(fp+tn)*100:.2f}%\n")
                f.write(f"  False Negative Rate:              {fn/(fn+tp)*100:.2f}%\n")
                f.write("\n")
                
                # Feature Importances
                f.write("Feature Importances (Sorted):\n")
                f.write("-" * 40 + "\n")
                for feature, importance in result.feature_importances.items():
                    bar = "█" * int(importance * 50) if importance > 0 else ""
                    f.write(f"  {feature:15} {importance:.4f}  {bar}\n")
                f.write("\n")
                
                # Extracted Feature Ranges
                if result.extracted_ranges:
                    f.write("Feature Ranges (from training data):\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"{'Feature':15} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}\n")
                    f.write("-" * 60 + "\n")
                    for feature, ranges in result.extracted_ranges.items():
                        f.write(f"{feature:15} {ranges['min']:10.3f} {ranges['max']:10.3f} "
                               f"{ranges.get('mean', 0):10.3f} {ranges.get('std', 0):10.3f}\n")
                    f.write("\n")
                
                # XGBoost Model Summary
                f.write("XGBoost Model Summary:\n")
                f.write("-" * 40 + "\n")
                f.write(result.rules_text + "\n\n")
                
                # Threshold Analysis Summary
                f.write("Threshold Analysis Summary:\n")
                f.write("-" * 40 + "\n")
                y_proba_pos = result.y_proba[:, 1]
                y_true = result.y_true
                
                # Calculate metrics at different thresholds
                thresholds_to_show = [0.3, 0.4, 0.5, 0.6, 0.7]
                f.write(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>6} {'FN':>6}\n")
                f.write("-" * 60 + "\n")
                
                best_f1 = 0
                best_thresh = 0.5
                for thresh in thresholds_to_show:
                    y_pred_thresh = (y_proba_pos >= thresh).astype(int)
                    tn_t = ((y_true == 0) & (y_pred_thresh == 0)).sum()
                    fp_t = ((y_true == 0) & (y_pred_thresh == 1)).sum()
                    fn_t = ((y_true == 1) & (y_pred_thresh == 0)).sum()
                    tp_t = ((y_true == 1) & (y_pred_thresh == 1)).sum()
                    
                    prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
                    rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
                    f1_t = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    
                    if f1_t > best_f1:
                        best_f1 = f1_t
                        best_thresh = thresh
                    
                    f.write(f"{thresh:10.2f} {prec:10.4f} {rec:10.4f} {f1_t:10.4f} {fp_t:6} {fn_t:6}\n")
                
                f.write(f"\n  Best F1 Score: {best_f1:.4f} at threshold {best_thresh:.2f}\n\n")
                
                # False Positive Analysis Summary
                fp_mask = (result.y_true == 0) & (result.y_pred == 1)
                tn_mask = (result.y_true == 0) & (result.y_pred == 0)
                fp_count = fp_mask.sum()
                
                if fp_count > 0:
                    f.write("False Positive Analysis:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  Total False Positives: {fp_count}\n")
                    f.write(f"  Total True Negatives: {tn_mask.sum()}\n")
                    f.write(f"  FP Rate: {fp_count / (fp_count + tn_mask.sum()) * 100:.2f}%\n\n")
                    
                    X_test = result.X_test
                    X_fp = X_test[fp_mask]
                    X_tn = X_test[tn_mask]
                    
                    f.write("  Feature Comparison (FP vs TN):\n")
                    f.write(f"  {'Feature':15} {'FP Mean':>10} {'TN Mean':>10} {'Difference':>12}\n")
                    f.write("  " + "-" * 50 + "\n")
                    
                    for feature in result.features_used:
                        if feature in X_fp.columns:
                            fp_mean = X_fp[feature].mean()
                            tn_mean = X_tn[feature].mean()
                            diff = fp_mean - tn_mean
                            f.write(f"  {feature:15} {fp_mean:10.3f} {tn_mean:10.3f} {diff:12.3f}\n")
                    f.write("\n")
                
                f.write("\n")
            
            # Summary Statistics
            f.write("=" * 80 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Model Comparison:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Domain':25} {'F1 Macro':>12} {'ROC-AUC':>12} {'Precision':>12} {'Recall':>12}\n")
            f.write("-" * 70 + "\n")
            
            for domain, result in self.results.items():
                metrics = result.metrics
                f.write(f"{domain:25} {metrics.get('f1_macro', 0):12.4f} {metrics.get('roc_auc', 0):12.4f} "
                       f"{metrics.get('precision', 0):12.4f} {metrics.get('recall', 0):12.4f}\n")
            f.write("\n")
            
            # Top Features Across All Domains
            f.write("Top Features by Domain:\n")
            f.write("-" * 70 + "\n")
            for domain, result in self.results.items():
                top_3 = list(result.feature_importances.items())[:3]
                top_str = ", ".join([f"{f}({v:.3f})" for f, v in top_3])
                f.write(f"  {domain}: {top_str}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Full analysis report saved to {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Domain-level XGBoost analysis")
    parser.add_argument("--config", "-c", default="preprocessing_config.json", help="Config file")
    parser.add_argument("--input", "-i", help="Override input file")
    parser.add_argument("--output", "-o", help="Override output directory")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Setup logging
    logger = setup_logging(config.get("logging", {}).get("log_level", "INFO"))
    logger.info(f"Loaded configuration from {config_path.absolute()}")
    
    # Get paths
    base_path = config_path.parent
    analysis_config = config.get("domain_analysis", {})
    
    input_file = args.input or analysis_config.get("input_file", "data/engineered/P1M_engineered.csv")
    output_dir = args.output or analysis_config.get("output_dir", "data/analysis/domains")
    report_output = analysis_config.get("report_output", "data/analysis/domains/report.json")
    
    input_path = base_path / input_file if not Path(input_file).is_absolute() else Path(input_file)
    output_path = base_path / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    report_path = base_path / report_output if not Path(report_output).is_absolute() else Path(report_output)
    
    # Load data
    logger.info(f"Loading data from {input_path}")
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} compounds with {len(df.columns)} columns")
    
    # Run analysis
    analyzer = DomainAnalyzer(config, logger)
    analyzer.report.input_file = str(input_path)
    analyzer.report.output_dir = str(output_path)
    
    results = analyzer.analyze(df)
    
    # Get feature matrix for viz
    X, _ = analyzer.prepare_features(df)
    X_clean = X[~X.isna().any(axis=1)]
    df_clean = df.loc[X_clean.index]
    
    # Generate visualizations
    analyzer.generate_visualizations(df_clean, X_clean, output_path)
    
    # Generate comprehensive text report
    analyzer.generate_full_report_text(df_clean, output_path)
    
    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(analyzer.report.to_dict(), f, indent=2)
    logger.info(f"Saved report to {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("DOMAIN ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\n  Compounds Analyzed: {analyzer.report.compounds_after_missing_removal:,}")
    print(f"  Domains: {analyzer.report.domains_analyzed}")
    
    print("\n  Domain Counts:")
    for domain, count in analyzer.report.domain_counts.items():
        print(f"    {domain}: {count:,}")
    
    print("\n  Model Performance:")
    for domain, model_info in analyzer.report.models.items():
        metrics = model_info.get("metrics", {})
        print(f"\n    {domain}:")
        print(f"      F1 Macro: {metrics.get('f1_macro', 0):.4f}")
        print(f"      ROC-AUC:  {metrics.get('roc_auc', 0):.4f}")
        
        top_features = list(model_info.get("feature_importances", {}).items())[:3]
        if top_features:
            print(f"      Top Features: {', '.join([f'{f[0]}({f[1]:.2f})' for f in top_features])}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
