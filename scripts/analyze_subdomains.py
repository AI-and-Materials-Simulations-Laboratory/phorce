#!/usr/bin/env python3
"""
PHORCE Sub-Domain Decision Tree Analysis Script

This script performs decision tree analysis on sub-domain classifications:
- Water_Soluble (within water_related)
- Aquatic_Bioavailable (within water_related)
- Bioconcentration_Risk (within water_related)
- Persistence (within water_related)
- Soil_Mobility (within water_related)

Pipeline Stage: 3 (After domain subsetting)
Input: data/subsets/P1M_with_subsets.csv
Output: data/analysis/subdomains/

Usage:
    python scripts/analyze_subdomains.py
    python scripts/analyze_subdomains.py --config custom_config.json
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

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, 
    f1_score, precision_score, recall_score,
    precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    logger = logging.getLogger("phorce_subdomain_analysis")
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
    subdomain_name: str
    parent_domain: Optional[str]
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
class SensitivityResult:
    """Container for sensitivity analysis results."""
    subdomain_name: str
    # Bootstrap confidence intervals
    bootstrap_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Feature perturbation analysis
    perturbation_stability: Dict[str, float] = field(default_factory=dict)
    # Leave-one-out feature analysis
    feature_ablation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Decision boundary analysis
    boundary_analysis: Dict[str, Any] = field(default_factory=dict)
    # Cross-validation stability
    cv_stability: Dict[str, float] = field(default_factory=dict)
    # Threshold robustness
    threshold_robustness: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubdomainAnalysisReport:
    """Report for subdomain-level analysis."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    input_file: str = ""
    output_dir: str = ""
    total_compounds: int = 0
    compounds_after_missing_removal: int = 0
    subdomains_analyzed: List[str] = field(default_factory=list)
    subdomain_info: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    pca_results: Dict[str, Any] = field(default_factory=dict)
    similarity_results: Dict[str, Any] = field(default_factory=dict)
    sensitivity_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return _to_json_serializable({
            "timestamp": self.timestamp,
            "input_file": self.input_file,
            "output_dir": self.output_dir,
            "total_compounds": self.total_compounds,
            "compounds_after_missing_removal": self.compounds_after_missing_removal,
            "subdomains_analyzed": self.subdomains_analyzed,
            "subdomain_info": self.subdomain_info,
            "models": self.models,
            "pca_results": self.pca_results,
            "similarity_results": self.similarity_results,
            "sensitivity_results": self.sensitivity_results
        })


# =============================================================================
# SUBDOMAIN ANALYZER
# =============================================================================

class SubdomainAnalyzer:
    """Decision tree analyzer for sub-domain classifications."""
    
    # Target subdomains with their parent domains
    TARGET_SUBDOMAINS = {
        "Water_Soluble": "water_related",
        "Aquatic_Bioavailable": "water_related",
        "Bioconcentration_Risk": "water_related",
        "Persistence": "water_related",
        "Soil_Mobility": "water_related"
    }
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("phorce_subdomain_analysis")
        self.report = SubdomainAnalysisReport()
        self.results: Dict[str, ModelResult] = {}
        self.sensitivity_results: Dict[str, SensitivityResult] = {}
        self.pca_result: Optional[PCAResult] = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature matrix."""
        analysis_config = self.config.get("subdomain_analysis", {})
        feature_config = analysis_config.get("features", {})
        
        numeric_features = feature_config.get("numeric", [
            "mw", "xlogp", "polararea", "hbonddonor", "hbondacc", 
            "TotalHBond", "complexity", "rotbonds", "charge"
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
        """Create preprocessing pipeline."""
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
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, subdomain: str, 
                    parent_domain: Optional[str], preprocessor: ColumnTransformer) -> ModelResult:
        """Train an XGBoost model for a subdomain."""
        self.logger.info(f"Training XGBoost model for subdomain: {subdomain}")
        if parent_domain:
            self.logger.info(f"  (Parent domain: {parent_domain})")
        
        analysis_config = self.config.get("subdomain_analysis", {})
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
                n_jobs=1  # Single-threaded to avoid memory issues
            ))
        ])
        
        # Grid search with XGBoost parameters
        param_grid = {
            "clf__max_depth": model_settings.get("max_depth_range", [3, 5, 7]),
            "clf__n_estimators": model_settings.get("n_estimators_range", [50, 100]),
            "clf__learning_rate": model_settings.get("learning_rate_range", [0.1, 0.3])
        }
        
        cv = StratifiedKFold(n_splits=analysis_config.get("cv_folds", 5), shuffle=True, random_state=random_seed)
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv,
            scoring=model_settings.get("scoring_metric", "f1_macro"),
            n_jobs=-1, refit=True
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
            subdomain_name=subdomain,
            parent_domain=parent_domain,
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
    
    def compute_similarity(self, X: pd.DataFrame, df: pd.DataFrame, subdomain: str, 
                          max_samples: int = 5000) -> Dict[str, Any]:
        """Compute intra-group similarity for a subdomain."""
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
        
        # Get subdomain indices
        subdomain_mask = df_sampled[subdomain] == 1
        subdomain_indices = np.where(subdomain_mask)[0]
        
        if len(subdomain_indices) < 2:
            return {"n_samples": 0}
        
        # Compute similarity matrices
        cos_sim = cosine_similarity(X_scaled)
        euc_dist = euclidean_distances(X_scaled)
        euc_sim = 1 / (1 + euc_dist)
        
        # Intra-group similarity
        cos_group = cos_sim[np.ix_(subdomain_indices, subdomain_indices)]
        euc_group = euc_sim[np.ix_(subdomain_indices, subdomain_indices)]
        
        # Exclude diagonal
        n = len(subdomain_indices)
        mask = ~np.eye(n, dtype=bool)
        
        return {
            "n_samples": n,
            "cosine_mean": float(cos_group[mask].mean()),
            "cosine_std": float(cos_group[mask].std()),
            "euclidean_mean": float(euc_group[mask].mean()),
            "euclidean_std": float(euc_group[mask].std())
        }
    
    # =========================================================================
    # SENSITIVITY ANALYSIS METHODS
    # =========================================================================
    
    def perform_sensitivity_analysis(self, X: pd.DataFrame, y: pd.Series, 
                                     subdomain: str, result: ModelResult,
                                     config: Dict[str, Any]) -> SensitivityResult:
        """Perform comprehensive sensitivity analysis for a subdomain model."""
        self.logger.info(f"  Performing sensitivity analysis for {subdomain}...")
        
        sensitivity_result = SensitivityResult(subdomain_name=subdomain)
        
        # Bootstrap confidence intervals
        if config.get("bootstrap_enabled", True):
            sensitivity_result.bootstrap_metrics = self._bootstrap_confidence_intervals(
                X, y, result.model, 
                n_iterations=config.get("bootstrap_iterations", 100),
                confidence_level=config.get("confidence_level", 0.95)
            )
        
        # Feature perturbation analysis
        if config.get("perturbation_enabled", True):
            sensitivity_result.perturbation_stability = self._feature_perturbation_analysis(
                X, y, result.model,
                noise_levels=config.get("noise_levels", [0.01, 0.05, 0.1, 0.2])
            )
        
        # Leave-one-out feature analysis
        if config.get("feature_ablation_enabled", True):
            sensitivity_result.feature_ablation = self._feature_ablation_analysis(
                X, y, result.features_used
            )
        
        # Decision boundary analysis
        if config.get("boundary_analysis_enabled", True):
            sensitivity_result.boundary_analysis = self._decision_boundary_analysis(
                X, y, result.model, result.y_proba
            )
        
        # Cross-validation stability
        if config.get("cv_stability_enabled", True):
            sensitivity_result.cv_stability = self._cv_stability_analysis(
                X, y, result.model, n_splits=config.get("cv_splits", 10)
            )
        
        # Threshold robustness
        if config.get("threshold_robustness_enabled", True):
            sensitivity_result.threshold_robustness = self._threshold_robustness_analysis(
                result.y_true, result.y_proba
            )
        
        return sensitivity_result
    
    def _bootstrap_confidence_intervals(self, X: pd.DataFrame, y: pd.Series,
                                        model: Pipeline, n_iterations: int = 100,
                                        confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals for model metrics."""
        metrics_samples = {
            'accuracy': [], 'f1_macro': [], 'precision': [], 'recall': [], 'roc_auc': []
        }
        
        n_samples = len(X)
        random_state = np.random.RandomState(42)
        
        for _ in range(n_iterations):
            # Bootstrap sample
            indices = random_state.choice(n_samples, size=n_samples, replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # Get out-of-bag samples
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[np.unique(indices)] = False
            
            if oob_mask.sum() < 10:
                continue
            
            X_oob = X.iloc[oob_mask]
            y_oob = y.iloc[oob_mask]
            
            try:
                y_pred = model.predict(X_oob)
                y_proba = model.predict_proba(X_oob)
                
                metrics_samples['accuracy'].append(float((y_pred == y_oob).mean()))
                metrics_samples['f1_macro'].append(float(f1_score(y_oob, y_pred, average='macro', zero_division=0)))
                metrics_samples['precision'].append(float(precision_score(y_oob, y_pred, average='binary', zero_division=0)))
                metrics_samples['recall'].append(float(recall_score(y_oob, y_pred, average='binary', zero_division=0)))
                
                if len(np.unique(y_oob)) > 1:
                    metrics_samples['roc_auc'].append(float(roc_auc_score(y_oob, y_proba[:, 1])))
            except Exception:
                continue
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        result = {}
        for metric, samples in metrics_samples.items():
            if len(samples) > 10:
                samples = np.array(samples)
                result[metric] = {
                    'mean': float(np.mean(samples)),
                    'std': float(np.std(samples)),
                    'ci_lower': float(np.percentile(samples, alpha/2 * 100)),
                    'ci_upper': float(np.percentile(samples, (1 - alpha/2) * 100)),
                    'n_samples': len(samples)
                }
        
        return result
    
    def _feature_perturbation_analysis(self, X: pd.DataFrame, y: pd.Series,
                                       model: Pipeline, 
                                       noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict[str, float]:
        """Analyze model stability under feature perturbations."""
        baseline_pred = model.predict(X)
        baseline_proba = model.predict_proba(X)[:, 1]
        
        results = {}
        
        for noise_level in noise_levels:
            # Add Gaussian noise proportional to feature standard deviation
            X_perturbed = X.copy()
            for col in X.columns:
                std = X[col].std()
                noise = np.random.normal(0, std * noise_level, len(X))
                X_perturbed[col] = X[col] + noise
            
            perturbed_pred = model.predict(X_perturbed)
            perturbed_proba = model.predict_proba(X_perturbed)[:, 1]
            
            # Calculate stability metrics
            prediction_stability = float((baseline_pred == perturbed_pred).mean())
            proba_mae = float(np.abs(baseline_proba - perturbed_proba).mean())
            proba_correlation = float(np.corrcoef(baseline_proba, perturbed_proba)[0, 1])
            
            results[f'noise_{int(noise_level*100)}pct'] = {
                'prediction_stability': prediction_stability,
                'probability_mae': proba_mae,
                'probability_correlation': proba_correlation
            }
        
        # Overall stability score (average stability across noise levels)
        avg_stability = np.mean([v['prediction_stability'] for v in results.values()])
        results['overall_stability_score'] = float(avg_stability)
        
        return results
    
    def _feature_ablation_analysis(self, X: pd.DataFrame, y: pd.Series,
                                   feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Analyze impact of removing each feature on model performance."""
        from sklearn.model_selection import cross_val_score
        
        results = {}
        
        # Train baseline model with all features
        baseline_model = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric='logloss', n_jobs=1
        )
        
        try:
            baseline_scores = cross_val_score(baseline_model, X, y, cv=5, scoring='f1_macro')
            baseline_mean = float(np.mean(baseline_scores))
            baseline_std = float(np.std(baseline_scores))
            results['_baseline'] = {
                'f1_macro_mean': baseline_mean,
                'f1_macro_std': baseline_std,
                'n_features': len(feature_names)
            }
        except Exception:
            return results
        
        # Test removing each feature
        for feature in feature_names:
            remaining_features = [f for f in feature_names if f != feature]
            if len(remaining_features) == 0:
                continue
            
            X_ablated = X[remaining_features]
            
            try:
                ablated_scores = cross_val_score(baseline_model, X_ablated, y, cv=5, scoring='f1_macro')
                ablated_mean = float(np.mean(ablated_scores))
                ablated_std = float(np.std(ablated_scores))
                
                # Impact = how much performance drops when feature is removed
                impact = baseline_mean - ablated_mean
                
                results[feature] = {
                    'f1_macro_without': ablated_mean,
                    'f1_macro_std': ablated_std,
                    'impact': impact,
                    'relative_impact_pct': float(impact / baseline_mean * 100) if baseline_mean > 0 else 0
                }
            except Exception:
                continue
        
        return results
    
    def _decision_boundary_analysis(self, X: pd.DataFrame, y: pd.Series,
                                    model: Pipeline, y_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze points near decision boundaries."""
        # Points near decision boundary have probability close to 0.5
        proba_pos = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
        
        # Define boundary regions
        boundary_thresholds = [0.1, 0.15, 0.2, 0.25]
        
        results = {
            'total_samples': len(proba_pos)
        }
        
        for thresh in boundary_thresholds:
            # Points within thresh of 0.5 decision boundary
            boundary_mask = np.abs(proba_pos - 0.5) < thresh
            n_boundary = boundary_mask.sum()
            pct_boundary = float(n_boundary / len(proba_pos) * 100)
            
            # Accuracy on boundary vs non-boundary points
            y_pred = (proba_pos >= 0.5).astype(int)
            
            results[f'within_{int(thresh*100)}pct_of_boundary'] = {
                'n_samples': int(n_boundary),
                'percentage': pct_boundary
            }
            
            if n_boundary > 0 and (~boundary_mask).sum() > 0:
                boundary_acc = float((y_pred[boundary_mask] == y.values[boundary_mask]).mean()) if boundary_mask.sum() > 0 else 0
                non_boundary_acc = float((y_pred[~boundary_mask] == y.values[~boundary_mask]).mean()) if (~boundary_mask).sum() > 0 else 0
                
                results[f'within_{int(thresh*100)}pct_of_boundary']['accuracy'] = boundary_acc
                results[f'within_{int(thresh*100)}pct_of_boundary']['non_boundary_accuracy'] = non_boundary_acc
        
        # Identify most uncertain predictions
        uncertainty = np.abs(proba_pos - 0.5)
        most_uncertain_idx = np.argsort(uncertainty)[:min(100, len(uncertainty))]
        results['most_uncertain'] = {
            'mean_probability': float(np.mean(proba_pos[most_uncertain_idx])),
            'min_uncertainty': float(np.min(uncertainty)),
            'median_uncertainty': float(np.median(uncertainty))
        }
        
        return results
    
    def _cv_stability_analysis(self, X: pd.DataFrame, y: pd.Series,
                               model: Pipeline, n_splits: int = 10) -> Dict[str, float]:
        """Analyze cross-validation stability across multiple splits."""
        from sklearn.model_selection import RepeatedStratifiedKFold
        
        # Use repeated stratified k-fold for more robust estimates
        cv = RepeatedStratifiedKFold(n_splits=min(5, n_splits), n_repeats=2, random_state=42)
        
        fold_metrics = {
            'f1_macro': [], 'accuracy': [], 'precision': [], 'recall': []
        }
        
        # Clone the model to avoid contamination
        base_model = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric='logloss', n_jobs=1
        )
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            try:
                base_model.fit(X_train, y_train)
                y_pred = base_model.predict(X_test)
                
                fold_metrics['f1_macro'].append(float(f1_score(y_test, y_pred, average='macro', zero_division=0)))
                fold_metrics['accuracy'].append(float((y_pred == y_test).mean()))
                fold_metrics['precision'].append(float(precision_score(y_test, y_pred, average='binary', zero_division=0)))
                fold_metrics['recall'].append(float(recall_score(y_test, y_pred, average='binary', zero_division=0)))
            except Exception:
                continue
        
        results = {}
        for metric, values in fold_metrics.items():
            if len(values) > 0:
                results[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'cv_coefficient': float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0,
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values))
                }
        
        return results
    
    def _threshold_robustness_analysis(self, y_true: np.ndarray, 
                                       y_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze model robustness across different decision thresholds."""
        proba_pos = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
        
        # Sweep thresholds and compute metrics
        thresholds = np.linspace(0.1, 0.9, 17)
        
        metrics_by_threshold = []
        for thresh in thresholds:
            y_pred = (proba_pos >= thresh).astype(int)
            
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            metrics_by_threshold.append({
                'threshold': float(thresh),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'fp_count': int(fp),
                'fn_count': int(fn)
            })
        
        # Find optimal thresholds
        f1_scores = [m['f1'] for m in metrics_by_threshold]
        best_f1_idx = np.argmax(f1_scores)
        
        # Find threshold range where F1 is within 5% of optimal
        best_f1 = f1_scores[best_f1_idx]
        robust_range = [m['threshold'] for m in metrics_by_threshold 
                       if m['f1'] >= best_f1 * 0.95]
        
        results = {
            'optimal_threshold': float(thresholds[best_f1_idx]),
            'optimal_f1': float(best_f1),
            'robust_threshold_range': [float(min(robust_range)), float(max(robust_range))] if robust_range else [0.5, 0.5],
            'robust_range_width': float(max(robust_range) - min(robust_range)) if robust_range else 0,
            'f1_at_0.5': float(metrics_by_threshold[8]['f1']),  # threshold = 0.5
            'f1_variance': float(np.var(f1_scores)),
            'f1_std': float(np.std(f1_scores)),
            'metrics_by_threshold': metrics_by_threshold
        }
        
        return results
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, ModelResult]:
        """Run the full analysis pipeline."""
        self.logger.info("Starting subdomain-level decision tree analysis...")
        
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
        
        # Find existing subdomains
        existing_subdomains = {k: v for k, v in self.TARGET_SUBDOMAINS.items() if k in df.columns}
        if not existing_subdomains:
            raise ValueError(f"No target subdomains found. Expected: {list(self.TARGET_SUBDOMAINS.keys())}")
        
        self.report.subdomains_analyzed = list(existing_subdomains.keys())
        self.report.subdomain_info = {
            k: {
                "parent_domain": v,
                "count": int(df[k].sum()),
                "percentage": float(df[k].mean() * 100)
            }
            for k, v in existing_subdomains.items()
        }
        
        self.logger.info(f"Analyzing {len(existing_subdomains)} subdomains: {list(existing_subdomains.keys())}")
        
        # PCA
        analysis_config = self.config.get("subdomain_analysis", {})
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
            for subdomain in tqdm(existing_subdomains.keys(), desc="Similarity analysis"):
                sim_result = self.compute_similarity(X, df, subdomain, sim_config.get("max_samples", 5000))
                self.report.similarity_results[subdomain] = sim_result
                if sim_result.get("n_samples", 0) > 0:
                    self.logger.info(f"  {subdomain}: Cosine sim = {sim_result['cosine_mean']:.3f} ± {sim_result['cosine_std']:.3f}")
                else:
                    self.logger.warning(f"  {subdomain}: Insufficient samples for similarity")
        
        # Train models
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING XGBOOST MODELS")
        self.logger.info("=" * 60)
        
        for subdomain, parent_domain in tqdm(existing_subdomains.items(), desc="Training models"):
            y = df[subdomain]
            class_counts = y.value_counts()
            self.logger.info(f"\n{subdomain} class distribution: {dict(class_counts)}")
            
            if len(class_counts) < 2 or class_counts.min() < 5:
                self.logger.warning(f"Skipping {subdomain}: insufficient samples")
                continue
            
            preprocessor = self.create_preprocessor(feature_names)
            result = self.train_model(X, y, subdomain, parent_domain, preprocessor)
            self.results[subdomain] = result
            
            self.report.models[subdomain] = {
                "parent_domain": parent_domain,
                "metrics": result.metrics,
                "best_params": result.best_params,
                "feature_importances": result.feature_importances,
                "extracted_ranges": result.extracted_ranges
            }
        
        # Sensitivity Analysis
        sensitivity_config = analysis_config.get("sensitivity_analysis", {})
        if sensitivity_config.get("enabled", True):
            self.logger.info("\n" + "=" * 60)
            self.logger.info("PERFORMING SENSITIVITY ANALYSIS")
            self.logger.info("=" * 60)
            
            for subdomain in tqdm(self.results.keys(), desc="Sensitivity analysis"):
                result = self.results[subdomain]
                y = df[subdomain]
                
                sens_result = self.perform_sensitivity_analysis(
                    X, y, subdomain, result, sensitivity_config
                )
                self.sensitivity_results[subdomain] = sens_result
                
                # Store in report
                self.report.sensitivity_results[subdomain] = {
                    "bootstrap_metrics": sens_result.bootstrap_metrics,
                    "perturbation_stability": sens_result.perturbation_stability,
                    "feature_ablation": sens_result.feature_ablation,
                    "boundary_analysis": sens_result.boundary_analysis,
                    "cv_stability": sens_result.cv_stability,
                    "threshold_robustness": {
                        k: v for k, v in sens_result.threshold_robustness.items()
                        if k != 'metrics_by_threshold'  # Exclude detailed list from JSON
                    }
                }
                
                # Log summary
                if sens_result.bootstrap_metrics.get('f1_macro'):
                    boot = sens_result.bootstrap_metrics['f1_macro']
                    self.logger.info(f"  {subdomain}: F1 = {boot['mean']:.3f} [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")
                if sens_result.perturbation_stability.get('overall_stability_score'):
                    self.logger.info(f"    Perturbation stability: {sens_result.perturbation_stability['overall_stability_score']:.3f}")
        
        self.logger.info("\nSubdomain analysis complete")
        return self.results
    
    def generate_visualizations(self, df: pd.DataFrame, X: pd.DataFrame, output_dir: Path):
        """Generate all visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Generating visualizations...")
        
        for subdomain, result in tqdm(self.results.items(), desc="Generating plots"):
            self.logger.info(f"  Generating plots for {subdomain}...")
            
            # Feature importance
            self._plot_feature_importance(result, output_dir)
            
            # Confusion matrix
            self._plot_confusion_matrix(result, output_dir)
            
            # ROC curve
            self._plot_roc_curve(result, output_dir)
            
            # Decision tree
            self._plot_decision_tree(result, output_dir)
            
            # Save rules
            self._save_rules(result, output_dir)
            
            # Threshold sweep analysis
            self._plot_threshold_sweep(result, output_dir)
            
            # Calibration curve
            self._plot_calibration_curve(result, output_dir)
            
            # False positive analysis
            self._analyze_false_positives(result, df, output_dir)
            
            # Feature attribution on FP cases
            self._plot_fp_feature_attribution(result, output_dir)
        
        # PCA plots
        if self.pca_result:
            self._plot_pca_variance(output_dir)
            self._plot_pca_loadings(output_dir)
            self._plot_pca_all_subdomains(df, output_dir)
            for subdomain in self.results.keys():
                self._plot_pca_scatter(df, subdomain, output_dir)
                self._plot_pca_pass_fail(df, subdomain, output_dir)
            self._plot_pca_pass_fail_combined(df, output_dir)
        
        # Comparison plots
        self._plot_subdomain_comparison(output_dir)
        self._plot_similarity_comparison(output_dir)
        
        # Sensitivity analysis plots
        if self.sensitivity_results:
            self._plot_sensitivity_summary(output_dir)
            self._plot_bootstrap_confidence_intervals(output_dir)
            self._plot_perturbation_stability(output_dir)
            self._plot_feature_ablation(output_dir)
            self._plot_decision_boundary_analysis(output_dir)
            self._save_sensitivity_report(output_dir)
    
    def _plot_feature_importance(self, result: ModelResult, output_dir: Path):
        """Plot feature importance (top 3 features only)."""
        features = list(result.feature_importances.keys())[:3]
        importances = list(result.feature_importances.values())[:3]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontdict={'size': 40})
        ax.set_xlabel('Importance', fontdict={'size': 40})
        ax.xaxis.set_tick_params(labelsize=32)
        title = f'{format_label(result.subdomain_name)}'
        #ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"feature_importance_{result.subdomain_name}.png", dpi=600)
        plt.close()
    
    def _plot_confusion_matrix(self, result: ModelResult, output_dir: Path):
        """Plot confusion matrix with detailed annotations."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cm = confusion_matrix(result.y_true, result.y_pred)
        
        # Calculate percentages
        cm_sum = cm.sum()
        cm_percentages = cm / cm_sum * 100
        
        # Calculate per-class metrics
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        # Create annotation labels with counts in scientific notation and percentages
        labels = np.array([
            [f'TN\n{tn:.2e}\n({tn/total*100:.1f}%)', f'FP\n{fp:.2e}\n({fp/total*100:.1f}%)'],
            [f'FN\n{fn:.2e}\n({fn/total*100:.1f}%)', f'TP\n{tp:.2e}\n({tp/total*100:.1f}%)']
        ])
        
        subdomain_name_display = format_label(result.subdomain_name)
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax,
                    xticklabels=['Predicted: No', 'Predicted: Yes'],
                    yticklabels=['Actual: No', 'Actual: Yes'],
                    annot_kws={'size': 38},
                    vmin=0, vmax=1200)
        
        ax.set_xlabel('Predicted Label', fontweight='bold', fontdict={'size': 32})
        ax.set_ylabel('Actual Label', fontweight='bold', fontdict={'size': 32})
        ax.set_title(f'{subdomain_name_display}', fontweight='bold', fontdict={'size': 36})
        
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_{result.subdomain_name}.png", dpi=600, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, result: ModelResult, output_dir: Path):
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(result.y_true, result.y_proba[:, 1])
        auc = result.metrics.get("roc_auc", 0)
        
        subdomain_display = format_label(result.subdomain_name)
        ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(f'{subdomain_display}', fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"roc_curve_{result.subdomain_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_decision_tree(self, result: ModelResult, output_dir: Path):
        """Plot XGBoost feature importance as a substitute for tree visualization.
        
        Note: XGBoost's plot_tree requires Graphviz which has dependency issues on Windows.
        Instead, we plot a detailed feature importance chart.
        """
        xgb_model = result.model.named_steps['clf']
        subdomain_name_display = result.subdomain_name.replace("_", " ").title()
        
        # Create a more detailed feature importance visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Gain-based importance
        importance_gain = xgb_model.get_booster().get_score(importance_type='gain')
        if importance_gain:
            features_gain = list(importance_gain.keys())
            values_gain = list(importance_gain.values())
            # Map f0, f1, etc. to actual feature names
            feature_map = {f'f{i}': name for i, name in enumerate(result.features_used)}
            features_gain = [feature_map.get(f, f) for f in features_gain]
            
            y_pos = np.arange(len(features_gain))
            axes[0].barh(y_pos, values_gain, color='steelblue')
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(features_gain)
            axes[0].invert_yaxis()
            axes[0].set_xlabel('Gain')
            axes[0].set_title(f'Feature Importance (Gain): {subdomain_name_display}')
        
        # Plot 2: Weight-based importance (number of times feature appears in trees)
        importance_weight = xgb_model.get_booster().get_score(importance_type='weight')
        if importance_weight:
            features_weight = list(importance_weight.keys())
            values_weight = list(importance_weight.values())
            features_weight = [feature_map.get(f, f) for f in features_weight]
            
            y_pos = np.arange(len(features_weight))
            axes[1].barh(y_pos, values_weight, color='coral')
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(features_weight)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Weight (Split Count)')
            axes[1].set_title(f'Feature Importance (Weight): {subdomain_name_display}')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"xgboost_importance_{result.subdomain_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save the tree structure as text file
        self._save_tree_structure(result, output_dir)
    
    def _save_tree_structure(self, result: ModelResult, output_dir: Path):
        """Save the XGBoost tree structure to a text file."""
        xgb_model = result.model.named_steps['clf']
        subdomain_name_display = result.subdomain_name.replace("_", " ").title()
        
        try:
            tree_dump = xgb_model.get_booster().get_dump(dump_format='text')
            if tree_dump:
                with open(output_dir / f"tree_structure_{result.subdomain_name}.txt", 'w') as f:
                    f.write(f"XGBoost Tree Structure for {subdomain_name_display}\n")
                    if result.parent_domain:
                        f.write(f"Parent Domain: {result.parent_domain}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Model Parameters:\n")
                    f.write(f"  - Number of estimators: {xgb_model.n_estimators}\n")
                    f.write(f"  - Max depth: {xgb_model.max_depth}\n")
                    f.write(f"  - Learning rate: {xgb_model.learning_rate}\n\n")
                    f.write(f"Number of trees: {len(tree_dump)}\n\n")
                    
                    # Create feature name mapping for readability
                    feature_map = {f'f{i}': name for i, name in enumerate(result.features_used)}
                    
                    # Write all tree structures with feature name mapping
                    for i, tree_text in enumerate(tree_dump):
                        f.write(f"\n{'='*60}\n")
                        f.write(f"Tree {i + 1} Structure:\n")
                        f.write("-" * 40 + "\n")
                        # Replace feature indices with actual feature names
                        mapped_tree_text = tree_text
                        for idx, name in feature_map.items():
                            mapped_tree_text = mapped_tree_text.replace(f'[{idx}<', f'[{name}<')
                            mapped_tree_text = mapped_tree_text.replace(f'[{idx}]', f'[{name}]')
                        f.write(mapped_tree_text)
                    
                self.logger.info(f"  Saved tree structure for {result.subdomain_name}")
        except Exception as e:
            self.logger.warning(f"Could not save tree structure for {result.subdomain_name}: {e}")
    
    def _save_rules(self, result: ModelResult, output_dir: Path):
        """Save model summary to text file."""
        with open(output_dir / f"rules_{result.subdomain_name}.txt", 'w') as f:
            f.write(f"XGBoost Model Summary for {result.subdomain_name}\n")
            if result.parent_domain:
                f.write(f"Parent Domain: {result.parent_domain}\n")
            f.write("=" * 50 + "\n\n")
            f.write(result.rules_text)
            f.write("\n\nFeature Ranges:\n")
            for feature, ranges in result.extracted_ranges.items():
                f.write(f"  {feature}: [{ranges['min']:.3f}, {ranges['max']:.3f}] (mean: {ranges['mean']:.3f}, std: {ranges['std']:.3f})\n")
    
    def _plot_pca_variance(self, output_dir: Path):
        """Plot PCA variance explained."""
        if not self.pca_result:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n = self.pca_result.n_components
        x = range(1, n + 1)
        
        ax.bar(x, self.pca_result.explained_variance_ratio, alpha=0.7, label='Individual')
        ax.plot(x, self.pca_result.cumulative_variance, 'ro-', label='Cumulative')
        ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.7)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA: Variance Explained')
        ax.legend()
        ax.set_xticks(x)
        
        plt.tight_layout()
        plt.savefig(output_dir / "pca_variance.png", dpi=150)
        plt.close()
    
    def _plot_pca_loadings(self, output_dir: Path):
        """Plot PCA loadings heatmap."""
        if not self.pca_result:
            return
        
        loadings_df = pd.DataFrame(self.pca_result.loadings).T
        loadings_df.columns = [f'PC{i+1}' for i in range(loadings_df.shape[1])]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
        ax.set_title('PCA Loadings')
        
        plt.tight_layout()
        plt.savefig(output_dir / "pca_loadings.png", dpi=150)
        plt.close()
    
    def _plot_pca_scatter(self, df: pd.DataFrame, subdomain: str, output_dir: Path):
        """Plot PCA scatter colored by subdomain."""
        if not self.pca_result or self.pca_result.coordinates is None:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get subdomain labels for clean data
        X_clean = df[list(self.pca_result.loadings.keys())].dropna()
        subdomain_labels = df.loc[X_clean.index, subdomain].values
        
        # Sort so True values plot on top
        sort_idx = np.argsort(subdomain_labels)
        coords = self.pca_result.coordinates[sort_idx]
        labels = subdomain_labels[sort_idx]
        
        #colors = ['#E8E8E8' if l == 0 else '#1f77b4' for l in labels]
        colors = ['#E8E8E8' if l == 0 else "#b41f1f" for l in labels]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.5, s=15)
        ax.set_xlabel(f'PC1 ({self.pca_result.explained_variance_ratio[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({self.pca_result.explained_variance_ratio[1]*100:.1f}%)')
        subdomain = format_label(subdomain)
        ax.set_title(f'{subdomain}', fontweight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E8E8E8', label='False'),
            Patch(facecolor='#1f77b4', label='True')
        ]
        #ax.legend(handles=legend_elements, title=subdomain)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"pca_scatter_{subdomain}.png", dpi=150)
        plt.close()
    
    def _plot_pca_pass_fail(self, df: pd.DataFrame, subdomain: str, output_dir: Path):
        """Plot PCA scatter showing pass (correct classification) vs fail (incorrect classification)."""
        if not self.pca_result or self.pca_result.coordinates is None:
            return
        
        if subdomain not in self.results:
            return
        
        result = self.results[subdomain]
        
        # Get clean data indices used for PCA
        X_clean = df[list(self.pca_result.loadings.keys())].dropna()
        df_aligned = df.loc[X_clean.index]
        
        # Get actual labels for all data
        y_actual = df_aligned[subdomain].values
        
        # Predict on all data using the trained model
        X_features = X_clean[result.features_used]
        y_pred_all = result.model.predict(X_features)
        
        # Determine pass/fail: pass if prediction matches actual label
        pass_mask = (y_actual == y_pred_all)
        
        # Create classification result array for coloring
        # 0 = True Negative (pass), 1 = True Positive (pass), 2 = False Positive (fail), 3 = False Negative (fail)
        classification = np.zeros(len(y_actual), dtype=int)
        classification[(y_actual == 0) & (y_pred_all == 0)] = 0  # TN
        classification[(y_actual == 1) & (y_pred_all == 1)] = 1  # TP
        classification[(y_actual == 0) & (y_pred_all == 1)] = 2  # FP
        classification[(y_actual == 1) & (y_pred_all == 0)] = 3  # FN
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        coords = self.pca_result.coordinates
        
        # Define colors: Pass (green shades), Fail (red shades)
        colors = {
            0: '#90EE90',  # TN - Light green
            1: '#228B22',  # TP - Forest green
            2: '#FF6B6B',  # FP - Light red
            3: '#8B0000'   # FN - Dark red
        }
        
        # Plot in order: fail cases on top for visibility
        plot_order = [0, 1, 2, 3]  # TN, TP, FP, FN
        for cls in plot_order:
            mask = classification == cls
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1], 
                          c=colors[cls], alpha=0.6, s=15, label=None)
        
        ax.set_xlabel(f'PC1 ({self.pca_result.explained_variance_ratio[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({self.pca_result.explained_variance_ratio[1]*100:.1f}%)')
        subdomain_display = format_label(subdomain)
        ax.set_title(f'{subdomain_display} - Pass/Fail', fontweight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#228B22', label='True Positive (Pass)'),
            Patch(facecolor='#90EE90', label='True Negative (Pass)'),
            Patch(facecolor='#FF6B6B', label='False Positive (Fail)'),
            Patch(facecolor='#8B0000', label='False Negative (Fail)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"pca_pass_fail_{subdomain}.png", dpi=300)
        plt.close()
    
    def _plot_pca_pass_fail_combined(self, df: pd.DataFrame, output_dir: Path):
        """Plot combined PCA pass/fail figure for all subdomains."""
        if not self.pca_result or self.pca_result.coordinates is None:
            return
        
        if not self.results:
            return
        
        subdomains = list(self.results.keys())
        n_subdomains = len(subdomains)
        
        if n_subdomains == 0:
            return
        
        # Calculate grid dimensions
        n_cols = min(3, n_subdomains)
        n_rows = (n_subdomains + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_subdomains == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Get clean data indices used for PCA
        X_clean = df[list(self.pca_result.loadings.keys())].dropna()
        df_aligned = df.loc[X_clean.index]
        coords = self.pca_result.coordinates
        
        # Define colors
        colors = {
            0: '#90EE90',  # TN - Light green
            1: '#228B22',  # TP - Forest green
            2: '#FF6B6B',  # FP - Light red
            3: '#8B0000'   # FN - Dark red
        }
        
        for idx, subdomain in enumerate(subdomains):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            result = self.results[subdomain]
            
            # Get actual labels
            y_actual = df_aligned[subdomain].values
            
            # Predict on all data
            X_features = X_clean[result.features_used]
            y_pred_all = result.model.predict(X_features)
            
            # Classification: TN=0, TP=1, FP=2, FN=3
            classification = np.zeros(len(y_actual), dtype=int)
            classification[(y_actual == 0) & (y_pred_all == 0)] = 0
            classification[(y_actual == 1) & (y_pred_all == 1)] = 1
            classification[(y_actual == 0) & (y_pred_all == 1)] = 2
            classification[(y_actual == 1) & (y_pred_all == 0)] = 3
            
            # Plot each class
            for cls in [0, 1, 2, 3]:
                mask = classification == cls
                if mask.sum() > 0:
                    ax.scatter(coords[mask, 0], coords[mask, 1], 
                              c=colors[cls], alpha=0.5, s=10)
            
            ax.set_xlabel(f'PC1 ({self.pca_result.explained_variance_ratio[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({self.pca_result.explained_variance_ratio[1]*100:.1f}%)')
            subdomain_display = format_label(subdomain)
            ax.set_title(f'{subdomain_display}', fontweight='bold', fontsize=14)
            
            # Calculate pass/fail counts
            pass_count = ((classification == 0) | (classification == 1)).sum()
            fail_count = ((classification == 2) | (classification == 3)).sum()
            accuracy = pass_count / (pass_count + fail_count) * 100
            ax.text(0.02, 0.98, f'Acc: {accuracy:.1f}%', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for idx in range(n_subdomains, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        # Add shared legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#228B22', label='TP (Pass)'),
            Patch(facecolor='#90EE90', label='TN (Pass)'),
            Patch(facecolor='#FF6B6B', label='FP (Fail)'),
            Patch(facecolor='#8B0000', label='FN (Fail)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=12)
        
        plt.suptitle('PCA: Classification Pass/Fail by Subdomain', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(output_dir / "pca_pass_fail_combined.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_all_subdomains(self, df: pd.DataFrame, output_dir: Path):
        """Plot a single PCA figure with all subdomains overlaid to show cluster separation and overlap."""
        if not self.pca_result or self.pca_result.coordinates is None:
            return
        
        if not self.results:
            return
        
        subdomains = list(self.results.keys())
        n_subdomains = len(subdomains)
        
        if n_subdomains == 0:
            return
        
        # Get clean data indices used for PCA
        X_clean = df[list(self.pca_result.loadings.keys())].dropna()
        df_aligned = df.loc[X_clean.index]
        coords = self.pca_result.coordinates
        
        # Define distinct colors for each subdomain
        subdomain_colors = {
            'Water_Soluble': '#1f77b4',         # Blue
            'Aquatic_Bioavailable': '#ff7f0e',  # Orange
            'Bioconcentration_Risk': '#2ca02c', # Green
            'Persistence': '#d62728',           # Red
            'Soil_Mobility': '#9467bd'          # Purple
        }
        # Fallback colors for any additional subdomains
        fallback_colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, sd in enumerate(subdomains):
            if sd not in subdomain_colors:
                subdomain_colors[sd] = fallback_colors[i % len(fallback_colors)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # First, plot points that don't belong to any subdomain (background)
        any_subdomain_mask = np.zeros(len(df_aligned), dtype=bool)
        for subdomain in subdomains:
            if subdomain in df_aligned.columns:
                any_subdomain_mask |= (df_aligned[subdomain].values == 1)
        
        # Plot non-subdomain points as light gray background
        background_mask = ~any_subdomain_mask
        if background_mask.sum() > 0:
            ax.scatter(coords[background_mask, 0], coords[background_mask, 1],
                      c='#E8E8E8', alpha=0.3, s=10, label='None', zorder=1)
        
        # Track overlap counts for each point
        overlap_count = np.zeros(len(df_aligned), dtype=int)
        for subdomain in subdomains:
            if subdomain in df_aligned.columns:
                overlap_count += (df_aligned[subdomain].values == 1).astype(int)
        
        # Plot each subdomain with distinct color
        # Plot single-membership points first, then multi-membership points on top
        for subdomain in subdomains:
            if subdomain not in df_aligned.columns:
                continue
            
            subdomain_mask = df_aligned[subdomain].values == 1
            single_mask = subdomain_mask & (overlap_count == 1)
            
            if single_mask.sum() > 0:
                ax.scatter(coords[single_mask, 0], coords[single_mask, 1],
                          c=subdomain_colors[subdomain], alpha=0.6, s=20,
                          label=format_label(subdomain), zorder=2)
        
        # Plot overlapping points with special marker
        overlap_mask = overlap_count > 1
        if overlap_mask.sum() > 0:
            ax.scatter(coords[overlap_mask, 0], coords[overlap_mask, 1],
                      c='black', alpha=0.8, s=40, marker='x',
                      label=f'Overlap (n={overlap_mask.sum()})', zorder=3)
        
        ax.set_xlabel(f'PC1 ({self.pca_result.explained_variance_ratio[0]*100:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'PC2 ({self.pca_result.explained_variance_ratio[1]*100:.1f}%)', fontweight='bold')
        ax.set_title('PCA: All Subdomains', fontweight='bold', fontsize=16)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add summary stats
        stats_text = "Subdomain Counts:\n"
        for subdomain in subdomains:
            if subdomain in df_aligned.columns:
                count = (df_aligned[subdomain].values == 1).sum()
                stats_text += f"  {format_label(subdomain)}: {count:,}\n"
        stats_text += f"\nOverlap: {overlap_mask.sum():,} pts"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(output_dir / "pca_all_subdomains.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a version with pass/fail for all subdomains combined
        self._plot_pca_all_subdomains_pass_fail(df, output_dir)
    
    def _plot_pca_all_subdomains_pass_fail(self, df: pd.DataFrame, output_dir: Path):
        """Plot a single PCA figure showing pass/fail for all subdomains combined."""
        if not self.pca_result or self.pca_result.coordinates is None:
            return
        
        if not self.results:
            return
        
        subdomains = list(self.results.keys())
        
        # Get clean data indices used for PCA
        X_clean = df[list(self.pca_result.loadings.keys())].dropna()
        df_aligned = df.loc[X_clean.index]
        coords = self.pca_result.coordinates
        
        # For each point, determine if the rule passed or failed for each subdomain it belongs to
        # Pass = correct classification, Fail = incorrect classification
        n_points = len(df_aligned)
        point_status = np.full(n_points, -1)  # -1 = not in any subdomain, 0 = all pass, 1 = some fail
        
        for subdomain in subdomains:
            if subdomain not in df_aligned.columns or subdomain not in self.results:
                continue
            
            result = self.results[subdomain]
            y_actual = df_aligned[subdomain].values
            
            # Predict on all data
            X_features = X_clean[result.features_used]
            y_pred_all = result.model.predict(X_features)
            
            # For points in this subdomain (actual = 1), check if correctly classified
            in_subdomain = y_actual == 1
            correct = y_actual == y_pred_all
            
            # Update point status
            for i in range(n_points):
                if in_subdomain[i]:
                    if point_status[i] == -1:
                        point_status[i] = 0 if correct[i] else 1
                    elif point_status[i] == 0 and not correct[i]:
                        point_status[i] = 1  # At least one fail
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Define colors
        colors_map = {
            -1: '#E8E8E8',  # Not in any subdomain - gray
            0: '#228B22',   # All pass - green
            1: '#FF4444'    # Some fail - red
        }
        
        # Plot each category
        for status, color in colors_map.items():
            mask = point_status == status
            if mask.sum() > 0:
                if status == -1:
                    label = 'Not in subdomain'
                    alpha = 0.3
                    s = 10
                    zorder = 1
                elif status == 0:
                    label = f'Pass (n={mask.sum():,})'
                    alpha = 0.6
                    s = 20
                    zorder = 2
                else:
                    label = f'Fail (n={mask.sum():,})'
                    alpha = 0.8
                    s = 30
                    zorder = 3
                
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=color, alpha=alpha, s=s, label=label, zorder=zorder)
        
        ax.set_xlabel(f'PC1 ({self.pca_result.explained_variance_ratio[0]*100:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'PC2 ({self.pca_result.explained_variance_ratio[1]*100:.1f}%)', fontweight='bold')
        ax.set_title('PCA: All Subdomains Pass/Fail', fontweight='bold', fontsize=16)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Calculate overall accuracy
        in_any_subdomain = point_status >= 0
        if in_any_subdomain.sum() > 0:
            pass_count = (point_status == 0).sum()
            total_in_subdomain = in_any_subdomain.sum()
            accuracy = pass_count / total_in_subdomain * 100
            ax.text(0.02, 0.98, f'Overall Accuracy: {accuracy:.1f}%\n({pass_count:,}/{total_in_subdomain:,})',
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(output_dir / "pca_all_subdomains_pass_fail.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_subdomain_comparison(self, output_dir: Path):
        """Plot comparison of subdomain model performance."""
        if not self.results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        subdomains = list(self.results.keys())
        f1_scores = [self.results[s].metrics.get('f1_macro', 0) for s in subdomains]
        roc_aucs = [self.results[s].metrics.get('roc_auc', 0) for s in subdomains]
        
        x = np.arange(len(subdomains))
        
        # F1 scores
        axes[0].bar(x, f1_scores, color='steelblue')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([s.replace('_', '\n') for s in subdomains], fontsize=9)
        axes[0].set_ylabel('F1 Macro Score')
        axes[0].set_title('Model Performance: F1 Score')
        axes[0].set_ylim(0, 1.1)
        
        # ROC-AUC
        axes[1].bar(x, roc_aucs, color='coral')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([s.replace('_', '\n') for s in subdomains], fontsize=9)
        axes[1].set_ylabel('ROC-AUC')
        axes[1].set_title('Model Performance: ROC-AUC')
        axes[1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_dir / "subdomain_comparison.png", dpi=150)
        plt.close()
    
    def _plot_similarity_comparison(self, output_dir: Path):
        """Plot comparison of subdomain similarities."""
        if not self.report.similarity_results:
            return
        
        valid_results = {k: v for k, v in self.report.similarity_results.items() 
                        if v.get('n_samples', 0) > 0}
        
        if not valid_results:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        subdomains = list(valid_results.keys())
        x = np.arange(len(subdomains))
        width = 0.35
        
        cos_means = [valid_results[s].get('cosine_mean', 0) for s in subdomains]
        euc_means = [valid_results[s].get('euclidean_mean', 0) for s in subdomains]
        
        ax.bar(x - width/2, cos_means, width, label='Cosine Similarity', color='steelblue')
        ax.bar(x + width/2, euc_means, width, label='Euclidean Similarity', color='coral')
        
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in subdomains], fontsize=9)
        ax.set_ylabel('Similarity')
        ax.set_title('Intra-Group Similarity by Subdomain')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "similarity_comparison.png", dpi=150)
        plt.close()

    # =========================================================================
    # SENSITIVITY ANALYSIS VISUALIZATION METHODS
    # =========================================================================
    
    def _plot_sensitivity_summary(self, output_dir: Path):
        """Plot summary of sensitivity analysis across all subdomains."""
        if not self.sensitivity_results:
            return
        
        subdomains = list(self.sensitivity_results.keys())
        n_subdomains = len(subdomains)
        
        if n_subdomains == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Bootstrap F1 confidence intervals
        ax1 = axes[0, 0]
        x = np.arange(n_subdomains)
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for sd in subdomains:
            sens = self.sensitivity_results[sd]
            if sens.bootstrap_metrics.get('f1_macro'):
                boot = sens.bootstrap_metrics['f1_macro']
                means.append(boot['mean'])
                ci_lowers.append(boot['mean'] - boot['ci_lower'])
                ci_uppers.append(boot['ci_upper'] - boot['mean'])
            else:
                means.append(0)
                ci_lowers.append(0)
                ci_uppers.append(0)
        
        ax1.bar(x, means, color='steelblue', alpha=0.7)
        ax1.errorbar(x, means, yerr=[ci_lowers, ci_uppers], fmt='none', color='black', capsize=5)
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.replace('_', '\n') for s in subdomains], fontsize=9)
        ax1.set_ylabel('F1 Macro Score')
        ax1.set_title('Bootstrap Confidence Intervals (95%)', fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Perturbation stability scores
        ax2 = axes[0, 1]
        stability_scores = []
        for sd in subdomains:
            sens = self.sensitivity_results[sd]
            score = sens.perturbation_stability.get('overall_stability_score', 0)
            stability_scores.append(score)
        
        colors = ['green' if s >= 0.9 else 'orange' if s >= 0.7 else 'red' for s in stability_scores]
        ax2.bar(x, stability_scores, color=colors, alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.replace('_', '\n') for s in subdomains], fontsize=9)
        ax2.set_ylabel('Stability Score')
        ax2.set_title('Perturbation Stability', fontweight='bold')
        ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='High (≥0.9)')
        ax2.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Medium (≥0.7)')
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: CV stability (coefficient of variation)
        ax3 = axes[1, 0]
        cv_coeffs = []
        for sd in subdomains:
            sens = self.sensitivity_results[sd]
            if sens.cv_stability.get('f1_macro'):
                cv_coeffs.append(sens.cv_stability['f1_macro'].get('cv_coefficient', 0))
            else:
                cv_coeffs.append(0)
        
        colors = ['green' if c <= 0.1 else 'orange' if c <= 0.2 else 'red' for c in cv_coeffs]
        ax3.bar(x, cv_coeffs, color=colors, alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.replace('_', '\n') for s in subdomains], fontsize=9)
        ax3.set_ylabel('Coefficient of Variation')
        ax3.set_title('Cross-Validation Stability', fontweight='bold')
        ax3.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Low (≤0.1)')
        ax3.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Medium (≤0.2)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Threshold robustness (range width)
        ax4 = axes[1, 1]
        robust_widths = []
        for sd in subdomains:
            sens = self.sensitivity_results[sd]
            width = sens.threshold_robustness.get('robust_range_width', 0)
            robust_widths.append(width)
        
        ax4.bar(x, robust_widths, color='purple', alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels([s.replace('_', '\n') for s in subdomains], fontsize=9)
        ax4.set_ylabel('Robust Range Width')
        ax4.set_title('Threshold Robustness', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Sensitivity Analysis Summary', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "sensitivity_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bootstrap_confidence_intervals(self, output_dir: Path):
        """Plot detailed bootstrap confidence intervals for all metrics."""
        if not self.sensitivity_results:
            return
        
        metrics = ['accuracy', 'f1_macro', 'precision', 'recall', 'roc_auc']
        subdomains = list(self.sensitivity_results.keys())
        n_metrics = len(metrics)
        n_subdomains = len(subdomains)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        
        x = np.arange(n_subdomains)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            means = []
            errors = []
            
            for sd in subdomains:
                sens = self.sensitivity_results[sd]
                if sens.bootstrap_metrics.get(metric):
                    boot = sens.bootstrap_metrics[metric]
                    means.append(boot['mean'])
                    errors.append([boot['mean'] - boot['ci_lower'], boot['ci_upper'] - boot['mean']])
                else:
                    means.append(0)
                    errors.append([0, 0])
            
            errors = np.array(errors).T
            ax.barh(x, means, color='steelblue', alpha=0.7)
            ax.errorbar(means, x, xerr=errors, fmt='none', color='black', capsize=3)
            ax.set_yticks(x)
            ax.set_yticklabels([s.replace('_', ' ') for s in subdomains], fontsize=8)
            ax.set_xlabel('Score')
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_xlim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Bootstrap Confidence Intervals (95%)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "bootstrap_confidence_intervals.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_perturbation_stability(self, output_dir: Path):
        """Plot perturbation stability analysis results."""
        if not self.sensitivity_results:
            return
        
        subdomains = list(self.sensitivity_results.keys())
        noise_levels = ['noise_1pct', 'noise_5pct', 'noise_10pct', 'noise_20pct']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Prediction stability across noise levels
        ax1 = axes[0]
        x = np.arange(len(noise_levels))
        width = 0.15
        
        for i, sd in enumerate(subdomains):
            sens = self.sensitivity_results[sd]
            stabilities = []
            for nl in noise_levels:
                if nl in sens.perturbation_stability:
                    stabilities.append(sens.perturbation_stability[nl].get('prediction_stability', 0))
                else:
                    stabilities.append(0)
            
            ax1.bar(x + i * width, stabilities, width, label=sd.replace('_', ' '), alpha=0.8)
        
        ax1.set_xticks(x + width * (len(subdomains) - 1) / 2)
        ax1.set_xticklabels(['1%', '5%', '10%', '20%'])
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Prediction Stability')
        ax1.set_title('Prediction Stability vs Noise Level', fontweight='bold')
        ax1.legend(loc='lower left', fontsize=8)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Probability correlation across noise levels
        ax2 = axes[1]
        
        for i, sd in enumerate(subdomains):
            sens = self.sensitivity_results[sd]
            correlations = []
            for nl in noise_levels:
                if nl in sens.perturbation_stability:
                    correlations.append(sens.perturbation_stability[nl].get('probability_correlation', 0))
                else:
                    correlations.append(0)
            
            ax2.plot(['1%', '5%', '10%', '20%'], correlations, 'o-', label=sd.replace('_', ' '), linewidth=2)
        
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Probability Correlation')
        ax2.set_title('Probability Correlation vs Noise Level', fontweight='bold')
        ax2.legend(loc='lower left', fontsize=8)
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Perturbation Stability Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "perturbation_stability.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_ablation(self, output_dir: Path):
        """Plot feature ablation analysis results."""
        if not self.sensitivity_results:
            return
        
        # Get all features across all subdomains
        all_features = set()
        for sens in self.sensitivity_results.values():
            all_features.update([f for f in sens.feature_ablation.keys() if not f.startswith('_')])
        
        if not all_features:
            return
        
        subdomains = list(self.sensitivity_results.keys())
        features = sorted(list(all_features))
        
        # Create heatmap of feature impacts
        impact_matrix = np.zeros((len(features), len(subdomains)))
        
        for j, sd in enumerate(subdomains):
            sens = self.sensitivity_results[sd]
            for i, feat in enumerate(features):
                if feat in sens.feature_ablation:
                    impact_matrix[i, j] = sens.feature_ablation[feat].get('relative_impact_pct', 0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(impact_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-10, vmax=10)
        
        ax.set_xticks(np.arange(len(subdomains)))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels([s.replace('_', '\n') for s in subdomains], fontsize=10)
        ax.set_yticklabels(features, fontsize=10)
        
        # Add text annotations
        for i in range(len(features)):
            for j in range(len(subdomains)):
                val = impact_matrix[i, j]
                color = 'white' if abs(val) > 5 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Performance Impact (%)')
        ax.set_title('Feature Ablation Analysis\n(% Performance Drop When Feature Removed)', fontweight='bold')
        ax.set_xlabel('Subdomain')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig(output_dir / "feature_ablation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_decision_boundary_analysis(self, output_dir: Path):
        """Plot decision boundary analysis results."""
        if not self.sensitivity_results:
            return
        
        subdomains = list(self.sensitivity_results.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Percentage of samples near boundary
        ax1 = axes[0]
        boundary_thresholds = ['within_10pct_of_boundary', 'within_15pct_of_boundary', 
                               'within_20pct_of_boundary', 'within_25pct_of_boundary']
        
        x = np.arange(len(boundary_thresholds))
        width = 0.15
        
        for i, sd in enumerate(subdomains):
            sens = self.sensitivity_results[sd]
            percentages = []
            for bt in boundary_thresholds:
                if bt in sens.boundary_analysis:
                    percentages.append(sens.boundary_analysis[bt].get('percentage', 0))
                else:
                    percentages.append(0)
            
            ax1.bar(x + i * width, percentages, width, label=sd.replace('_', ' '), alpha=0.8)
        
        ax1.set_xticks(x + width * (len(subdomains) - 1) / 2)
        ax1.set_xticklabels(['±10%', '±15%', '±20%', '±25%'])
        ax1.set_xlabel('Distance from Decision Boundary (prob=0.5)')
        ax1.set_ylabel('Percentage of Samples')
        ax1.set_title('Samples Near Decision Boundary', fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy comparison (boundary vs non-boundary)
        ax2 = axes[1]
        
        boundary_acc = []
        non_boundary_acc = []
        for sd in subdomains:
            sens = self.sensitivity_results[sd]
            if 'within_20pct_of_boundary' in sens.boundary_analysis:
                ba = sens.boundary_analysis['within_20pct_of_boundary']
                boundary_acc.append(ba.get('accuracy', 0) * 100)
                non_boundary_acc.append(ba.get('non_boundary_accuracy', 0) * 100)
            else:
                boundary_acc.append(0)
                non_boundary_acc.append(0)
        
        x = np.arange(len(subdomains))
        width = 0.35
        
        ax2.bar(x - width/2, boundary_acc, width, label='Near Boundary (±20%)', color='coral', alpha=0.8)
        ax2.bar(x + width/2, non_boundary_acc, width, label='Away from Boundary', color='steelblue', alpha=0.8)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.replace('_', '\n') for s in subdomains], fontsize=9)
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy: Boundary vs Non-Boundary', fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.set_ylim(0, 110)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Decision Boundary Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "decision_boundary_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_sensitivity_report(self, output_dir: Path):
        """Save detailed sensitivity analysis report to text file."""
        output_file = output_dir / "sensitivity_analysis_report.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SENSITIVITY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # =====================================================================
            # SUMMARY TABLE - All Subdomains
            # =====================================================================
            f.write("=" * 120 + "\n")
            f.write("COMPREHENSIVE SUMMARY TABLE\n")
            f.write("=" * 120 + "\n\n")
            
            # Header
            f.write(f"{'Subdomain':<25} {'Perturb.':<10} {'Perturb.':<10} {'Perturb.':<10} {'Perturb.':<10} ")
            f.write(f"{'Cohesion':<10} {'Cohesion':<10} {'Z-Score':<10} {'Z-Score':<10} {'CV':<10}\n")
            f.write(f"{'':25} {'1%':<10} {'5%':<10} {'10%':<10} {'20%':<10} ")
            f.write(f"{'Cosine':<10} {'Euclidean':<10} {'F1':<10} {'Acc':<10} {'Coeff':<10}\n")
            f.write("-" * 120 + "\n")
            
            # Data rows
            for subdomain in self.sensitivity_results.keys():
                sens = self.sensitivity_results[subdomain]
                
                # Perturbation stability values
                perturb_1 = sens.perturbation_stability.get('noise_1pct', {}).get('prediction_stability', 0) if sens.perturbation_stability else 0
                perturb_5 = sens.perturbation_stability.get('noise_5pct', {}).get('prediction_stability', 0) if sens.perturbation_stability else 0
                perturb_10 = sens.perturbation_stability.get('noise_10pct', {}).get('prediction_stability', 0) if sens.perturbation_stability else 0
                perturb_20 = sens.perturbation_stability.get('noise_20pct', {}).get('prediction_stability', 0) if sens.perturbation_stability else 0
                
                # Cohesion (from similarity analysis)
                sim_result = self.report.similarity_results.get(subdomain, {})
                cosine_sim = sim_result.get('cosine_mean', 0)
                euclidean_sim = sim_result.get('euclidean_mean', 0)
                
                # Z-scores (computed from bootstrap metrics - deviation from perfect score normalized by std)
                f1_boot = sens.bootstrap_metrics.get('f1_macro', {}) if sens.bootstrap_metrics else {}
                acc_boot = sens.bootstrap_metrics.get('accuracy', {}) if sens.bootstrap_metrics else {}
                
                f1_mean = f1_boot.get('mean', 0)
                f1_std = f1_boot.get('std', 0.001)  # Avoid div by zero
                acc_mean = acc_boot.get('mean', 0)
                acc_std = acc_boot.get('std', 0.001)
                
                # Z-score: how many std devs the mean is from perfect (1.0)
                z_f1 = (1.0 - f1_mean) / f1_std if f1_std > 0 else 0
                z_acc = (1.0 - acc_mean) / acc_std if acc_std > 0 else 0
                
                # CV coefficient
                cv_coeff = sens.cv_stability.get('f1_macro', {}).get('cv_coefficient', 0) if sens.cv_stability else 0
                
                # Format row
                subdomain_name = subdomain.replace('_', ' ')[:24]
                f.write(f"{subdomain_name:<25} {perturb_1:<10.4f} {perturb_5:<10.4f} {perturb_10:<10.4f} {perturb_20:<10.4f} ")
                f.write(f"{cosine_sim:<10.4f} {euclidean_sim:<10.4f} {z_f1:<10.4f} {z_acc:<10.4f} {cv_coeff:<10.4f}\n")
            
            f.write("-" * 120 + "\n\n")
            
            # Additional summary metrics table
            f.write("=" * 100 + "\n")
            f.write("BOOTSTRAP & THRESHOLD SUMMARY TABLE\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"{'Subdomain':<25} {'F1 Mean':<10} {'F1 CI Low':<10} {'F1 CI High':<10} ")
            f.write(f"{'Opt Thresh':<10} {'Robust Width':<12} {'Samples':<10}\n")
            f.write("-" * 100 + "\n")
            
            for subdomain in self.sensitivity_results.keys():
                sens = self.sensitivity_results[subdomain]
                
                f1_boot = sens.bootstrap_metrics.get('f1_macro', {}) if sens.bootstrap_metrics else {}
                f1_mean = f1_boot.get('mean', 0)
                f1_ci_low = f1_boot.get('ci_lower', 0)
                f1_ci_high = f1_boot.get('ci_upper', 0)
                
                opt_thresh = sens.threshold_robustness.get('optimal_threshold', 0.5) if sens.threshold_robustness else 0.5
                robust_width = sens.threshold_robustness.get('robust_range_width', 0) if sens.threshold_robustness else 0
                
                n_samples = sens.boundary_analysis.get('total_samples', 0) if sens.boundary_analysis else 0
                
                subdomain_name = subdomain.replace('_', ' ')[:24]
                f.write(f"{subdomain_name:<25} {f1_mean:<10.4f} {f1_ci_low:<10.4f} {f1_ci_high:<10.4f} ")
                f.write(f"{opt_thresh:<10.3f} {robust_width:<12.4f} {n_samples:<10}\n")
            
            f.write("-" * 100 + "\n\n")
            
            # Legend/Key
            f.write("KEY:\n")
            f.write("-" * 40 + "\n")
            f.write("  Perturb. 1-20%  : Prediction stability at noise levels (higher = more stable)\n")
            f.write("  Cohesion Cosine : Mean cosine similarity within subdomain (higher = more cohesive)\n")
            f.write("  Cohesion Euclid.: Mean Euclidean similarity within subdomain (higher = more cohesive)\n")
            f.write("  Z-Score F1/Acc  : Std deviations from perfect score (lower = closer to perfect)\n")
            f.write("  CV Coeff        : Coefficient of variation in CV (lower = more stable)\n")
            f.write("  Opt Thresh      : Optimal decision threshold for best F1\n")
            f.write("  Robust Width    : Range of thresholds maintaining 95% of optimal F1\n")
            f.write("\n\n")
            
            # =====================================================================
            # DETAILED RESULTS PER SUBDOMAIN
            # =====================================================================
            f.write("=" * 80 + "\n")
            f.write("DETAILED RESULTS BY SUBDOMAIN\n")
            f.write("=" * 80 + "\n")
            
            for subdomain, sens in self.sensitivity_results.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"SUBDOMAIN: {subdomain.upper().replace('_', ' ')}\n")
                f.write(f"{'='*60}\n\n")
                
                # Bootstrap results
                f.write("BOOTSTRAP CONFIDENCE INTERVALS (95%):\n")
                f.write("-" * 40 + "\n")
                if sens.bootstrap_metrics:
                    for metric, values in sens.bootstrap_metrics.items():
                        f.write(f"  {metric:15}: {values['mean']:.4f} [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}]\n")
                        f.write(f"  {'':15}  (std: {values['std']:.4f}, n={values['n_samples']})\n")
                else:
                    f.write("  No bootstrap results available\n")
                f.write("\n")
                
                # Perturbation stability
                f.write("PERTURBATION STABILITY:\n")
                f.write("-" * 40 + "\n")
                if sens.perturbation_stability:
                    f.write(f"  Overall stability score: {sens.perturbation_stability.get('overall_stability_score', 0):.4f}\n\n")
                    for key, value in sens.perturbation_stability.items():
                        if key.startswith('noise_'):
                            f.write(f"  {key}:\n")
                            f.write(f"    Prediction stability: {value['prediction_stability']:.4f}\n")
                            f.write(f"    Probability MAE: {value['probability_mae']:.4f}\n")
                            f.write(f"    Probability correlation: {value['probability_correlation']:.4f}\n")
                else:
                    f.write("  No perturbation results available\n")
                f.write("\n")
                
                # Feature ablation
                f.write("FEATURE ABLATION ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                if sens.feature_ablation:
                    baseline = sens.feature_ablation.get('_baseline', {})
                    if baseline:
                        f.write(f"  Baseline F1: {baseline.get('f1_macro_mean', 0):.4f} ± {baseline.get('f1_macro_std', 0):.4f}\n\n")
                    
                    # Sort by impact
                    feature_impacts = [(f, v) for f, v in sens.feature_ablation.items() if not f.startswith('_')]
                    feature_impacts.sort(key=lambda x: abs(x[1].get('impact', 0)), reverse=True)
                    
                    f.write(f"  {'Feature':15} {'Impact':>10} {'Rel. Impact':>12} {'F1 Without':>12}\n")
                    f.write(f"  {'-'*50}\n")
                    for feat, values in feature_impacts:
                        f.write(f"  {feat:15} {values.get('impact', 0):>10.4f} {values.get('relative_impact_pct', 0):>11.2f}% {values.get('f1_macro_without', 0):>12.4f}\n")
                else:
                    f.write("  No feature ablation results available\n")
                f.write("\n")
                
                # CV stability
                f.write("CROSS-VALIDATION STABILITY:\n")
                f.write("-" * 40 + "\n")
                if sens.cv_stability:
                    for metric, values in sens.cv_stability.items():
                        f.write(f"  {metric}:\n")
                        f.write(f"    Mean: {values['mean']:.4f}, Std: {values['std']:.4f}\n")
                        f.write(f"    CV Coefficient: {values['cv_coefficient']:.4f}\n")
                        f.write(f"    Range: [{values['min']:.4f}, {values['max']:.4f}]\n")
                else:
                    f.write("  No CV stability results available\n")
                f.write("\n")
                
                # Threshold robustness
                f.write("THRESHOLD ROBUSTNESS:\n")
                f.write("-" * 40 + "\n")
                if sens.threshold_robustness:
                    f.write(f"  Optimal threshold: {sens.threshold_robustness.get('optimal_threshold', 0.5):.3f}\n")
                    f.write(f"  Optimal F1: {sens.threshold_robustness.get('optimal_f1', 0):.4f}\n")
                    robust_range = sens.threshold_robustness.get('robust_threshold_range', [0.5, 0.5])
                    f.write(f"  Robust range (F1 ≥ 95% of optimal): [{robust_range[0]:.3f}, {robust_range[1]:.3f}]\n")
                    f.write(f"  Robust range width: {sens.threshold_robustness.get('robust_range_width', 0):.3f}\n")
                    f.write(f"  F1 at default (0.5): {sens.threshold_robustness.get('f1_at_0.5', 0):.4f}\n")
                    f.write(f"  F1 variance: {sens.threshold_robustness.get('f1_variance', 0):.6f}\n")
                else:
                    f.write("  No threshold robustness results available\n")
                f.write("\n")
                
                # Decision boundary
                f.write("DECISION BOUNDARY ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                if sens.boundary_analysis:
                    f.write(f"  Total samples: {sens.boundary_analysis.get('total_samples', 0)}\n\n")
                    for key in ['within_10pct_of_boundary', 'within_15pct_of_boundary', 
                               'within_20pct_of_boundary', 'within_25pct_of_boundary']:
                        if key in sens.boundary_analysis:
                            ba = sens.boundary_analysis[key]
                            f.write(f"  {key}:\n")
                            f.write(f"    Samples: {ba.get('n_samples', 0)} ({ba.get('percentage', 0):.2f}%)\n")
                            if 'accuracy' in ba:
                                f.write(f"    Boundary accuracy: {ba['accuracy']*100:.2f}%\n")
                                f.write(f"    Non-boundary accuracy: {ba['non_boundary_accuracy']*100:.2f}%\n")
                else:
                    f.write("  No decision boundary results available\n")
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF SENSITIVITY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"  Sensitivity analysis report saved to {output_file}")

    def _plot_threshold_sweep(self, result: ModelResult, output_dir: Path):
        """Sweep decision thresholds and observe FP/FN/precision/recall behavior."""
        subdomain_name_display = result.subdomain_name.replace("_", " ").title()
        
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
                 'b-', label='Precision', linewidth=2)
        ax1.plot(metrics_by_threshold['threshold'], metrics_by_threshold['recall'], 
                 'r-', label='Recall', linewidth=2)
        ax1.plot(metrics_by_threshold['threshold'], metrics_by_threshold['f1'], 
                 'g-', label='F1 Score', linewidth=2)
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
        ax1.set_xlabel('Decision Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Precision/Recall/F1 vs Threshold: {subdomain_name_display}')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: FP and FN counts vs Threshold
        ax2 = axes[0, 1]
        ax2.plot(metrics_by_threshold['threshold'], metrics_by_threshold['fp_count'], 
                 'r-', label='False Positives', linewidth=2)
        ax2.plot(metrics_by_threshold['threshold'], metrics_by_threshold['fn_count'], 
                 'b-', label='False Negatives', linewidth=2)
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Decision Threshold')
        ax2.set_ylabel('Count')
        ax2.set_title(f'FP/FN Counts vs Threshold: {subdomain_name_display}')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: FP Rate vs Threshold
        ax3 = axes[1, 0]
        ax3.plot(metrics_by_threshold['threshold'], metrics_by_threshold['fp_rate'], 
                 'r-', linewidth=2)
        ax3.fill_between(metrics_by_threshold['threshold'], metrics_by_threshold['fp_rate'], 
                         alpha=0.3, color='red')
        ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Decision Threshold')
        ax3.set_ylabel('False Positive Rate')
        ax3.set_title(f'FP Rate vs Threshold: {subdomain_name_display}')
        ax3.grid(True, alpha=0.3)
        
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
        ax4.scatter(metrics_by_threshold['fp_rate'], metrics_by_threshold['recall'], 
                   c=metrics_by_threshold['threshold'], cmap='viridis', s=50)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('Recall (True Positive Rate)')
        ax4.set_title(f'FPR vs Recall Trade-off: {subdomain_name_display}')
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Threshold')
        ax4.grid(True, alpha=0.3)
        
        # Add text annotations
        textstr = f'Best F1 Threshold: {best_f1_thresh:.2f}\n'
        textstr += f'Best F1 Score: {metrics_by_threshold["f1"][best_f1_idx]:.3f}\n'
        if low_fp_thresh:
            textstr += f'Low FP (<10%) Threshold: {low_fp_thresh:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax4.text(0.95, 0.05, textstr, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"threshold_sweep_{result.subdomain_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save threshold analysis to text file
        with open(output_dir / f"threshold_analysis_{result.subdomain_name}.txt", 'w') as f:
            f.write(f"Threshold Analysis for {subdomain_name_display}\n")
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
        subdomain_name_display = result.subdomain_name.replace("_", " ").title()
        
        y_true = result.y_true
        y_proba_pos = result.y_proba[:, 1]
        
        # Compute calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_proba_pos, n_bins=10, strategy='uniform')
        except:
            self.logger.warning(f"Could not compute calibration curve for {result.subdomain_name}")
            return
        
        # Compute Brier score
        brier = brier_score_loss(y_true, y_proba_pos)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Calibration curve
        ax1 = axes[0]
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax1.plot(prob_pred, prob_true, 'bo-', label=f'Model (Brier={brier:.3f})')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'Calibration Curve: {subdomain_name_display}')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram of predicted probabilities
        ax2 = axes[1]
        ax2.hist(y_proba_pos[y_true == 0], bins=20, alpha=0.5, label='Negative class', color='blue')
        ax2.hist(y_proba_pos[y_true == 1], bins=20, alpha=0.5, label='Positive class', color='red')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Probability Distribution by Class: {subdomain_name_display}')
        ax2.legend(loc='best')
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"calibration_{result.subdomain_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Add calibration info to metrics
        result.metrics['brier_score'] = float(brier)

    def _analyze_false_positives(self, result: ModelResult, df: pd.DataFrame, output_dir: Path):
        """Slice false positives by meaningful subgroups to understand FP behavior."""
        subdomain_name_display = result.subdomain_name.replace("_", " ").title()
        
        # Identify FP cases
        fp_mask = (result.y_true == 0) & (result.y_pred == 1)
        tn_mask = (result.y_true == 0) & (result.y_pred == 0)
        
        fp_count = fp_mask.sum()
        if fp_count == 0:
            self.logger.info(f"No false positives for {result.subdomain_name}")
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
                
                ax.hist(tn_data, bins=20, alpha=0.5, label=f'TN (n={len(tn_data)})', color='green')
                ax.hist(fp_data, bins=20, alpha=0.5, label=f'FP (n={len(fp_data)})', color='red')
                ax.set_xlabel(feature)
                ax.set_ylabel('Count')
                ax.set_title(f'{feature}\nΔmean={stats["diff_means"]:.3f}')
                ax.legend(loc='best', fontsize=8)
            
            # Hide unused subplots
            for j in range(top_n, len(axes)):
                axes[j].set_visible(False)
            
            plt.suptitle(f'FP vs TN Feature Distributions: {subdomain_name_display}', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / f"fp_analysis_{result.subdomain_name}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save FP analysis report
        with open(output_dir / f"fp_analysis_{result.subdomain_name}.txt", 'w', encoding='utf-8') as f:
            f.write(f"False Positive Analysis for {subdomain_name_display}\n")
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
        subdomain_name_display = result.subdomain_name.replace("_", " ").title()
        
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
            
            features = [x[0] for x in sorted_contributions[:top_n]]
            contributions = [x[1] for x in sorted_contributions[:top_n]]
            colors = ['red' if c > 0 else 'blue' for c in contributions]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, contributions, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            ax.set_xlabel('Contribution (importance × z-score)')
            ax.set_title(f'Feature Attribution for False Positives: {subdomain_name_display}')
            ax.axvline(x=0, color='black', linewidth=0.5)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='Pushes toward FP (higher in FP)'),
                Patch(facecolor='blue', label='Pushes toward TN (lower in FP)')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"fp_attribution_{result.subdomain_name}.png", dpi=150, bbox_inches='tight')
            plt.close()

    def generate_full_report_text(self, df: pd.DataFrame, output_dir: Path):
        """Generate a comprehensive text file with all analysis data."""
        output_file = output_dir / "full_analysis_report.txt"
        self.logger.info(f"Generating comprehensive text report: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("PHORCE SUBDOMAIN-LEVEL XGBOOST ANALYSIS - FULL REPORT\n")
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
            
            f.write("Subdomains Analyzed:\n")
            for subdomain in self.report.subdomains_analyzed:
                info = self.report.subdomain_info.get(subdomain, {})
                count = info.get('count', 0)
                pct = info.get('percentage', 0)
                parent = info.get('parent_domain', 'N/A')
                f.write(f"  - {subdomain}: {count:,} compounds ({pct:.2f}%) [Parent: {parent}]\n")
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
                for subdomain, sim_data in self.report.similarity_results.items():
                    f.write(f"{subdomain}:\n")
                    if sim_data.get('n_samples', 0) > 0:
                        f.write(f"  Samples: {sim_data['n_samples']:,}\n")
                        f.write(f"  Cosine Similarity: {sim_data['cosine_mean']:.4f} ± {sim_data['cosine_std']:.4f}\n")
                        f.write(f"  Euclidean Similarity: {sim_data['euclidean_mean']:.4f} ± {sim_data['euclidean_std']:.4f}\n")
                    else:
                        f.write("  Insufficient samples for analysis\n")
                    f.write("\n")
            
            # Model Results for Each Subdomain
            for subdomain, result in self.results.items():
                f.write("=" * 80 + "\n")
                f.write(f"MODEL RESULTS: {subdomain.upper().replace('_', ' ')}\n")
                if result.parent_domain:
                    f.write(f"Parent Domain: {result.parent_domain}\n")
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
                cm = confusion_matrix(result.y_true, result.y_pred)
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
            f.write("-" * 80 + "\n")
            f.write(f"{'Subdomain':25} {'F1 Macro':>12} {'ROC-AUC':>12} {'Precision':>12} {'Recall':>12}\n")
            f.write("-" * 80 + "\n")
            
            for subdomain, result in self.results.items():
                metrics = result.metrics
                f.write(f"{subdomain:25} {metrics.get('f1_macro', 0):12.4f} {metrics.get('roc_auc', 0):12.4f} "
                       f"{metrics.get('precision', 0):12.4f} {metrics.get('recall', 0):12.4f}\n")
            f.write("\n")
            
            # Top Features Across All Subdomains
            f.write("Top Features by Subdomain:\n")
            f.write("-" * 80 + "\n")
            for subdomain, result in self.results.items():
                top_3 = list(result.feature_importances.items())[:3]
                top_str = ", ".join([f"{f}({v:.3f})" for f, v in top_3])
                f.write(f"  {subdomain}: {top_str}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"Full analysis report saved to {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Subdomain-level decision tree analysis")
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
    analysis_config = config.get("subdomain_analysis", {})
    
    input_file = args.input or analysis_config.get("input_file", "data/subsets/P1M_with_subsets.csv")
    output_dir = args.output or analysis_config.get("output_dir", "data/analysis/subdomains")
    report_output = analysis_config.get("report_output", "data/analysis/subdomains/report.json")
    
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
    analyzer = SubdomainAnalyzer(config, logger)
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
    print("SUBDOMAIN ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\n  Compounds Analyzed: {analyzer.report.compounds_after_missing_removal:,}")
    print(f"  Subdomains: {analyzer.report.subdomains_analyzed}")
    
    print("\n  Subdomain Counts:")
    for subdomain, info in analyzer.report.subdomain_info.items():
        parent = f" (within {info['parent_domain']})" if info['parent_domain'] else " (global)"
        print(f"    {subdomain}{parent}: {info['count']:,} ({info['percentage']:.2f}%)")
    
    print("\n  Model Performance:")
    for subdomain, model_info in analyzer.report.models.items():
        metrics = model_info.get("metrics", {})
        print(f"\n    {subdomain}:")
        print(f"      F1 Macro: {metrics.get('f1_macro', 0):.4f}")
        print(f"      ROC-AUC:  {metrics.get('roc_auc', 0):.4f}")
        
        top_features = list(model_info.get("feature_importances", {}).items())[:3]
        if top_features:
            print(f"      Top Features: {', '.join([f'{f[0]}({f[1]:.2f})' for f in top_features])}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
