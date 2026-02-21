#!/usr/bin/env python3
"""
PHORCE Post-Processed Data Analysis Script

This script performs exploratory analysis on the preprocessed data BEFORE
domain labeling. It provides insights into the overall feature distributions,
correlations, and data quality.

Pipeline Stage: 1 (After preprocessing, before domain labeling)
Input: data/processed/P1M_preprocessed.csv
Output: data/analysis/processed/

Usage:
    python scripts/analyze_processed.py
    python scripts/analyze_processed.py --config custom_config.json
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set matplotlib backend for non-interactive environments
plt.switch_backend('Agg')

# Set global font settings: Arial font, size 20
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("phorce_processed_analysis")
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
class ProcessedDataReport:
    """Report for processed data analysis."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    input_file: str = ""
    output_dir: str = ""
    total_compounds: int = 0
    total_features: int = 0
    numeric_features: List[str] = field(default_factory=list)
    missing_value_summary: Dict[str, Any] = field(default_factory=dict)
    feature_statistics: Dict[str, Any] = field(default_factory=dict)
    correlation_analysis: Dict[str, Any] = field(default_factory=dict)
    pca_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return _to_json_serializable({
            "timestamp": self.timestamp,
            "input_file": self.input_file,
            "output_dir": self.output_dir,
            "total_compounds": self.total_compounds,
            "total_features": self.total_features,
            "numeric_features": self.numeric_features,
            "missing_value_summary": self.missing_value_summary,
            "feature_statistics": self.feature_statistics,
            "correlation_analysis": self.correlation_analysis,
            "pca_results": self.pca_results
        })


# =============================================================================
# PROCESSED DATA ANALYZER
# =============================================================================

class ProcessedDataAnalyzer:
    """
    Analyzer for post-processed data before domain labeling.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("phorce_processed_analysis")
        self.report = ProcessedDataReport()
        self.scaler = StandardScaler()
        
    def compute_feature_statistics(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Compute descriptive statistics for numeric features."""
        stats = {}
        
        for col in numeric_cols:
            if col in df.columns:
                data = pd.to_numeric(df[col], errors='coerce')
                stats[col] = {
                    "count": int(data.notna().sum()),
                    "missing": int(data.isna().sum()),
                    "missing_pct": float(data.isna().mean() * 100),
                    "mean": float(data.mean()) if data.notna().any() else None,
                    "std": float(data.std()) if data.notna().any() else None,
                    "min": float(data.min()) if data.notna().any() else None,
                    "max": float(data.max()) if data.notna().any() else None,
                    "median": float(data.median()) if data.notna().any() else None,
                    "q25": float(data.quantile(0.25)) if data.notna().any() else None,
                    "q75": float(data.quantile(0.75)) if data.notna().any() else None,
                }
        
        return stats
    
    def compute_correlations(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Compute correlation matrix and find highly correlated pairs."""
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Find highly correlated pairs (|r| > 0.7)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlation_pairs": sorted(high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)
        }
    
    def perform_pca(self, df: pd.DataFrame, numeric_cols: List[str], n_components: int = 5) -> Dict[str, Any]:
        """Perform PCA on numeric features."""
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(numeric_df) < 10:
            self.logger.warning("Insufficient data for PCA")
            return {}
        
        n_components = min(n_components, len(numeric_cols), len(numeric_df))
        
        X_scaled = self.scaler.fit_transform(numeric_df)
        pca = PCA(n_components=n_components)
        pca_coords = pca.fit_transform(X_scaled)
        
        loadings = {}
        for i, col in enumerate(numeric_cols):
            loadings[col] = pca.components_[:, i].tolist()
        
        return {
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "loadings": loadings,
            "n_samples_used": len(numeric_df)
        }
    
    def plot_feature_distributions(self, df: pd.DataFrame, numeric_cols: List[str], output_dir: Path):
        """Plot histograms for numeric features."""
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        for i, col in enumerate(numeric_cols):
            if col in df.columns:
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(data) > 0:
                    axes[i].hist(data, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
                    axes[i].set_title(col, fontsize=10)
                    axes[i].set_xlabel('')
                    axes[i].set_ylabel('Count')
        
        # Hide empty axes
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        output_path = output_dir / "feature_distributions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  Saved feature distributions: {output_path}")
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, numeric_cols: List[str], output_dir: Path):
        """Plot correlation heatmap."""
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            fmt=".2f", 
            cmap='RdBu_r',
            center=0,
            vmin=-1, 
            vmax=1,
            square=True,
            ax=ax,
            annot_kws={"size": 16}  # Slightly smaller annotation text
        )
        ax.set_title('Pearson Correlation Matrix', fontsize=26)
        
        plt.tight_layout()
        output_path = output_dir / "correlation_heatmap.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  Saved correlation heatmap: {output_path}")
    
    def plot_pca_variance(self, pca_results: Dict[str, Any], output_dir: Path):
        """Plot PCA explained variance as separate figures."""
        if not pca_results:
            return
        
        n_components = pca_results["n_components"]
        variance = pca_results["explained_variance_ratio"]
        cumulative = pca_results["cumulative_variance"]
        
        # Individual variance (separate figure)
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.bar(range(1, n_components + 1), variance, color='steelblue', edgecolor='white')
        ax1.set_xlabel('Principal Component', fontsize=20)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=20)
        ax1.set_title('Variance Explained by Each PC', fontsize=26)
        ax1.set_xticks(range(1, n_components + 1))
        plt.tight_layout()
        output_path1 = output_dir / "pca_variance_individual.png"
        plt.savefig(output_path1, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        self.logger.info(f"  Saved PCA individual variance plot: {output_path1}")
        
        # Cumulative variance (separate figure)
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.plot(range(1, n_components + 1), cumulative, 'bo-', linewidth=2, markersize=8)
        ax2.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        ax2.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Components', fontsize=20)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=20)
        ax2.set_title('Cumulative Variance Explained', fontsize=26)
        ax2.set_xticks(range(1, n_components + 1))
        ax2.legend()
        ax2.set_ylim(0, 1.05)
        plt.tight_layout()
        output_path2 = output_dir / "pca_variance_cumulative.png"
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        self.logger.info(f"  Saved PCA cumulative variance plot: {output_path2}")
    
    def plot_pca_scatter(self, df: pd.DataFrame, numeric_cols: List[str], output_dir: Path):
        """Plot PCA scatter (PC1 vs PC2)."""
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(numeric_df) < 10:
            return
        
        X_scaled = self.scaler.fit_transform(numeric_df)
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sample if too large
        if len(pca_coords) > 10000:
            idx = np.random.choice(len(pca_coords), 10000, replace=False)
            plot_coords = pca_coords[idx]
        else:
            plot_coords = pca_coords
        
        ax.scatter(plot_coords[:, 0], plot_coords[:, 1], alpha=0.5, s=10, c='steelblue')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('PCA: All Compounds')
        
        plt.tight_layout()
        output_path = output_dir / "pca_scatter.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  Saved PCA scatter plot: {output_path}")
    
    def plot_missing_values(self, df: pd.DataFrame, numeric_cols: List[str], output_dir: Path):
        """Plot missing value summary."""
        missing_counts = df[numeric_cols].isna().sum()
        missing_pct = (missing_counts / len(df) * 100).sort_values(ascending=True)
        
        if missing_pct.max() == 0:
            self.logger.info("  No missing values to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(numeric_cols) * 0.3)))
        
        colors = ['#d73027' if pct > 50 else '#fc8d59' if pct > 20 else '#91bfdb' for pct in missing_pct]
        missing_pct.plot(kind='barh', ax=ax, color=colors)
        ax.set_xlabel('Missing Values (%)', fontsize=20)
        ax.set_title('Missing Values by Feature', fontsize=26)
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
        
        plt.tight_layout()
        output_path = output_dir / "missing_values.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  Saved missing values plot: {output_path}")
    
    def analyze(self, df: pd.DataFrame) -> ProcessedDataReport:
        """Run the complete analysis pipeline."""
        self.logger.info("Starting post-processed data analysis...")
        
        # Get config
        analysis_config = self.config.get("processed_analysis", {})
        numeric_cols = analysis_config.get("numeric_features", [
            "mw", "xlogp", "polararea", "complexity", "heavycnt",
            "hbonddonor", "hbondacc", "rotbonds", "charge"
        ])
        
        # Filter to existing columns
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        self.report.total_compounds = len(df)
        self.report.total_features = len(df.columns)
        self.report.numeric_features = numeric_cols
        
        self.logger.info(f"Analyzing {len(df):,} compounds with {len(numeric_cols)} numeric features")
        
        # Compute statistics
        self.logger.info("Computing feature statistics...")
        self.report.feature_statistics = self.compute_feature_statistics(df, numeric_cols)
        
        # Missing value summary
        missing_summary = {
            col: {
                "count": int(df[col].isna().sum()),
                "percentage": float(df[col].isna().mean() * 100)
            }
            for col in numeric_cols if col in df.columns
        }
        self.report.missing_value_summary = missing_summary
        
        # Correlation analysis
        self.logger.info("Computing correlations...")
        self.report.correlation_analysis = self.compute_correlations(df, numeric_cols)
        
        # PCA
        self.logger.info("Performing PCA...")
        n_components = analysis_config.get("n_pca_components", 5)
        self.report.pca_results = self.perform_pca(df, numeric_cols, n_components)
        
        return self.report
    
    def generate_visualizations(self, df: pd.DataFrame, output_dir: Path):
        """Generate all visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        numeric_cols = self.report.numeric_features
        
        self.logger.info("Generating visualizations...")
        self.plot_feature_distributions(df, numeric_cols, output_dir)
        self.plot_correlation_heatmap(df, numeric_cols, output_dir)
        self.plot_missing_values(df, numeric_cols, output_dir)
        self.plot_pca_variance(self.report.pca_results, output_dir)
        self.plot_pca_scatter(df, numeric_cols, output_dir)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze post-processed data")
    parser.add_argument("--config", "-c", default="preprocessing_config.json", help="Config file path")
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
    analysis_config = config.get("processed_analysis", {})
    
    input_file = args.input or analysis_config.get("input_file", "data/processed/P1M_preprocessed.csv")
    output_dir = args.output or analysis_config.get("output_dir", "data/analysis/processed")
    report_output = analysis_config.get("report_output", "data/analysis/processed/report.json")
    
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
    analyzer = ProcessedDataAnalyzer(config, logger)
    analyzer.report.input_file = str(input_path)
    analyzer.report.output_dir = str(output_path)
    
    report = analyzer.analyze(df)
    
    # Generate visualizations
    analyzer.generate_visualizations(df, output_path)
    
    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Saved report to {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("POST-PROCESSED DATA ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\n  Total Compounds: {report.total_compounds:,}")
    print(f"  Numeric Features: {len(report.numeric_features)}")
    
    if report.pca_results:
        print(f"\n  PCA Variance (first 3 components):")
        for i, var in enumerate(report.pca_results.get("explained_variance_ratio", [])[:3]):
            print(f"    PC{i+1}: {var*100:.1f}%")
    
    high_corr = report.correlation_analysis.get("high_correlation_pairs", [])[:5]
    if high_corr:
        print(f"\n  Highly Correlated Feature Pairs (|r| > 0.7):")
        for pair in high_corr:
            print(f"    {pair['feature_1']} ↔ {pair['feature_2']}: {pair['correlation']:.3f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
