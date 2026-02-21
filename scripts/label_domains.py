#!/usr/bin/env python3
"""
PHORCE Domain Labeling Script

This script reads preprocessed chemical compound data and labels compounds
based on keyword matches in text fields. It creates domain-specific subsets
(e.g., soil-related, water-related, crop-related compounds).

The labeling configuration is defined in the preprocessing_config.json file
under the "domain_labeling" section.

Usage:
    python scripts/label_domains.py
    python scripts/label_domains.py --config custom_config.json
    python scripts/label_domains.py --input processed_data.csv --output labeled_data.csv
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set matplotlib backend for non-interactive environments
plt.switch_backend('Agg')


class DomainLabelingReport:
    """Class to track and report domain labeling statistics."""
    
    def __init__(self):
        self.total_compounds = 0
        self.domain_counts = {}
        self.domain_percentages = {}
        self.overlap_matrix = {}
        self.unlabeled_count = 0
        self.multi_domain_count = 0
        self.keyword_match_counts = {}
        self.start_time = datetime.now()
        self.end_time = None
        
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        self.end_time = datetime.now()
        return {
            "labeling_summary": {
                "total_compounds": self.total_compounds,
                "unlabeled_compounds": self.unlabeled_count,
                "unlabeled_percentage": round(self.unlabeled_count / self.total_compounds * 100, 2) if self.total_compounds > 0 else 0,
                "multi_domain_compounds": self.multi_domain_count,
                "multi_domain_percentage": round(self.multi_domain_count / self.total_compounds * 100, 2) if self.total_compounds > 0 else 0,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds()
            },
            "domain_statistics": {
                domain: {
                    "count": count,
                    "percentage": self.domain_percentages.get(domain, 0)
                }
                for domain, count in self.domain_counts.items()
            },
            "domain_overlap_matrix": self.overlap_matrix,
            "keyword_match_statistics": self.keyword_match_counts
        }
    
    def save(self, filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class DomainLabeler:
    """Main class for labeling chemical compounds by domain keywords."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config.get("domain_labeling", {})
        self.logger = logger
        self.report = DomainLabelingReport()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load preprocessed data from CSV file."""
        self.logger.info(f"Loading data from {filepath}")
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise
            
        self.report.total_compounds = len(df)
        self.logger.info(f"Loaded {len(df)} compounds with {len(df.columns)} columns")
        
        return df
    
    def combine_text_fields(self, df: pd.DataFrame) -> pd.Series:
        """Combine specified text columns into a single searchable text field."""
        text_columns = self.config.get("text_columns", [
            "meshheadings", "annotation", "cmpdsynonym", "annothits", "aids", "sidsrcname"
        ])
        
        # Filter to columns that exist in the dataframe
        existing_cols = [col for col in text_columns if col in df.columns]
        
        if not existing_cols:
            self.logger.warning(f"None of the specified text columns found in data: {text_columns}")
            return pd.Series([""] * len(df), index=df.index)
        
        missing_cols = set(text_columns) - set(existing_cols)
        if missing_cols:
            self.logger.warning(f"Some text columns not found in data: {missing_cols}")
        
        self.logger.info(f"Combining text from columns: {existing_cols}")
        
        # Combine text fields
        combined = df[existing_cols].fillna('').astype(str).agg(' '.join, axis=1)
        
        # Apply case transformation if configured
        if not self.config.get("case_sensitive", False):
            combined = combined.str.lower()
        
        return combined
    
    def create_keyword_matcher(self, keywords: List[str], options: dict) -> callable:
        """Create a function to match keywords in text."""
        match_whole_word = options.get("match_whole_word", False)
        use_regex = options.get("use_regex", False)
        case_sensitive = self.config.get("case_sensitive", False)
        
        if not case_sensitive:
            keywords = [k.lower() for k in keywords]
        
        if use_regex:
            # Compile regex patterns
            patterns = []
            for kw in keywords:
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    patterns.append(re.compile(kw, flags))
                except re.error as e:
                    self.logger.warning(f"Invalid regex pattern '{kw}': {e}")
            
            def regex_matcher(text: str) -> int:
                if pd.isna(text) or not isinstance(text, str):
                    return 0
                return int(any(p.search(text) for p in patterns))
            
            return regex_matcher
        
        elif match_whole_word:
            # Word boundary matching
            word_patterns = []
            for kw in keywords:
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    word_patterns.append(re.compile(rf'\b{re.escape(kw)}\b', flags))
                except re.error as e:
                    self.logger.warning(f"Invalid pattern for keyword '{kw}': {e}")
            
            def word_matcher(text: str) -> int:
                if pd.isna(text) or not isinstance(text, str):
                    return 0
                return int(any(p.search(text) for p in word_patterns))
            
            return word_matcher
        
        else:
            # Simple substring matching (fastest)
            def substring_matcher(text: str) -> int:
                if pd.isna(text) or not isinstance(text, str):
                    return 0
                return int(any(kw in text for kw in keywords))
            
            return substring_matcher
    
    def count_keyword_matches(self, text: str, keywords: List[str], case_sensitive: bool = False) -> Dict[str, int]:
        """Count matches for each keyword in text."""
        if pd.isna(text) or not isinstance(text, str):
            return {kw: 0 for kw in keywords}
        
        if not case_sensitive:
            text = text.lower()
            keywords = [k.lower() for k in keywords]
        
        return {kw: text.count(kw) for kw in keywords}
    
    def label_domains(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply domain labels to the dataframe."""
        self.logger.info("Starting domain labeling...")
        
        # Combine text fields
        combined_text = self.combine_text_fields(df)
        
        # Optionally save combined text column
        if self.config.get("save_combined_text_column", False):
            df["combined_text"] = combined_text
        
        # Get labeling options
        labeling_options = self.config.get("labeling_options", {})
        min_matches = labeling_options.get("minimum_keyword_matches", 1)
        
        # Process each domain
        domains_config = self.config.get("domains", {})
        enabled_domains = []
        
        for domain_name, domain_config in domains_config.items():
            if not domain_config.get("enabled", True):
                self.logger.info(f"Skipping disabled domain: {domain_name}")
                continue
            
            enabled_domains.append(domain_name)
            keywords = domain_config.get("keywords", [])
            
            if not keywords:
                self.logger.warning(f"No keywords defined for domain: {domain_name}")
                df[domain_name] = 0
                continue
            
            self.logger.info(f"Labeling domain '{domain_name}' with {len(keywords)} keywords")
            
            # Create matcher function
            matcher = self.create_keyword_matcher(keywords, labeling_options)
            
            # Apply labeling
            df[domain_name] = combined_text.apply(matcher)
            
            # Count matches
            domain_count = df[domain_name].sum()
            self.report.domain_counts[domain_name] = int(domain_count)
            self.report.domain_percentages[domain_name] = round(domain_count / len(df) * 100, 2)
            
            self.logger.info(f"  {domain_name}: {domain_count} compounds ({self.report.domain_percentages[domain_name]}%)")
            
            # Track keyword-level statistics (sample for performance)
            if len(df) > 10000:
                sample_texts = combined_text.sample(min(1000, len(df)), random_state=42)
            else:
                sample_texts = combined_text
            
            keyword_counts = {}
            for kw in keywords:
                kw_lower = kw.lower() if not self.config.get("case_sensitive", False) else kw
                count = sample_texts.str.contains(kw_lower, regex=False, na=False).sum()
                keyword_counts[kw] = int(count)
            
            self.report.keyword_match_counts[domain_name] = keyword_counts
        
        # Calculate overlap matrix
        if len(enabled_domains) > 1:
            self.logger.info("Calculating domain overlap matrix...")
            overlap_matrix = {}
            
            for d1 in enabled_domains:
                overlap_matrix[d1] = {}
                for d2 in enabled_domains:
                    if d1 == d2:
                        overlap_matrix[d1][d2] = int(df[d1].sum())
                    else:
                        overlap = ((df[d1] == 1) & (df[d2] == 1)).sum()
                        overlap_matrix[d1][d2] = int(overlap)
            
            self.report.overlap_matrix = overlap_matrix
        
        # Count unlabeled compounds
        if enabled_domains:
            unlabeled_mask = df[enabled_domains].sum(axis=1) == 0
            self.report.unlabeled_count = int(unlabeled_mask.sum())
            
            # Count multi-domain compounds
            multi_domain_mask = df[enabled_domains].sum(axis=1) > 1
            self.report.multi_domain_count = int(multi_domain_mask.sum())
        
        return df
    
    def save_subsets(self, df: pd.DataFrame, output_dir: Path):
        """Save domain-specific subsets to separate files."""
        domains_config = self.config.get("domains", {})
        labeling_options = self.config.get("labeling_options", {})
        
        enabled_domains = []
        
        for domain_name, domain_config in domains_config.items():
            if not domain_config.get("enabled", True):
                continue
            
            enabled_domains.append(domain_name)
            
            if domain_config.get("save_subset", True):
                subset_filename = domain_config.get("subset_filename", f"{domain_name}_compounds.csv")
                subset_path = output_dir / subset_filename
                
                # Filter to this domain
                subset = df[df[domain_name] == 1].copy()
                
                if len(subset) > 0:
                    subset.to_csv(subset_path, index=False)
                    self.logger.info(f"Saved {len(subset)} {domain_name} compounds to {subset_path}")
                else:
                    self.logger.warning(f"No compounds found for domain {domain_name}, skipping subset file")
        
        # Save multi-domain subset if configured
        if labeling_options.get("create_multi_domain_subset", True) and len(enabled_domains) > 1:
            multi_domain_mask = df[enabled_domains].sum(axis=1) > 1
            multi_domain_df = df[multi_domain_mask].copy()
            
            if len(multi_domain_df) > 0:
                multi_filename = labeling_options.get("multi_domain_filename", "multi_domain_compounds.csv")
                multi_path = output_dir / multi_filename
                multi_domain_df.to_csv(multi_path, index=False)
                self.logger.info(f"Saved {len(multi_domain_df)} multi-domain compounds to {multi_path}")
    
    def generate_correlation_heatmaps(self, df: pd.DataFrame, output_dir: Path):
        """Generate correlation heatmaps for each domain."""
        viz_config = self.config.get("visualization", {})
        corr_config = viz_config.get("correlation_heatmaps", {})
        
        if not corr_config.get("enabled", True):
            return
        
        self.logger.info("Generating correlation heatmaps...")
        
        # Get feature columns
        features = corr_config.get("features", [
            "mw", "xlogp", "polararea", "complexity", "rotbonds", 
            "hbonddonor", "hbondacc", "heavycnt", "charge"
        ])
        
        # Filter to existing columns
        existing_features = [f for f in features if f in df.columns]
        
        if len(existing_features) < 2:
            self.logger.warning("Not enough numeric features found for correlation analysis")
            return
        
        # Get visualization settings
        colormap = corr_config.get("colormap", "coolwarm")
        annotate = corr_config.get("annotate", True)
        figsize = tuple(corr_config.get("figsize", [10, 8]))
        dpi = viz_config.get("dpi", 150)
        fig_format = viz_config.get("figure_format", "png")
        
        # Get enabled domains
        domains_config = self.config.get("domains", {})
        enabled_domains = [d for d, cfg in domains_config.items() if cfg.get("enabled", True)]
        
        # Generate correlation heatmap for each domain
        for domain_name in enabled_domains:
            if domain_name not in df.columns:
                continue
            
            subset = df[df[domain_name] == 1]
            
            if len(subset) < 10:
                self.logger.warning(f"Not enough data for correlation analysis in {domain_name} (n={len(subset)})")
                continue
            
            # Calculate correlation matrix
            corr_data = subset[existing_features].apply(pd.to_numeric, errors='coerce')
            corr_matrix = corr_data.corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=figsize)
            
            sns.heatmap(
                corr_matrix,
                annot=annotate,
                cmap=colormap,
                fmt=".2f",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                ax=ax
            )
            
            ax.set_title(f"Feature Correlation Matrix: {domain_name}\n(n={len(subset):,} compounds)", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / f"correlation_{domain_name}.{fig_format}"
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"  Saved correlation heatmap: {output_path}")
        
        # Generate overall correlation heatmap (all data)
        corr_data = df[existing_features].apply(pd.to_numeric, errors='coerce')
        corr_matrix = corr_data.corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            corr_matrix,
            annot=annotate,
            cmap=colormap,
            fmt=".2f",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title(f"Feature Correlation Matrix: All Compounds\n(n={len(df):,} compounds)", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = output_dir / f"correlation_all_compounds.{fig_format}"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  Saved correlation heatmap: {output_path}")
    
    def generate_violin_plots(self, df: pd.DataFrame, output_dir: Path):
        """Generate violin plots comparing feature distributions across domains."""
        viz_config = self.config.get("visualization", {})
        violin_config = viz_config.get("violin_plots", {})
        
        if not violin_config.get("enabled", True):
            return
        
        self.logger.info("Generating violin plots...")
        
        # Get feature columns
        features = violin_config.get("features", [
            "mw", "xlogp", "polararea", "complexity", "rotbonds", "hbonddonor", "hbondacc"
        ])
        
        # Filter to existing columns
        existing_features = [f for f in features if f in df.columns]
        
        if not existing_features:
            self.logger.warning("No valid features found for violin plots")
            return
        
        # Get visualization settings
        figsize = tuple(violin_config.get("figsize", [12, 6]))
        palette = violin_config.get("palette", "Set2")
        dpi = viz_config.get("dpi", 150)
        fig_format = viz_config.get("figure_format", "png")
        
        # Get enabled domains
        domains_config = self.config.get("domains", {})
        enabled_domains = [d for d, cfg in domains_config.items() if cfg.get("enabled", True)]
        
        # Create a melted dataframe for domain comparison
        # First, create domain assignment column
        df_plot = df.copy()
        
        # Create domain labels for each compound
        def get_domain_label(row):
            labels = []
            for domain in enabled_domains:
                if domain in row and row[domain] == 1:
                    labels.append(domain.replace('_related', '').replace('_', ' ').title())
            if not labels:
                return 'Unlabeled'
            return ' + '.join(labels)
        
        df_plot['Domain'] = df_plot.apply(get_domain_label, axis=1)
        
        # Generate violin plot for each feature
        for feature in existing_features:
            # Convert to numeric
            df_plot[feature] = pd.to_numeric(df_plot[feature], errors='coerce')
            
            # Filter out NaN values for this feature
            plot_data = df_plot[df_plot[feature].notna()].copy()
            
            if len(plot_data) < 10:
                self.logger.warning(f"Not enough data for violin plot of {feature}")
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create violin plot
            sns.violinplot(
                data=plot_data,
                x='Domain',
                y=feature,
                palette=palette,
                ax=ax,
                cut=0,
                inner='box'
            )
            
            ax.set_title(f"Distribution of {feature} by Domain", fontsize=12)
            ax.set_xlabel("Domain", fontsize=10)
            ax.set_ylabel(feature, fontsize=10)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / f"violin_{feature}.{fig_format}"
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"  Saved violin plot: {output_path}")
        
        # Create a combined violin plot with multiple features (normalized)
        self.logger.info("Generating combined feature comparison plot...")
        
        # Melt the dataframe for multi-feature comparison
        melt_features = existing_features[:6]  # Limit to 6 features for readability
        
        # Normalize features for comparison (z-score)
        df_normalized = df_plot.copy()
        for feat in melt_features:
            df_normalized[feat] = pd.to_numeric(df_normalized[feat], errors='coerce')
            mean_val = df_normalized[feat].mean()
            std_val = df_normalized[feat].std()
            if std_val > 0:
                df_normalized[f"{feat}_norm"] = (df_normalized[feat] - mean_val) / std_val
            else:
                df_normalized[f"{feat}_norm"] = 0
        
        norm_features = [f"{f}_norm" for f in melt_features]
        
        # Melt for plotting
        df_melted = df_normalized.melt(
            id_vars=['Domain'],
            value_vars=norm_features,
            var_name='Feature',
            value_name='Normalized Value'
        )
        df_melted['Feature'] = df_melted['Feature'].str.replace('_norm', '')
        
        # Create combined plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.violinplot(
            data=df_melted,
            x='Feature',
            y='Normalized Value',
            hue='Domain',
            palette=palette,
            ax=ax,
            cut=0,
            inner='quartile',
            split=False
        )
        
        ax.set_title("Normalized Feature Distributions by Domain", fontsize=12)
        ax.set_xlabel("Feature", fontsize=10)
        ax.set_ylabel("Normalized Value (Z-score)", fontsize=10)
        ax.legend(title='Domain', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = output_dir / f"violin_combined_features.{fig_format}"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"  Saved combined violin plot: {output_path}")
    
    def generate_visualizations(self, df: pd.DataFrame, output_dir: Path):
        """Generate all visualizations."""
        viz_config = self.config.get("visualization", {})
        
        if not viz_config.get("enabled", True):
            self.logger.info("Visualization is disabled in configuration")
            return
        
        # Create plots output directory
        plots_dir = Path(viz_config.get("output_dir", output_dir / "plots"))
        if not plots_dir.is_absolute():
            plots_dir = output_dir / plots_dir.name
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating visualizations in {plots_dir}")
        
        # Generate correlation heatmaps
        self.generate_correlation_heatmaps(df, plots_dir)
        
        # Generate violin plots
        self.generate_violin_plots(df, plots_dir)
        
        self.logger.info("Visualization generation complete")
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full domain labeling pipeline."""
        self.logger.info("Starting domain labeling pipeline...")
        
        # Apply labels
        df = self.label_domains(df)
        
        self.logger.info("Domain labeling complete")
        
        return df


def setup_logging(config: dict) -> logging.Logger:
    """Set up logging based on configuration."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("log_level", "INFO").upper())
    
    logger = logging.getLogger("phorce_domain_labeling")
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
    
    # File handler (if specified)
    log_file = log_config.get("log_file")
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file.replace("preprocessing", "domain_labeling"))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PHORCE Domain Labeling Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/label_domains.py
    python scripts/label_domains.py --config custom_config.json
    python scripts/label_domains.py --input processed.csv --output labeled.csv
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
        help='Output CSV file path (overrides config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def print_summary(report: DomainLabelingReport):
    """Print a formatted summary of the labeling results."""
    data = report.to_dict()
    summary = data["labeling_summary"]
    domain_stats = data["domain_statistics"]
    
    print("\n" + "="*60)
    print("DOMAIN LABELING SUMMARY")
    print("="*60)
    print(f"Total compounds:        {summary['total_compounds']:,}")
    print(f"Unlabeled compounds:    {summary['unlabeled_compounds']:,} ({summary['unlabeled_percentage']}%)")
    print(f"Multi-domain compounds: {summary['multi_domain_compounds']:,} ({summary['multi_domain_percentage']}%)")
    print(f"Duration:               {summary['duration_seconds']:.2f} seconds")
    print("-"*60)
    print("DOMAIN COUNTS:")
    print("-"*60)
    
    for domain, stats in domain_stats.items():
        print(f"  {domain:20s}: {stats['count']:>10,} ({stats['percentage']:>6.2f}%)")
    
    # Print overlap matrix if available
    overlap = data.get("domain_overlap_matrix", {})
    if overlap and len(overlap) > 1:
        print("-"*60)
        print("DOMAIN OVERLAP MATRIX:")
        print("-"*60)
        
        domains = list(overlap.keys())
        
        # Header
        header = " " * 20 + " ".join(f"{d[:10]:>10s}" for d in domains)
        print(header)
        
        # Rows
        for d1 in domains:
            row = f"{d1[:20]:20s}"
            for d2 in domains:
                row += f"{overlap[d1][d2]:>10,}"
            print(row)
    
    print("="*60)


def main():
    """Main entry point for the domain labeling script."""
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
    
    # Check if domain labeling is enabled
    domain_config = config.get("domain_labeling", {})
    if not domain_config.get("enabled", True):
        print("Domain labeling is disabled in configuration. Exiting.")
        sys.exit(0)
    
    # Override verbose setting if specified
    if args.verbose:
        config.setdefault("logging", {})["log_level"] = "DEBUG"
    
    # Set up logging
    logger = setup_logging(config)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Determine input file
    input_file = args.input or domain_config.get("input_file", "data/processed/P1M_preprocessed.csv")
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output file and directory
    output_file = args.output or domain_config.get("output_file", "data/labeled/domain_labeled_compounds.csv")
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize labeler
    labeler = DomainLabeler(config, logger)
    
    # Load and process data
    try:
        df = labeler.load_data(str(input_path))
        df = labeler.process(df)
        
        # Save main labeled dataset
        logger.info(f"Saving labeled data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} labeled compounds to {output_path}")
        
        # Save domain subsets
        labeler.save_subsets(df, output_dir)
        
        # Generate visualizations
        labeler.generate_visualizations(df, output_dir)
        
        # Save labeling report
        report_path = domain_config.get("report_output", "data/labeled/domain_labeling_report.json")
        report_full_path = Path(report_path)
        if not report_full_path.is_absolute():
            report_full_path = project_root / report_full_path
        
        report_full_path.parent.mkdir(parents=True, exist_ok=True)
        labeler.report.save(str(report_full_path))
        logger.info(f"Domain labeling report saved to {report_full_path}")
        
        # Print summary
        print_summary(labeler.report)
        
    except Exception as e:
        logger.error(f"Domain labeling failed: {e}")
        raise


if __name__ == "__main__":
    main()
