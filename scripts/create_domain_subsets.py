#!/usr/bin/env python3
"""
PHORCE Domain Subset Creator
=============================

Creates subsets within domain-labeled compounds based on physicochemical property
thresholds. This enables more granular classification of compounds within each
environmental domain.

Example subsets for water_related domain:
- Water_Soluble: Highly water-soluble compounds
- Aquatic_Bioavailable: Compounds bioavailable in aquatic environments
- Bioconcentration_Risk: Compounds with bioconcentration potential

Usage:
    python scripts/create_domain_subsets.py
    python scripts/create_domain_subsets.py --config custom_config.json
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging for the domain subset creator."""
    logger = logging.getLogger("phorce_domain_subsets")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
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
class SubsetDefinition:
    """Definition of a domain subset with its criteria."""
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    logic: str = "and"  # "and" or "or" for combining conditions
    parent_domain: Optional[str] = None  # If None, applies to all compounds
    
    def __post_init__(self):
        if self.logic not in ["and", "or"]:
            raise ValueError(f"Logic must be 'and' or 'or', got '{self.logic}'")


@dataclass
class SubsetResult:
    """Results from creating a subset."""
    name: str
    parent_domain: Optional[str]
    total_candidates: int
    subset_count: int
    percentage: float
    conditions_applied: List[str]
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubsetReport:
    """Complete report for domain subset creation."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    input_file: str = ""
    output_file: str = ""
    total_compounds: int = 0
    subsets_created: List[SubsetResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return _to_json_serializable({
            "timestamp": self.timestamp,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "total_compounds": self.total_compounds,
            "subsets_created": [
                {
                    "name": s.name,
                    "parent_domain": s.parent_domain,
                    "total_candidates": s.total_candidates,
                    "subset_count": s.subset_count,
                    "percentage": s.percentage,
                    "conditions_applied": s.conditions_applied,
                    "statistics": s.statistics
                }
                for s in self.subsets_created
            ],
            "warnings": self.warnings
        })


# =============================================================================
# DOMAIN SUBSET CREATOR
# =============================================================================

class DomainSubsetCreator:
    """
    Creates subsets within domain-labeled compounds based on configurable criteria.
    """
    
    # Supported comparison operators
    OPERATORS = {
        "<=": lambda x, v: x <= v,
        "<": lambda x, v: x < v,
        ">=": lambda x, v: x >= v,
        ">": lambda x, v: x > v,
        "==": lambda x, v: x == v,
        "!=": lambda x, v: x != v,
        "between": lambda x, v: x.between(v[0], v[1]),
        "not_between": lambda x, v: ~x.between(v[0], v[1]),
        "in": lambda x, v: x.isin(v),
        "not_in": lambda x, v: ~x.isin(v),
    }
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the domain subset creator.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger("phorce_domain_subsets")
        self.report = SubsetReport()
        
        # Get subset configuration
        self.subset_config = config.get("domain_subsets", {})
        
        # Column name mappings (handle different naming conventions)
        self.column_aliases = self.subset_config.get("column_aliases", {
            "MolLogP": ["xlogp", "logp", "MolLogP", "mollogp"],
            "TPSA": ["polararea", "tpsa", "TPSA", "polar_surface_area"],
            "TotalHBond": ["TotalHBond", "totalhbond", "total_hbond"],
            "MW": ["mw", "MW", "molecular_weight", "MolecularWeight"],
        })
    
    def _resolve_column_name(self, df: pd.DataFrame, column: str) -> Optional[str]:
        """
        Resolve a column name using aliases.
        
        Args:
            df: DataFrame to check columns against
            column: Column name or alias to resolve
            
        Returns:
            Actual column name in DataFrame, or None if not found
        """
        # Direct match
        if column in df.columns:
            return column
        
        # Check aliases
        for canonical, aliases in self.column_aliases.items():
            if column in aliases or column == canonical:
                for alias in aliases:
                    if alias in df.columns:
                        return alias
        
        # Case-insensitive search
        column_lower = column.lower()
        for col in df.columns:
            if col.lower() == column_lower:
                return col
        
        return None
    
    def _evaluate_condition(
        self, 
        df: pd.DataFrame, 
        condition: Dict[str, Any]
    ) -> Tuple[pd.Series, str]:
        """
        Evaluate a single condition on the DataFrame.
        
        Args:
            df: DataFrame to evaluate condition on
            condition: Condition dictionary with 'column', 'operator', 'value'
            
        Returns:
            Tuple of (boolean Series, condition description string)
        """
        column = condition.get("column")
        operator = condition.get("operator")
        value = condition.get("value")
        
        # Resolve column name
        actual_column = self._resolve_column_name(df, column)
        if actual_column is None:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        # Get operator function
        if operator not in self.OPERATORS:
            raise ValueError(f"Unknown operator '{operator}'. Supported: {list(self.OPERATORS.keys())}")
        
        op_func = self.OPERATORS[operator]
        
        # Apply condition
        result = op_func(df[actual_column], value)
        
        # Create description
        if operator == "between":
            desc = f"{actual_column} BETWEEN {value[0]} AND {value[1]}"
        elif operator == "not_between":
            desc = f"{actual_column} NOT BETWEEN {value[0]} AND {value[1]}"
        elif operator in ["in", "not_in"]:
            desc = f"{actual_column} {operator.upper().replace('_', ' ')} {value}"
        else:
            desc = f"{actual_column} {operator} {value}"
        
        return result, desc
    
    def _create_subset(
        self, 
        df: pd.DataFrame, 
        subset_def: SubsetDefinition
    ) -> Tuple[pd.Series, SubsetResult]:
        """
        Create a subset based on the definition.
        
        Args:
            df: DataFrame to create subset from
            subset_def: Subset definition
            
        Returns:
            Tuple of (boolean Series for subset membership, SubsetResult)
        """
        # Determine candidate rows (parent domain filter)
        if subset_def.parent_domain and subset_def.parent_domain in df.columns:
            candidates = df[subset_def.parent_domain] == 1
            total_candidates = candidates.sum()
        else:
            candidates = pd.Series([True] * len(df), index=df.index)
            total_candidates = len(df)
        
        # Evaluate all conditions
        condition_results = []
        condition_descriptions = []
        
        for condition in subset_def.conditions:
            try:
                result, desc = self._evaluate_condition(df, condition)
                condition_results.append(result)
                condition_descriptions.append(desc)
            except ValueError as e:
                self.logger.warning(f"  Skipping condition: {e}")
                self.report.warnings.append(f"{subset_def.name}: {str(e)}")
        
        if not condition_results:
            # No valid conditions
            subset_mask = pd.Series([False] * len(df), index=df.index)
        else:
            # Combine conditions
            if subset_def.logic == "and":
                combined = condition_results[0]
                for result in condition_results[1:]:
                    combined = combined & result
            else:  # "or"
                combined = condition_results[0]
                for result in condition_results[1:]:
                    combined = combined | result
            
            # Apply parent domain filter
            subset_mask = combined & candidates
        
        subset_count = subset_mask.sum()
        percentage = (subset_count / total_candidates * 100) if total_candidates > 0 else 0.0
        
        # Calculate statistics for subset
        statistics = {}
        if subset_count > 0:
            subset_df = df[subset_mask]
            numeric_cols = subset_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
                if col in subset_df.columns and not subset_df[col].isna().all():
                    statistics[col] = {
                        "mean": float(subset_df[col].mean()),
                        "std": float(subset_df[col].std()),
                        "min": float(subset_df[col].min()),
                        "max": float(subset_df[col].max())
                    }
        
        result = SubsetResult(
            name=subset_def.name,
            parent_domain=subset_def.parent_domain,
            total_candidates=int(total_candidates),
            subset_count=int(subset_count),
            percentage=round(percentage, 2),
            conditions_applied=condition_descriptions,
            statistics=statistics
        )
        
        return subset_mask, result
    
    def create_subsets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all configured subsets and add them as columns.
        
        Args:
            df: Input DataFrame with domain labels
            
        Returns:
            DataFrame with new subset columns added
        """
        df = df.copy()
        
        # Get subset definitions from config
        subsets_config = self.subset_config.get("subsets", [])
        
        if not subsets_config:
            self.logger.warning("No subset definitions found in configuration")
            return df
        
        self.logger.info(f"\nCreating {len(subsets_config)} domain subsets...")
        self.logger.info("=" * 60)
        
        for subset_cfg in subsets_config:
            # Parse subset definition
            subset_def = SubsetDefinition(
                name=subset_cfg.get("name"),
                description=subset_cfg.get("description", ""),
                conditions=subset_cfg.get("conditions", []),
                logic=subset_cfg.get("logic", "and"),
                parent_domain=subset_cfg.get("parent_domain")
            )
            
            self.logger.info(f"\n  Creating subset: {subset_def.name}")
            if subset_def.parent_domain:
                self.logger.info(f"    Parent domain: {subset_def.parent_domain}")
            self.logger.info(f"    Description: {subset_def.description}")
            
            try:
                # Create the subset
                subset_mask, result = self._create_subset(df, subset_def)
                
                # Add as column
                df[subset_def.name] = subset_mask.astype(int)
                
                # Log results
                self.logger.info(f"    Conditions: {', '.join(result.conditions_applied)}")
                self.logger.info(f"    Candidates: {result.total_candidates:,}")
                self.logger.info(f"    Matches: {result.subset_count:,} ({result.percentage:.2f}%)")
                
                # Add to report
                self.report.subsets_created.append(result)
                
            except Exception as e:
                self.logger.error(f"    Failed to create subset: {e}")
                self.report.warnings.append(f"Failed to create {subset_def.name}: {str(e)}")
        
        return df
    
    def save_subset_files(
        self, 
        df: pd.DataFrame, 
        output_dir: Path
    ) -> Dict[str, str]:
        """
        Save individual subset files if configured.
        
        Args:
            df: DataFrame with subset columns
            output_dir: Directory to save subset files
            
        Returns:
            Dictionary mapping subset names to output file paths
        """
        saved_files = {}
        
        save_config = self.subset_config.get("save_individual_subsets", False)
        if not save_config:
            return saved_files
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for result in self.report.subsets_created:
            if result.subset_count > 0 and result.name in df.columns:
                subset_df = df[df[result.name] == 1]
                filename = f"{result.name.lower()}_compounds.csv"
                filepath = output_dir / filename
                
                subset_df.to_csv(filepath, index=False)
                saved_files[result.name] = str(filepath)
                self.logger.info(f"  Saved {result.name}: {filepath}")
        
        return saved_files


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for domain subset creation."""
    parser = argparse.ArgumentParser(
        description="Create subsets within domain-labeled compounds"
    )
    parser.add_argument(
        "--config", "-c",
        default="preprocessing_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input", "-i",
        help="Override input file path"
    )
    parser.add_argument(
        "--output", "-o",
        help="Override output file path"
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Setup logging
    log_config = config.get("logging", {})
    logger = setup_logging(
        log_level=log_config.get("log_level", "INFO"),
        log_file=log_config.get("log_file")
    )
    
    logger.info(f"Loaded configuration from {config_path.absolute()}")
    
    # Get paths from config
    subset_config = config.get("domain_subsets", {})
    base_path = config_path.parent
    
    input_file = args.input or subset_config.get(
        "input_file", 
        config.get("decision_tree_analysis", {}).get("input_file", "data/engineered/P1M_engineered.csv")
    )
    output_file = args.output or subset_config.get(
        "output_file",
        "data/subsets/P1M_with_subsets.csv"
    )
    output_dir = subset_config.get("output_dir", "data/subsets")
    report_output = subset_config.get("report_output", "data/subsets/subset_report.json")
    
    # Resolve paths
    input_path = base_path / input_file if not Path(input_file).is_absolute() else Path(input_file)
    output_path = base_path / output_file if not Path(output_file).is_absolute() else Path(output_file)
    output_dir_path = base_path / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    report_path = base_path / report_output if not Path(report_output).is_absolute() else Path(report_output)
    
    # Load data
    logger.info(f"Loading data from {input_path}")
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} compounds with {len(df.columns)} columns")
    
    # Create subset creator and process
    creator = DomainSubsetCreator(config, logger)
    creator.report.input_file = str(input_path)
    creator.report.output_file = str(output_path)
    creator.report.total_compounds = len(df)
    
    # Create subsets
    df = creator.create_subsets(df)
    
    # Save main output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved output to {output_path}")
    
    # Save individual subset files
    saved_files = creator.save_subset_files(df, output_dir_path)
    
    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(creator.report.to_dict(), f, indent=2)
    logger.info(f"Saved report to {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("DOMAIN SUBSET SUMMARY")
    print("=" * 70)
    
    for result in creator.report.subsets_created:
        parent_str = f" (within {result.parent_domain})" if result.parent_domain else ""
        print(f"\n  {result.name}{parent_str}")
        print(f"  {'-' * 50}")
        print(f"    Candidates:  {result.total_candidates:,}")
        print(f"    Matches:     {result.subset_count:,} ({result.percentage:.2f}%)")
        print(f"    Conditions:  {' AND '.join(result.conditions_applied)}")
    
    print("\n" + "=" * 70)
    
    return df


if __name__ == "__main__":
    main()
