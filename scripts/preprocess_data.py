#!/usr/bin/env python3
"""
PHORCE Data Preprocessing Script

This script reads raw chemical compound data from CSV files and performs
extensive preprocessing and feature cleaning operations. The preprocessing
options are configurable via a JSON configuration file.

Note: This script does NOT perform imputation of missing numeric values.
Missing values are handled by dropping rows/columns based on configuration.

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --config custom_config.json
    python scripts/preprocess_data.py --config preprocessing_config.json --input data.csv --output processed.csv
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Optional: RDKit for chemical validation (if available)
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDKIT_AVAILABLE = True
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    RDKIT_AVAILABLE = False


class PreprocessingReport:
    """Class to track and report preprocessing statistics."""
    
    def __init__(self):
        self.initial_rows = 0
        self.initial_columns = 0
        self.final_rows = 0
        self.final_columns = 0
        self.steps = []
        self.column_stats = {}
        self.start_time = datetime.now()
        self.end_time = None
        
    def add_step(self, step_name: str, rows_before: int, rows_after: int,
                 cols_before: int, cols_after: int, details: Optional[dict] = None):
        """Record a preprocessing step."""
        step_info = {
            "step_name": step_name,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
            "columns_before": cols_before,
            "columns_after": cols_after,
            "columns_removed": cols_before - cols_after,
            "timestamp": datetime.now().isoformat()
        }
        if details:
            step_info["details"] = details
        self.steps.append(step_info)
        
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        self.end_time = datetime.now()
        return {
            "preprocessing_summary": {
                "initial_rows": self.initial_rows,
                "initial_columns": self.initial_columns,
                "final_rows": self.final_rows,
                "final_columns": self.final_columns,
                "total_rows_removed": self.initial_rows - self.final_rows,
                "total_columns_removed": self.initial_columns - self.final_columns,
                "retention_rate_rows": round(self.final_rows / self.initial_rows * 100, 2) if self.initial_rows > 0 else 0,
                "retention_rate_columns": round(self.final_columns / self.initial_columns * 100, 2) if self.initial_columns > 0 else 0,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds()
            },
            "preprocessing_steps": self.steps,
            "column_statistics": self.column_stats
        }
    
    def save(self, filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class DataPreprocessor:
    """Main class for preprocessing chemical compound data."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.report = PreprocessingReport()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        self.logger.info(f"Loading data from {filepath}")
        
        # Detect encoding and handle large files
        try:
            df = pd.read_csv(
                filepath,
                low_memory=False,
                on_bad_lines='warn'
            )
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise
            
        self.report.initial_rows = len(df)
        self.report.initial_columns = len(df.columns)
        self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def standardize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize various missing value representations to NaN."""
        rows_before, cols_before = df.shape
        
        missing_indicators = self.config.get("missing_value_handling", {}).get(
            "missing_value_indicators", 
            ["", "NA", "N/A", "NaN", "nan", "null", "None", ".", "-", "?"]
        )
        
        # Replace missing value indicators with NaN
        for indicator in missing_indicators:
            if indicator != "":  # Empty string handled separately
                df = df.replace(indicator, np.nan)
        
        # Replace empty strings
        df = df.replace(r'^\s*$', np.nan, regex=True)
        
        self.report.add_step(
            "Standardize Missing Values",
            rows_before, len(df),
            cols_before, len(df.columns),
            {"missing_indicators": missing_indicators}
        )
        
        return df
    
    def drop_columns_with_high_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with missing values above threshold."""
        config = self.config.get("missing_value_handling", {})
        threshold = config.get("missing_threshold_columns", 0.9)
        
        rows_before, cols_before = df.shape
        
        # Calculate missing percentage per column
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        # Build detailed drop reasons
        drop_reasons = {}
        if cols_to_drop:
            self.logger.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100:.1f}% missing values:")
            for col in cols_to_drop:
                pct = missing_pct[col] * 100
                drop_reasons[col] = f"{pct:.2f}% missing (threshold: {threshold*100:.1f}%)"
                self.logger.info(f"  - {col}: {pct:.2f}% missing")
            df = df.drop(columns=cols_to_drop)
        
        self.report.add_step(
            "Drop High-Missing Columns",
            rows_before, len(df),
            cols_before, len(df.columns),
            {
                "threshold": threshold,
                "threshold_percent": f"{threshold*100:.1f}%",
                "columns_dropped": cols_to_drop,
                "drop_reasons": drop_reasons,
                "reason": f"Columns with more than {threshold*100:.1f}% missing values are dropped"
            }
        )
        
        return df
    
    def drop_rows_with_high_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing values above threshold."""
        config = self.config.get("missing_value_handling", {})
        threshold = config.get("missing_threshold_rows", 0.5)
        
        rows_before, cols_before = df.shape
        
        # Calculate missing percentage per row
        missing_pct = df.isnull().sum(axis=1) / len(df.columns)
        rows_to_keep = missing_pct <= threshold
        
        rows_dropped = (~rows_to_keep).sum()
        
        # Calculate statistics about dropped rows
        dropped_missing_pcts = missing_pct[~rows_to_keep]
        stats = {}
        if rows_dropped > 0:
            stats = {
                "min_missing_pct": float(dropped_missing_pcts.min() * 100),
                "max_missing_pct": float(dropped_missing_pcts.max() * 100),
                "avg_missing_pct": float(dropped_missing_pcts.mean() * 100)
            }
            self.logger.info(f"Dropping {rows_dropped:,} rows with >{threshold*100:.1f}% missing values")
            self.logger.info(f"  Reason: Rows have between {stats['min_missing_pct']:.1f}% and {stats['max_missing_pct']:.1f}% missing values (avg: {stats['avg_missing_pct']:.1f}%)")
            self.logger.info(f"  Threshold: Rows with more than {threshold*100:.1f}% missing values ({int(threshold * len(df.columns))} of {len(df.columns)} columns) are dropped")
            df = df[rows_to_keep].reset_index(drop=True)
        
        self.report.add_step(
            "Drop High-Missing Rows",
            rows_before, len(df),
            cols_before, len(df.columns),
            {
                "threshold": threshold,
                "threshold_percent": f"{threshold*100:.1f}%",
                "max_missing_columns_allowed": int(threshold * cols_before),
                "total_columns": cols_before,
                "rows_dropped": int(rows_dropped),
                "dropped_row_stats": stats,
                "reason": f"Rows with more than {threshold*100:.1f}% missing values (>{int(threshold * cols_before)} of {cols_before} columns missing) are dropped"
            }
        )
        
        return df
    
    def drop_rows_missing_required(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows missing required columns."""
        # Check if this step is enabled in configuration
        missing_config = self.config.get("missing_value_handling", {})
        if not missing_config.get("drop_rows_with_missing_required", True):
            self.logger.info("Skipping drop_rows_missing_required (disabled in config: drop_rows_with_missing_required=false)")
            return df
        
        config = self.config.get("column_handling", {})
        required_cols = config.get("required_columns", [])
        
        if not required_cols:
            return df
            
        rows_before, cols_before = df.shape
        
        # Check which required columns exist
        existing_required = [col for col in required_cols if col in df.columns]
        missing_required = [col for col in required_cols if col not in df.columns]
        
        if missing_required:
            self.logger.warning(f"Required columns not found in data: {missing_required}")
        
        # Calculate missing counts per required column BEFORE dropping
        missing_per_column = {}
        for col in existing_required:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df) * 100
            missing_per_column[col] = {
                "missing_count": int(missing_count),
                "missing_percent": round(missing_pct, 2),
                "has_value_count": int(len(df) - missing_count)
            }
        
        rows_dropped = 0
        if existing_required:
            initial_count = len(df)
            df = df.dropna(subset=existing_required).reset_index(drop=True)
            rows_dropped = initial_count - len(df)
            
            if rows_dropped > 0:
                self.logger.info(f"Dropped {rows_dropped:,} rows with missing required columns")
                self.logger.info(f"  Reason: Rows must have non-null values in ALL required columns: {existing_required}")
                self.logger.info(f"  Missing value breakdown per required column:")
                for col, stats in missing_per_column.items():
                    self.logger.info(f"    - {col}: {stats['missing_count']:,} missing ({stats['missing_percent']:.2f}%), {stats['has_value_count']:,} have values")
        
        self.report.add_step(
            "Drop Rows Missing Required Columns",
            rows_before, len(df),
            cols_before, len(df.columns),
            {
                "required_columns": existing_required,
                "required_columns_not_found": missing_required,
                "rows_dropped": rows_dropped,
                "missing_per_column": missing_per_column,
                "reason": f"Rows missing ANY of the required columns {existing_required} are dropped"
            }
        )
        
        return df
    
    def drop_rows_with_any_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with any missing numeric values based on strategy configuration."""
        config = self.config.get("missing_value_handling", {})
        strategy = config.get("strategy", None)
        
        # Only apply if strategy is explicitly set to "drop_rows"
        if strategy != "drop_rows":
            self.logger.info(f"Skipping drop_rows_with_any_missing (strategy is '{strategy}', not 'drop_rows')")
            return df
            
        rows_before, cols_before = df.shape
        
        # Get numeric columns from config (integer + float columns)
        data_type_config = self.config.get("data_type_enforcement", {})
        integer_cols = data_type_config.get("integer_columns", [])
        float_cols = data_type_config.get("float_columns", [])
        numeric_cols = integer_cols + float_cols
        
        # Get required columns to exclude from this check (they're handled separately)
        column_config = self.config.get("column_handling", {})
        required_cols = set(column_config.get("required_columns", []))
        
        # Filter to only columns that exist in the dataframe and exclude required columns
        # Use a set to deduplicate, then convert back to list to preserve order
        existing_numeric_cols = list(dict.fromkeys(
            col for col in numeric_cols if col in df.columns and col not in required_cols
        ))
        
        if not existing_numeric_cols:
            self.logger.info("No numeric columns found for drop_rows strategy (after excluding required columns)")
            return df
        
        # Calculate missing counts per numeric column BEFORE dropping
        missing_per_column = {}
        for col in existing_numeric_cols:
            # Handle case where column might appear multiple times in dataframe
            col_data = df[col]
            if isinstance(col_data, pd.DataFrame):
                # If multiple columns with same name, take the first one
                col_data = col_data.iloc[:, 0]
            missing_count = int(col_data.isnull().sum())
            missing_pct = missing_count / len(df) * 100
            if missing_count > 0:
                missing_per_column[col] = {
                    "missing_count": missing_count,
                    "missing_percent": round(missing_pct, 2)
                }
        
        initial_count = len(df)
        df = df.dropna(subset=existing_numeric_cols).reset_index(drop=True)
        rows_dropped = initial_count - len(df)
        
        if rows_dropped > 0:
            self.logger.info(f"Dropped {rows_dropped:,} rows with missing numeric values (strategy: drop_rows)")
            self.logger.info(f"  Reason: Rows must have non-null values in ALL numeric columns when strategy='drop_rows'")
            self.logger.info(f"  Numeric columns checked ({len(existing_numeric_cols)}): {existing_numeric_cols}")
            if missing_per_column:
                self.logger.info(f"  Columns with missing values that caused drops:")
                for col, stats in sorted(missing_per_column.items(), key=lambda x: x[1]['missing_count'], reverse=True):
                    self.logger.info(f"    - {col}: {stats['missing_count']:,} missing ({stats['missing_percent']:.2f}%)")
        
        self.report.add_step(
            "Drop Rows With Missing Numeric Values",
            rows_before, len(df),
            cols_before, len(df.columns),
            {
                "strategy": strategy,
                "rows_dropped": rows_dropped,
                "numeric_columns_checked": existing_numeric_cols,
                "columns_with_missing_values": missing_per_column,
                "reason": f"Rows with ANY missing value in numeric columns {existing_numeric_cols} are dropped (strategy='drop_rows')"
            }
        )
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows based on configuration."""
        config = self.config.get("duplicate_handling", {})
        
        if not config.get("remove_duplicates", True):
            return df
            
        rows_before, cols_before = df.shape
        
        subset = config.get("duplicate_subset", None)
        keep = config.get("keep_strategy", "first")
        
        # Check if subset columns exist
        if subset:
            subset = [col for col in subset if col in df.columns]
            if not subset:
                subset = None
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        duplicates_removed = initial_count - len(df)
        
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        self.report.add_step(
            "Remove Duplicates",
            rows_before, len(df),
            cols_before, len(df.columns),
            {"subset": subset, "keep": keep, "duplicates_removed": duplicates_removed}
        )
        
        return df
    
    def clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns (strip whitespace, etc.)."""
        config = self.config.get("text_cleaning", {})
        
        if not config.get("enabled", True):
            return df
            
        rows_before, cols_before = df.shape
        
        # Strip whitespace from all string columns
        if config.get("strip_whitespace", True):
            string_cols = df.select_dtypes(include=['object']).columns
            for col in string_cols:
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Lowercase specific columns
        lowercase_cols = config.get("lowercase_columns", [])
        for col in lowercase_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
        
        self.report.add_step(
            "Clean Text Columns",
            rows_before, len(df),
            cols_before, len(df.columns),
            {"strip_whitespace": config.get("strip_whitespace", True),
             "lowercase_columns": lowercase_cols}
        )
        
        return df
    
    def clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns (remove invalid values, handle outliers)."""
        config = self.config.get("numeric_cleaning", {})
        
        if not config.get("enabled", True):
            return df
            
        rows_before, cols_before = df.shape
        removed_details = {}
        
        # Remove negative values where they shouldn't exist
        neg_cols = config.get("remove_negative_values_columns", [])
        for col in neg_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    neg_count = (df[col] < 0).sum()
                    if neg_count > 0:
                        df.loc[df[col] < 0, col] = np.nan
                        removed_details[f"{col}_negative_values"] = int(neg_count)
                except Exception as e:
                    self.logger.warning(f"Could not process negative values in {col}: {e}")
        
        # Remove zero values where specified
        zero_cols = config.get("remove_zero_values_columns", [])
        for col in zero_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    zero_count = (df[col] == 0).sum()
                    if zero_count > 0:
                        df.loc[df[col] == 0, col] = np.nan
                        removed_details[f"{col}_zero_values"] = int(zero_count)
                except Exception as e:
                    self.logger.warning(f"Could not process zero values in {col}: {e}")
        
        # Handle outliers if configured (mark as NaN, not impute)
        if config.get("clip_outliers", False):
            method = config.get("outlier_method", "iqr")
            multiplier = config.get("outlier_multiplier", 3.0)
            outlier_cols = config.get("outlier_columns", [])
            
            for col in outlier_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        if method == "iqr":
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - multiplier * IQR
                            upper_bound = Q3 + multiplier * IQR
                        elif method == "zscore":
                            mean = df[col].mean()
                            std = df[col].std()
                            lower_bound = mean - multiplier * std
                            upper_bound = mean + multiplier * std
                        else:
                            continue
                            
                        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                        outlier_count = outlier_mask.sum()
                        
                        if outlier_count > 0:
                            df.loc[outlier_mask, col] = np.nan
                            removed_details[f"{col}_outliers"] = int(outlier_count)
                            
                    except Exception as e:
                        self.logger.warning(f"Could not process outliers in {col}: {e}")
        
        self.report.add_step(
            "Clean Numeric Columns",
            rows_before, len(df),
            cols_before, len(df.columns),
            removed_details
        )
        
        return df
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate a SMILES string using RDKit."""
        if not RDKIT_AVAILABLE:
            return True  # Cannot validate without RDKit
        if pd.isna(smiles) or not isinstance(smiles, str):
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def validate_inchi(self, inchi: str) -> bool:
        """Validate an InChI string using RDKit."""
        if not RDKIT_AVAILABLE:
            return True  # Cannot validate without RDKit
        if pd.isna(inchi) or not isinstance(inchi, str):
            return False
        try:
            mol = Chem.MolFromInchi(inchi)
            return mol is not None
        except:
            return False
    
    def validate_chemical_structures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate chemical structures (SMILES, InChI)."""
        config = self.config.get("chemical_validation", {})
        
        if not config.get("enabled", True):
            return df
            
        rows_before, cols_before = df.shape
        validation_details = {}
        
        smiles_col = self.config.get("column_handling", {}).get("smiles_column", "canonicalsmiles")
        
        # Validate SMILES
        if config.get("validate_smiles", True) and smiles_col in df.columns:
            if RDKIT_AVAILABLE:
                self.logger.info("Validating SMILES strings...")
                valid_smiles = df[smiles_col].apply(self.validate_smiles)
                invalid_count = (~valid_smiles).sum()
                validation_details["invalid_smiles_count"] = int(invalid_count)
                
                if config.get("remove_invalid_smiles", True) and invalid_count > 0:
                    self.logger.info(f"Removing {invalid_count} rows with invalid SMILES")
                    df = df[valid_smiles].reset_index(drop=True)
            else:
                self.logger.warning("RDKit not available - skipping SMILES validation")
                validation_details["smiles_validation"] = "skipped - RDKit not available"
        
        # Validate InChI
        if config.get("validate_inchi", False):
            inchi_col = "inchi"
            if inchi_col in df.columns and RDKIT_AVAILABLE:
                self.logger.info("Validating InChI strings...")
                valid_inchi = df[inchi_col].apply(self.validate_inchi)
                invalid_count = (~valid_inchi).sum()
                validation_details["invalid_inchi_count"] = int(invalid_count)
                
                if config.get("remove_invalid_inchi", False) and invalid_count > 0:
                    self.logger.info(f"Removing {invalid_count} rows with invalid InChI")
                    df = df[valid_inchi].reset_index(drop=True)
        
        # Remove disconnected structures (salts, mixtures)
        if config.get("remove_disconnected_structures", False) and smiles_col in df.columns:
            if RDKIT_AVAILABLE:
                self.logger.info("Checking for disconnected structures...")
                disconnected_mask = df[smiles_col].apply(
                    lambda x: '.' in str(x) if pd.notna(x) else False
                )
                disconnected_count = disconnected_mask.sum()
                validation_details["disconnected_structures_count"] = int(disconnected_count)
                
                if disconnected_count > 0:
                    self.logger.info(f"Removing {disconnected_count} disconnected structures")
                    df = df[~disconnected_mask].reset_index(drop=True)
        
        self.report.add_step(
            "Validate Chemical Structures",
            rows_before, len(df),
            cols_before, len(df.columns),
            validation_details
        )
        
        return df
    
    def enforce_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce data types for columns."""
        config = self.config.get("data_type_enforcement", {})
        
        if not config.get("enabled", True):
            return df
            
        rows_before, cols_before = df.shape
        type_conversion_details = {}
        
        # Convert integer columns
        int_cols = config.get("integer_columns", [])
        for col in int_cols:
            if col in df.columns:
                try:
                    # Use nullable integer type to handle NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    type_conversion_details[col] = "Int64"
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to integer: {e}")
        
        # Convert float columns
        float_cols = config.get("float_columns", [])
        for col in float_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                    type_conversion_details[col] = "float64"
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to float: {e}")
        
        # Ensure string columns are string type
        str_cols = config.get("string_columns", [])
        for col in str_cols:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(str).replace('nan', np.nan)
                    type_conversion_details[col] = "string"
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to string: {e}")
        
        self.report.add_step(
            "Enforce Data Types",
            rows_before, len(df),
            cols_before, len(df.columns),
            type_conversion_details
        )
        
        return df
    
    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with constant or quasi-constant values."""
        config = self.config.get("feature_selection", {})
        
        if not config.get("enabled", False):
            return df
            
        rows_before, cols_before = df.shape
        removed_cols = []
        
        # Remove constant columns
        if config.get("remove_constant_columns", True):
            constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
            if constant_cols:
                self.logger.info(f"Removing {len(constant_cols)} constant columns")
                df = df.drop(columns=constant_cols)
                removed_cols.extend(constant_cols)
        
        # Remove quasi-constant columns
        if config.get("remove_quasi_constant_columns", True):
            threshold = config.get("quasi_constant_threshold", 0.99)
            quasi_constant_cols = []
            
            for col in df.columns:
                if col not in removed_cols:
                    value_counts = df[col].value_counts(normalize=True, dropna=True)
                    if len(value_counts) > 0 and value_counts.iloc[0] >= threshold:
                        quasi_constant_cols.append(col)
            
            if quasi_constant_cols:
                self.logger.info(f"Removing {len(quasi_constant_cols)} quasi-constant columns")
                df = df.drop(columns=quasi_constant_cols)
                removed_cols.extend(quasi_constant_cols)
        
        self.report.add_step(
            "Remove Constant/Quasi-Constant Columns",
            rows_before, len(df),
            cols_before, len(df.columns),
            {"removed_columns": removed_cols}
        )
        
        return df
    
    def remove_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated numeric features."""
        config = self.config.get("feature_selection", {})
        
        if not config.get("enabled", False) or not config.get("remove_highly_correlated", False):
            return df
            
        rows_before, cols_before = df.shape
        threshold = config.get("correlation_threshold", 0.95)
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return df
            
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        cols_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        
        if cols_to_drop:
            self.logger.info(f"Removing {len(cols_to_drop)} highly correlated columns")
            df = df.drop(columns=cols_to_drop)
        
        self.report.add_step(
            "Remove Highly Correlated Features",
            rows_before, len(df),
            cols_before, len(df.columns),
            {"threshold": threshold, "removed_columns": cols_to_drop}
        )
        
        return df
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply custom filters to the data."""
        config = self.config.get("filtering", {})
        
        if not config.get("enabled", False):
            return df
            
        rows_before, cols_before = df.shape
        filter_details = {}
        
        # Molecular weight filter
        if "mw_range" in config and "mw" in df.columns:
            mw_range = config["mw_range"]
            df["mw"] = pd.to_numeric(df["mw"], errors='coerce')
            initial = len(df)
            df = df[(df["mw"] >= mw_range[0]) & (df["mw"] <= mw_range[1]) | df["mw"].isna()]
            filter_details["mw_filtered"] = initial - len(df)
        
        # XLogP filter
        if "xlogp_range" in config and "xlogp" in df.columns:
            xlogp_range = config["xlogp_range"]
            df["xlogp"] = pd.to_numeric(df["xlogp"], errors='coerce')
            initial = len(df)
            df = df[(df["xlogp"] >= xlogp_range[0]) & (df["xlogp"] <= xlogp_range[1]) | df["xlogp"].isna()]
            filter_details["xlogp_filtered"] = initial - len(df)
        
        # Heavy atom count filter
        if "heavy_atom_range" in config and "heavycnt" in df.columns:
            ha_range = config["heavy_atom_range"]
            df["heavycnt"] = pd.to_numeric(df["heavycnt"], errors='coerce')
            initial = len(df)
            df = df[(df["heavycnt"] >= ha_range[0]) & (df["heavycnt"] <= ha_range[1]) | df["heavycnt"].isna()]
            filter_details["heavy_atom_filtered"] = initial - len(df)
        
        df = df.reset_index(drop=True)
        
        self.report.add_step(
            "Apply Filters",
            rows_before, len(df),
            cols_before, len(df.columns),
            filter_details
        )
        
        return df
    
    def drop_specified_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns specified in configuration."""
        config = self.config.get("column_handling", {})
        
        rows_before, cols_before = df.shape
        
        # Drop specified columns
        drop_cols = config.get("drop_columns", [])
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        
        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)
            self.logger.info(f"Dropped {len(existing_drop_cols)} specified columns")
        
        # Keep only specified columns (if provided)
        keep_cols = config.get("keep_columns", None)
        if keep_cols is not None:
            existing_keep_cols = [col for col in keep_cols if col in df.columns]
            df = df[existing_keep_cols]
            self.logger.info(f"Keeping only {len(existing_keep_cols)} specified columns")
        
        self.report.add_step(
            "Drop/Keep Specified Columns",
            rows_before, len(df),
            cols_before, len(df.columns),
            {"dropped": existing_drop_cols, "kept": keep_cols}
        )
        
        return df
    
    def compute_column_statistics(self, df: pd.DataFrame):
        """Compute statistics for each column."""
        stats = {}
        
        # Get unique column names to avoid duplicates
        seen_cols = set()
        for col in df.columns:
            if col in seen_cols:
                continue
            seen_cols.add(col)
            
            # Handle case where column might appear multiple times
            col_data = df[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            
            col_stats = {
                "dtype": str(col_data.dtype),
                "non_null_count": int(col_data.notna().sum()),
                "null_count": int(col_data.isna().sum()),
                "null_percentage": round(col_data.isna().sum() / len(df) * 100, 2),
                "unique_count": int(col_data.nunique())
            }
            
            # Add numeric statistics
            if col_data.dtype in ['int64', 'Int64', 'float64']:
                col_stats.update({
                    "min": float(col_data.min()) if pd.notna(col_data.min()) else None,
                    "max": float(col_data.max()) if pd.notna(col_data.max()) else None,
                    "mean": float(col_data.mean()) if pd.notna(col_data.mean()) else None,
                    "median": float(col_data.median()) if pd.notna(col_data.median()) else None,
                    "std": float(col_data.std()) if pd.notna(col_data.std()) else None
                })
            
            stats[col] = col_stats
        
        self.report.column_stats = stats
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full preprocessing pipeline."""
        self.logger.info("Starting preprocessing pipeline...")
        
        # Step 1: Standardize missing values
        df = self.standardize_missing_values(df)
        
        # Step 2: Drop/keep specified columns
        df = self.drop_specified_columns(df)
        
        # Step 3: Drop columns with high missing values
        df = self.drop_columns_with_high_missing(df)
        
        # Step 4: Drop rows with high missing values
        df = self.drop_rows_with_high_missing(df)
        
        # Step 5: Drop rows missing required columns
        df = self.drop_rows_missing_required(df)
        
        # Step 6: Drop rows with any missing values (if strategy is "drop_rows")
        df = self.drop_rows_with_any_missing(df)
        
        # Step 7: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 7: Clean text columns
        df = self.clean_text_columns(df)
        
        # Step 8: Clean numeric columns
        df = self.clean_numeric_columns(df)
        
        # Step 9: Validate chemical structures
        df = self.validate_chemical_structures(df)
        
        # Step 10: Enforce data types
        df = self.enforce_data_types(df)
        
        # Step 11: Remove constant/quasi-constant columns
        df = self.remove_constant_columns(df)
        
        # Step 12: Remove highly correlated features
        df = self.remove_correlated_features(df)
        
        # Step 13: Apply filters
        df = self.apply_filters(df)
        
        # Update report
        self.report.final_rows = len(df)
        self.report.final_columns = len(df.columns)
        
        # Compute column statistics
        self.compute_column_statistics(df)
        
        self.logger.info(f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
        
        return df


def setup_logging(config: dict) -> logging.Logger:
    """Set up logging based on configuration."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("log_level", "INFO").upper())
    
    logger = logging.getLogger("phorce_preprocessing")
    logger.setLevel(log_level)
    
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
        file_handler = logging.FileHandler(log_file)
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
        description="PHORCE Data Preprocessing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --config custom_config.json
    python scripts/preprocess_data.py --input data.csv --output processed.csv
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


def main():
    """Main entry point for the preprocessing script."""
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
    
    # Override verbose setting if specified
    if args.verbose:
        config.setdefault("general_settings", {})["verbose"] = True
        config.setdefault("logging", {})["log_level"] = "DEBUG"
    
    # Set up logging
    logger = setup_logging(config)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Determine input file
    input_file = args.input or config.get("input_file", "P1M.csv")
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output file
    output_file = args.output or config.get("output_file", "data/processed/P1M_preprocessed.csv")
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config, logger)
    
    # Load and preprocess data
    try:
        df = preprocessor.load_data(str(input_path))
        df = preprocessor.preprocess(df)
        
        # Save processed data
        logger.info(f"Saving preprocessed data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} rows to {output_path}")
        
        # Save preprocessing report
        if config.get("general_settings", {}).get("save_preprocessing_report", True):
            report_path = config.get("general_settings", {}).get(
                "report_output", 
                "data/processed/preprocessing_report.json"
            )
            report_full_path = Path(report_path)
            if not report_full_path.is_absolute():
                report_full_path = project_root / report_full_path
            
            report_full_path.parent.mkdir(parents=True, exist_ok=True)
            preprocessor.report.save(str(report_full_path))
            logger.info(f"Preprocessing report saved to {report_full_path}")
        
        # Print summary
        report = preprocessor.report.to_dict()
        summary = report["preprocessing_summary"]
        
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Initial:  {summary['initial_rows']:,} rows × {summary['initial_columns']} columns")
        print(f"Final:    {summary['final_rows']:,} rows × {summary['final_columns']} columns")
        print(f"Removed:  {summary['total_rows_removed']:,} rows ({100-summary['retention_rate_rows']:.1f}%)")
        print(f"Removed:  {summary['total_columns_removed']} columns ({100-summary['retention_rate_columns']:.1f}%)")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
