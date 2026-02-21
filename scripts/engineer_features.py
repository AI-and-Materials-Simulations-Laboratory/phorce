#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Engineering Script for PHORCE Pipeline
===============================================

This script creates additional features based on user-defined configurations.
It supports arithmetic operations, conditional logic, and custom transformations.

Features can be defined in the preprocessing_config.json file under the
"feature_engineering" section.

Usage:
    python scripts/engineer_features.py
    python scripts/engineer_features.py --config custom_config.json
    python scripts/engineer_features.py --input data.csv --output output.csv

Author: PHORCE Pipeline
Date: 2024
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# JSON serialization helpers for numpy/pandas types
def _to_json_serializable(obj):
    try:
        import numpy as _np
        import pandas as _pd
    except Exception:
        _np = None
        _pd = None
    if _np is not None:
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    if _pd is not None:
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, _pd.Timedelta):
            return str(obj)
        if isinstance(obj, _pd.Series) or isinstance(obj, _pd.Index):
            return obj.tolist()
    return str(obj)


# =============================================================================
# Configuration and Data Classes
# =============================================================================

@dataclass
class FeatureDefinition:
    """Definition for a single engineered feature."""
    name: str
    operation: str  # 'add', 'subtract', 'multiply', 'divide', 'formula', 'conditional', 'bin', 'log', 'sqrt', 'power', 'clip'
    columns: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureDefinition':
        """Create FeatureDefinition from dictionary."""
        return cls(
            name=data.get('name', ''),
            operation=data.get('operation', ''),
            columns=data.get('columns', []),
            parameters=data.get('parameters', {}),
            description=data.get('description', '')
        )


@dataclass
class EngineeringReport:
    """Report for feature engineering process."""
    input_file: str
    output_file: str
    timestamp: str
    original_columns: int
    original_rows: int
    features_created: int
    features_failed: int
    final_columns: int
    final_rows: int
    feature_details: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'input_file': self.input_file,
            'output_file': self.output_file,
            'timestamp': self.timestamp,
            'original_shape': {
                'columns': self.original_columns,
                'rows': self.original_rows
            },
            'final_shape': {
                'columns': self.final_columns,
                'rows': self.final_rows
            },
            'features_created': self.features_created,
            'features_failed': self.features_failed,
            'feature_details': self.feature_details,
            'errors': self.errors
        }


# =============================================================================
# Feature Engineering Class
# =============================================================================

class FeatureEngineer:
    """
    Feature engineering class for creating derived features.
    
    Supports various operations:
    - Arithmetic: add, subtract, multiply, divide
    - Mathematical: log, sqrt, power, abs
    - Statistical: clip, normalize, standardize
    - Binning: equal_width, equal_frequency, custom
    - Conditional: if-then-else based on conditions
    - Formula: custom pandas eval expressions
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.fe_config = config.get('feature_engineering', {})
        self.logger = logger or self._setup_logger()
        
        # Extract settings
        self.enabled = self.fe_config.get('enabled', True)
        self.input_file = self.fe_config.get('input_file', 'data/processed/P1M_preprocessed.csv')
        self.output_dir = self.fe_config.get('output_dir', 'data/engineered')
        self.output_file = self.fe_config.get('output_file', 'data/engineered/P1M_engineered.csv')
        self.report_output = self.fe_config.get('report_output', 'data/engineered/feature_engineering_report.json')
        
        # Parse feature definitions
        self.feature_definitions = self._parse_feature_definitions()
        
        # Report tracking
        self.report = None
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger('FeatureEngineer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _parse_feature_definitions(self) -> List[FeatureDefinition]:
        """Parse feature definitions from config."""
        features = []
        feature_list = self.fe_config.get('features', [])
        
        for feat_dict in feature_list:
            try:
                feat = FeatureDefinition.from_dict(feat_dict)
                features.append(feat)
            except Exception as e:
                self.logger.warning(f"Failed to parse feature definition: {e}")
                
        return features
    
    def load_data(self, input_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load input data.
        
        Args:
            input_file: Optional override for input file path
            
        Returns:
            DataFrame with loaded data
        """
        file_path = input_file or self.input_file
        self.logger.info(f"Loading data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
            
        df = pd.read_csv(file_path, low_memory=False)
        self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Apply all feature engineering operations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (engineered DataFrame, list of feature details)
        """
        feature_details = []
        
        for feat_def in self.feature_definitions:
            try:
                self.logger.info(f"Creating feature: {feat_def.name} ({feat_def.operation})")
                df, detail = self._apply_operation(df, feat_def)
                feature_details.append(detail)
                
            except Exception as e:
                error_detail = {
                    'name': feat_def.name,
                    'operation': feat_def.operation,
                    'status': 'failed',
                    'error': str(e)
                }
                feature_details.append(error_detail)
                self.logger.error(f"Failed to create feature '{feat_def.name}': {e}")
                
        return df, feature_details
    
    def _apply_operation(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply a single feature operation.
        
        Args:
            df: Input DataFrame
            feat_def: Feature definition
            
        Returns:
            Tuple of (modified DataFrame, feature detail dict)
        """
        operation = feat_def.operation.lower()
        
        operation_map = {
            'add': self._op_add,
            'subtract': self._op_subtract,
            'multiply': self._op_multiply,
            'divide': self._op_divide,
            'formula': self._op_formula,
            'conditional': self._op_conditional,
            'bin': self._op_bin,
            'log': self._op_log,
            'log10': self._op_log10,
            'sqrt': self._op_sqrt,
            'power': self._op_power,
            'abs': self._op_abs,
            'clip': self._op_clip,
            'normalize': self._op_normalize,
            'standardize': self._op_standardize,
            'ratio': self._op_ratio,
            'interaction': self._op_interaction,
            'polynomial': self._op_polynomial
        }
        
        if operation not in operation_map:
            raise ValueError(f"Unknown operation: {operation}")
            
        return operation_map[operation](df, feat_def)
    
    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------
    
    def _op_add(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Add columns together."""
        cols = feat_def.columns
        self._validate_columns(df, cols)
        
        df[feat_def.name] = df[cols].sum(axis=1)
        
        return df, {
            'name': feat_def.name,
            'operation': 'add',
            'columns': cols,
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"Sum of {', '.join(cols)}"
        }
    
    def _op_subtract(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Subtract second column from first."""
        cols = feat_def.columns
        if len(cols) != 2:
            raise ValueError("Subtract operation requires exactly 2 columns")
        self._validate_columns(df, cols)
        
        df[feat_def.name] = df[cols[0]] - df[cols[1]]
        
        return df, {
            'name': feat_def.name,
            'operation': 'subtract',
            'columns': cols,
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"{cols[0]} - {cols[1]}"
        }
    
    def _op_multiply(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Multiply columns together."""
        cols = feat_def.columns
        self._validate_columns(df, cols)
        
        df[feat_def.name] = df[cols].prod(axis=1)
        
        return df, {
            'name': feat_def.name,
            'operation': 'multiply',
            'columns': cols,
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"Product of {', '.join(cols)}"
        }
    
    def _op_divide(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Divide first column by second."""
        cols = feat_def.columns
        if len(cols) != 2:
            raise ValueError("Divide operation requires exactly 2 columns")
        self._validate_columns(df, cols)
        
        # Handle division by zero
        epsilon = feat_def.parameters.get('epsilon', 1e-10)
        df[feat_def.name] = df[cols[0]] / (df[cols[1]] + epsilon)
        
        return df, {
            'name': feat_def.name,
            'operation': 'divide',
            'columns': cols,
            'parameters': {'epsilon': epsilon},
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"{cols[0]} / {cols[1]}"
        }
    
    def _op_ratio(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Calculate ratio between columns (handles zeros)."""
        cols = feat_def.columns
        if len(cols) != 2:
            raise ValueError("Ratio operation requires exactly 2 columns")
        self._validate_columns(df, cols)
        
        # Safe ratio calculation
        numerator = df[cols[0]].values
        denominator = df[cols[1]].values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(denominator != 0, numerator / denominator, np.nan)
            
        df[feat_def.name] = ratio
        
        return df, {
            'name': feat_def.name,
            'operation': 'ratio',
            'columns': cols,
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"Ratio of {cols[0]} to {cols[1]}"
        }
    
    # -------------------------------------------------------------------------
    # Mathematical Operations
    # -------------------------------------------------------------------------
    
    def _op_log(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Natural log transformation."""
        cols = feat_def.columns
        if len(cols) != 1:
            raise ValueError("Log operation requires exactly 1 column")
        self._validate_columns(df, cols)
        
        offset = feat_def.parameters.get('offset', 1)  # Add offset to handle zeros
        df[feat_def.name] = np.log(df[cols[0]] + offset)
        
        return df, {
            'name': feat_def.name,
            'operation': 'log',
            'columns': cols,
            'parameters': {'offset': offset},
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"log({cols[0]} + {offset})"
        }
    
    def _op_log10(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Log base 10 transformation."""
        cols = feat_def.columns
        if len(cols) != 1:
            raise ValueError("Log10 operation requires exactly 1 column")
        self._validate_columns(df, cols)
        
        offset = feat_def.parameters.get('offset', 1)
        df[feat_def.name] = np.log10(df[cols[0]] + offset)
        
        return df, {
            'name': feat_def.name,
            'operation': 'log10',
            'columns': cols,
            'parameters': {'offset': offset},
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"log10({cols[0]} + {offset})"
        }
    
    def _op_sqrt(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Square root transformation."""
        cols = feat_def.columns
        if len(cols) != 1:
            raise ValueError("Sqrt operation requires exactly 1 column")
        self._validate_columns(df, cols)
        
        # Handle negative values
        df[feat_def.name] = np.sqrt(np.abs(df[cols[0]]))
        
        return df, {
            'name': feat_def.name,
            'operation': 'sqrt',
            'columns': cols,
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"sqrt(|{cols[0]}|)"
        }
    
    def _op_power(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Power transformation."""
        cols = feat_def.columns
        if len(cols) != 1:
            raise ValueError("Power operation requires exactly 1 column")
        self._validate_columns(df, cols)
        
        exponent = feat_def.parameters.get('exponent', 2)
        df[feat_def.name] = np.power(df[cols[0]], exponent)
        
        return df, {
            'name': feat_def.name,
            'operation': 'power',
            'columns': cols,
            'parameters': {'exponent': exponent},
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"{cols[0]}^{exponent}"
        }
    
    def _op_abs(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Absolute value transformation."""
        cols = feat_def.columns
        if len(cols) != 1:
            raise ValueError("Abs operation requires exactly 1 column")
        self._validate_columns(df, cols)
        
        df[feat_def.name] = np.abs(df[cols[0]])
        
        return df, {
            'name': feat_def.name,
            'operation': 'abs',
            'columns': cols,
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"|{cols[0]}|"
        }
    
    # -------------------------------------------------------------------------
    # Statistical Operations
    # -------------------------------------------------------------------------
    
    def _op_clip(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clip values to a range."""
        cols = feat_def.columns
        if len(cols) != 1:
            raise ValueError("Clip operation requires exactly 1 column")
        self._validate_columns(df, cols)
        
        lower = feat_def.parameters.get('lower', None)
        upper = feat_def.parameters.get('upper', None)
        df[feat_def.name] = df[cols[0]].clip(lower=lower, upper=upper)
        
        return df, {
            'name': feat_def.name,
            'operation': 'clip',
            'columns': cols,
            'parameters': {'lower': lower, 'upper': upper},
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"{cols[0]} clipped to [{lower}, {upper}]"
        }
    
    def _op_normalize(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Min-max normalize to [0, 1]."""
        cols = feat_def.columns
        if len(cols) != 1:
            raise ValueError("Normalize operation requires exactly 1 column")
        self._validate_columns(df, cols)
        
        col_min = df[cols[0]].min()
        col_max = df[cols[0]].max()
        
        if col_max - col_min == 0:
            df[feat_def.name] = 0
        else:
            df[feat_def.name] = (df[cols[0]] - col_min) / (col_max - col_min)
        
        return df, {
            'name': feat_def.name,
            'operation': 'normalize',
            'columns': cols,
            'parameters': {'min': float(col_min), 'max': float(col_max)},
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"{cols[0]} normalized to [0, 1]"
        }
    
    def _op_standardize(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Z-score standardization."""
        cols = feat_def.columns
        if len(cols) != 1:
            raise ValueError("Standardize operation requires exactly 1 column")
        self._validate_columns(df, cols)
        
        col_mean = df[cols[0]].mean()
        col_std = df[cols[0]].std()
        
        if col_std == 0:
            df[feat_def.name] = 0
        else:
            df[feat_def.name] = (df[cols[0]] - col_mean) / col_std
        
        return df, {
            'name': feat_def.name,
            'operation': 'standardize',
            'columns': cols,
            'parameters': {'mean': float(col_mean), 'std': float(col_std)},
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"{cols[0]} z-score standardized"
        }
    
    # -------------------------------------------------------------------------
    # Binning Operations
    # -------------------------------------------------------------------------
    
    def _op_bin(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Bin continuous values into categories."""
        cols = feat_def.columns
        if len(cols) != 1:
            raise ValueError("Bin operation requires exactly 1 column")
        self._validate_columns(df, cols)
        
        method = feat_def.parameters.get('method', 'equal_width')
        n_bins = feat_def.parameters.get('n_bins', 5)
        labels = feat_def.parameters.get('labels', None)
        bins = feat_def.parameters.get('bins', None)  # Custom bin edges
        
        if bins is not None:
            # Custom bins
            df[feat_def.name] = pd.cut(df[cols[0]], bins=bins, labels=labels, include_lowest=True)
        elif method == 'equal_width':
            df[feat_def.name] = pd.cut(df[cols[0]], bins=n_bins, labels=labels)
        elif method == 'equal_frequency':
            df[feat_def.name] = pd.qcut(df[cols[0]], q=n_bins, labels=labels, duplicates='drop')
        else:
            raise ValueError(f"Unknown binning method: {method}")
        
        return df, {
            'name': feat_def.name,
            'operation': 'bin',
            'columns': cols,
            'parameters': {'method': method, 'n_bins': n_bins, 'custom_bins': bins},
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"{cols[0]} binned ({method})"
        }
    
    # -------------------------------------------------------------------------
    # Advanced Operations
    # -------------------------------------------------------------------------
    
    def _op_formula(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply a custom formula using pandas eval."""
        formula = feat_def.parameters.get('formula', '')
        if not formula:
            raise ValueError("Formula operation requires 'formula' parameter")
        
        # Evaluate the formula
        df[feat_def.name] = df.eval(formula)
        
        return df, {
            'name': feat_def.name,
            'operation': 'formula',
            'parameters': {'formula': formula},
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"Formula: {formula}"
        }
    
    def _op_conditional(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create feature based on conditions."""
        conditions = feat_def.parameters.get('conditions', [])
        default = feat_def.parameters.get('default', 0)
        
        if not conditions:
            raise ValueError("Conditional operation requires 'conditions' parameter")
        
        # Start with default value
        result = pd.Series([default] * len(df), index=df.index)
        
        # Apply conditions in reverse order (last condition takes precedence)
        for cond in reversed(conditions):
            condition_str = cond.get('condition', '')
            value = cond.get('value', default)
            
            # Evaluate condition
            mask = df.eval(condition_str)
            result = result.where(~mask, value)
        
        df[feat_def.name] = result
        
        return df, {
            'name': feat_def.name,
            'operation': 'conditional',
            'parameters': {'conditions': conditions, 'default': default},
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"Conditional feature with {len(conditions)} conditions"
        }
    
    def _op_interaction(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create interaction term between columns."""
        cols = feat_def.columns
        if len(cols) < 2:
            raise ValueError("Interaction operation requires at least 2 columns")
        self._validate_columns(df, cols)
        
        # Multiply all columns together
        df[feat_def.name] = df[cols].prod(axis=1)
        
        return df, {
            'name': feat_def.name,
            'operation': 'interaction',
            'columns': cols,
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"Interaction: {' × '.join(cols)}"
        }
    
    def _op_polynomial(self, df: pd.DataFrame, feat_def: FeatureDefinition) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create polynomial features."""
        cols = feat_def.columns
        if len(cols) != 1:
            raise ValueError("Polynomial operation requires exactly 1 column")
        self._validate_columns(df, cols)
        
        degree = feat_def.parameters.get('degree', 2)
        df[feat_def.name] = np.power(df[cols[0]], degree)
        
        return df, {
            'name': feat_def.name,
            'operation': 'polynomial',
            'columns': cols,
            'parameters': {'degree': degree},
            'status': 'success',
            'non_null_count': df[feat_def.name].notna().sum(),
            'description': feat_def.description or f"{cols[0]}^{degree}"
        }
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def _validate_columns(self, df: pd.DataFrame, columns: List[str]) -> None:
        """Validate that required columns exist."""
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    
    def save_results(self, df: pd.DataFrame, output_file: Optional[str] = None) -> str:
        """
        Save engineered data.
        
        Args:
            df: DataFrame to save
            output_file: Optional override for output path
            
        Returns:
            Path to saved file
        """
        output_path = output_file or self.output_file
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save data
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved engineered data to {output_path}")
        
        return output_path
    
    def save_report(self, report: EngineeringReport, output_file: Optional[str] = None) -> str:
        """
        Save engineering report.
        
        Args:
            report: EngineeringReport instance
            output_file: Optional override for report path
            
        Returns:
            Path to saved report
        """
        output_path = output_file or self.report_output
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save report (use custom serializer for numpy / pandas types)
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=_to_json_serializable)
            
        self.logger.info(f"Saved report to {output_path}")
        
        return output_path
    
    def run(self, input_file: Optional[str] = None, output_file: Optional[str] = None) -> Tuple[pd.DataFrame, EngineeringReport]:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            input_file: Optional override for input file
            output_file: Optional override for output file
            
        Returns:
            Tuple of (engineered DataFrame, report)
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Feature Engineering Pipeline")
        self.logger.info("=" * 60)
        
        # Load data
        df = self.load_data(input_file)
        original_cols = len(df.columns)
        original_rows = len(df)
        
        # Apply feature engineering
        df, feature_details = self.engineer_features(df)
        
        # Count successes and failures
        features_created = sum(1 for d in feature_details if d.get('status') == 'success')
        features_failed = sum(1 for d in feature_details if d.get('status') == 'failed')
        
        # Create report
        report = EngineeringReport(
            input_file=input_file or self.input_file,
            output_file=output_file or self.output_file,
            timestamp=datetime.now().isoformat(),
            original_columns=original_cols,
            original_rows=original_rows,
            features_created=features_created,
            features_failed=features_failed,
            final_columns=len(df.columns),
            final_rows=len(df),
            feature_details=feature_details,
            errors=[d.get('error', '') for d in feature_details if d.get('status') == 'failed']
        )
        
        # Save outputs
        self.save_results(df, output_file)
        self.save_report(report)
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("Feature Engineering Complete")
        self.logger.info(f"  Features created: {features_created}")
        self.logger.info(f"  Features failed: {features_failed}")
        self.logger.info(f"  Original columns: {original_cols}")
        self.logger.info(f"  Final columns: {len(df.columns)}")
        self.logger.info("=" * 60)
        
        return df, report


# =============================================================================
# Main Entry Point
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Feature Engineering Script for PHORCE Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/engineer_features.py
    python scripts/engineer_features.py --config custom_config.json
    python scripts/engineer_features.py --input data.csv --output output.csv
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='preprocessing_config.json',
        help='Path to configuration file (default: preprocessing_config.json)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Override input file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Override output file path'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('FeatureEngineer')
    
    # Load config
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
        
    config = load_config(args.config)
    
    # Check if feature engineering is enabled
    if not config.get('feature_engineering', {}).get('enabled', True):
        logger.info("Feature engineering is disabled in config. Exiting.")
        sys.exit(0)
    
    # Run feature engineering
    engineer = FeatureEngineer(config, logger)
    df, report = engineer.run(
        input_file=args.input,
        output_file=args.output
    )
    
    logger.info("Feature engineering completed successfully!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
