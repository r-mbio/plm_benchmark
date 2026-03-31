"""
EC Benchmark Configuration
==========================
Unified configuration system that loads from config.yaml.

All settings are centralized in config.yaml for easy customization.
This module provides:
- Config loading from YAML
- Path validation
- Default fallbacks
- Type-safe access to settings

Usage:
    from config import Config
    cfg = Config()  # Auto-loads config.yaml from project root
    cfg = Config("/path/to/custom/config.yaml")  # Custom config
    
    # Access settings
    print(cfg.embeddings['esm2_650m']['path'])
    print(cfg.training['batch_size'])

Author: Rajesh Sathyamoorthy
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field


# =============================================================================
# DEFAULT CONFIGURATION (used if config.yaml not found)
# =============================================================================

DEFAULT_CONFIG = {
    'paths': {
        'project_root': None,
        'data_dir': 'data',
        'model_dir': 'models',
        'result_dir': 'results',
        'figure_dir': 'figures',
        'log_dir': 'logs',
    },
    'embeddings': {
        'esm2_650m': {
            'path': 'data/embeddings/esm2_650m.h5',
            'id_key': 'ids',
            'emb_key': 'embeddings',
            'dim': 1280,
            'display_name': 'ESM2-650M',
        },
    },
    'labels': {
        'path': 'data/labels/ec_labels.npz',
        'id_key': 'uniprot_ids',
        'ec4_key': 'ec4_multilabel',
        'classes_key': 'ec4_classes',
    },
    'clusters': {
        50: {'path': 'data/clusters/clusters_50pct.tsv'},
    },
    'experiment': {
        'plms': ['esm2_650m'],
        'ec_levels': ['ec1', 'ec2', 'ec3', 'ec4'],
        'thresholds': [30, 50, 70, 90],
        'architectures': ['mlp', 'deep_mlp', 'wide_mlp', 'attention_mlp'],
    },
    'training': {
        'batch_size': 2048,
        'epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'dropout': 0.3,
        'early_stop_patience': 10,
        'min_samples_per_class': 10,
        'train_ratio': 0.8,
        'num_replicates': 3,
        'seeds': [42, 123, 456],
        'num_workers': 8,
        'pin_memory': True,
        'use_amp': True,
        'amp_dtype': 'bfloat16',
    },
    'hardware': {
        'device': 'cuda',
        'gpu_id': 0,
    },
}


# =============================================================================
# CONFIG CLASS
# =============================================================================

class Config:
    """
    Unified configuration manager for EC Benchmark.
    
    Loads settings from config.yaml and provides type-safe access.
    Falls back to defaults if config.yaml not found.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml. If None, searches in:
                1. Current directory
                2. Project root (parent of src/)
                3. Uses defaults
        """
        self._config = {}
        self._config_path = None
        self._project_root = None
        
        # Find and load config
        if config_path:
            self._load_config(Path(config_path))
        else:
            self._find_and_load_config()
        
        # Set project root
        self._set_project_root()
        
        # Validate and resolve paths
        self._resolve_paths()
    
    def _find_and_load_config(self):
        """Search for config.yaml in standard locations."""
        search_paths = []

        # Highest priority: explicit environment override.
        env_config = os.environ.get('EC_BENCHMARK_CONFIG')
        if env_config:
            search_paths.append(Path(env_config))

        search_paths.extend([
            Path.cwd() / 'config.yaml',
            Path(__file__).parent.parent / 'config.yaml',
            Path.home() / '.ec_benchmark' / 'config.yaml',
        ])
        
        for path in search_paths:
            if path.exists():
                self._load_config(path)
                return
        
        # Use defaults
        print("Warning: config.yaml not found, using defaults")
        self._config = DEFAULT_CONFIG.copy()
    
    def _load_config(self, path: Path):
        """Load configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        self._config_path = path
        print(f"Loaded config from: {path}")
    
    def _set_project_root(self):
        """Determine project root directory."""
        if self._config.get('paths', {}).get('project_root'):
            self._project_root = Path(self._config['paths']['project_root'])
        elif self._config_path:
            self._project_root = self._config_path.parent
        else:
            self._project_root = Path(__file__).parent.parent
    
    def _resolve_paths(self):
        """Resolve relative paths to absolute paths."""
        # Resolve output directories
        paths = self._config.get('paths', {})
        for key in ['model_dir', 'result_dir', 'figure_dir', 'log_dir']:
            if key in paths and paths[key]:
                p = Path(paths[key])
                if not p.is_absolute():
                    paths[key] = str(self._project_root / p)
    
    # -------------------------------------------------------------------------
    # PROPERTY ACCESSORS
    # -------------------------------------------------------------------------
    
    @property
    def project_root(self) -> Path:
        """Project root directory."""
        return self._project_root
    
    @property
    def paths(self) -> Dict[str, str]:
        """All path settings."""
        return self._config.get('paths', DEFAULT_CONFIG['paths'])
    
    @property
    def embeddings(self) -> Dict[str, Dict[str, Any]]:
        """Embedding file configurations."""
        return self._config.get('embeddings', DEFAULT_CONFIG['embeddings'])
    
    @property
    def labels(self) -> Dict[str, str]:
        """Label file configuration."""
        return self._config.get('labels', DEFAULT_CONFIG['labels'])
    
    @property
    def clusters(self) -> Dict[int, Dict[str, str]]:
        """Cluster file configurations."""
        clusters = self._config.get('clusters', DEFAULT_CONFIG['clusters'])
        # Ensure keys are integers
        return {int(k): v for k, v in clusters.items()}
    
    @property
    def experiment(self) -> Dict[str, Any]:
        """Experiment settings."""
        return self._config.get('experiment', DEFAULT_CONFIG['experiment'])
    
    @property
    def training(self) -> Dict[str, Any]:
        """Training configuration."""
        return self._config.get('training', DEFAULT_CONFIG['training'])
    
    @property
    def hardware(self) -> Dict[str, Any]:
        """Hardware settings."""
        return self._config.get('hardware', DEFAULT_CONFIG['hardware'])
    
    @property
    def model(self) -> Dict[str, Any]:
        """Model architecture settings."""
        return self._config.get('model', {})
    
    @property
    def evaluation(self) -> Dict[str, Any]:
        """Evaluation settings."""
        return self._config.get('evaluation', {})
    
    @property
    def figures(self) -> Dict[str, Any]:
        """Figure settings."""
        return self._config.get('figures', {})
    
    # -------------------------------------------------------------------------
    # CONVENIENCE METHODS
    # -------------------------------------------------------------------------
    
    def get_embedding_path(self, plm: str) -> Path:
        """Get path to embedding file for a PLM."""
        return Path(self.embeddings[plm]['path'])
    
    def get_embedding_dim(self, plm: str) -> int:
        """Get embedding dimension for a PLM."""
        return self.embeddings[plm]['dim']
    
    def get_embedding_id_key(self, plm: str) -> str:
        """Get ID key for a PLM's embedding file."""
        return self.embeddings[plm].get('id_key', 'ids')
    
    def get_cluster_path(self, threshold: int) -> Path:
        """Get path to cluster file for a threshold."""
        return Path(self.clusters[threshold]['path'])
    
    def get_label_path(self) -> Path:
        """Get path to label file."""
        return Path(self.labels['path'])
    
    def get_model_dir(self) -> Path:
        """Get model output directory."""
        return Path(self.paths.get('model_dir', 'models'))
    
    def get_result_dir(self) -> Path:
        """Get results output directory."""
        return Path(self.paths.get('result_dir', 'results'))
    
    def get_figure_dir(self) -> Path:
        """Get figures output directory."""
        return Path(self.paths.get('figure_dir', 'figures'))
    
    def get_plms(self) -> List[str]:
        """Get list of PLMs to benchmark."""
        return self.experiment.get('plms', ['esm2_650m'])
    
    def get_ec_levels(self) -> List[str]:
        """Get list of EC levels to evaluate."""
        return self.experiment.get('ec_levels', ['ec1', 'ec2', 'ec3', 'ec4'])
    
    def get_thresholds(self) -> List[int]:
        """Get list of sequence identity thresholds."""
        return self.experiment.get('thresholds', [30, 50, 70, 90])
    
    def get_architectures(self) -> List[str]:
        """Get list of architectures to evaluate."""
        return self.experiment.get('architectures', ['mlp'])
    
    def get_device(self) -> str:
        """Get compute device."""
        return self.hardware.get('device', 'cuda')
    
    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------
    
    def validate(self) -> bool:
        """
        Validate all required files exist.
        
        Returns:
            True if all files exist, False otherwise
        """
        missing = []
        
        # Check embedding files
        for plm in self.get_plms():
            path = self.get_embedding_path(plm)
            if not path.exists():
                missing.append(f"Embedding ({plm}): {path}")
        
        # Check label file
        label_path = self.get_label_path()
        if not label_path.exists():
            missing.append(f"Labels: {label_path}")
        
        # Check cluster files
        for threshold in self.get_thresholds():
            path = self.get_cluster_path(threshold)
            if not path.exists():
                missing.append(f"Clusters ({threshold}%): {path}")
        
        if missing:
            print("Missing files:")
            for f in missing:
                print(f"  - {f}")
            return False
        
        print("All required data files found.")
        return True
    
    def setup_dirs(self):
        """Create all output directories."""
        for dir_path in [self.get_model_dir(), self.get_result_dir(), 
                         self.get_figure_dir(), Path(self.paths.get('log_dir', 'logs'))]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create PLM-specific model directories
        for plm in self.get_plms():
            (self.get_model_dir() / plm).mkdir(exist_ok=True)
    
    def __repr__(self) -> str:
        return f"Config(path={self._config_path}, project_root={self._project_root})"


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================
# These provide backward compatibility with the old config.py interface

# Global config instance (lazy loaded)
_global_config: Optional[Config] = None

def get_config() -> Config:
    """Get global config instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def setup_dirs():
    """Backward-compatible module function used by legacy scripts."""
    return get_config().setup_dirs()


def validate_config() -> bool:
    """Validate configured inputs exist."""
    return get_config().validate()


# Legacy constants (for backward compatibility)
def _get_legacy_constant(name: str):
    """Get legacy constant from config."""
    cfg = get_config()
    
    if name == 'PROJECT_ROOT':
        return cfg.project_root
    elif name == 'DATA_DIR':
        return Path(cfg.paths.get('data_dir', 'data'))
    elif name == 'MODEL_DIR':
        return cfg.get_model_dir()
    elif name == 'RESULT_DIR':
        return cfg.get_result_dir()
    elif name == 'FIGURE_DIR':
        return cfg.get_figure_dir()
    elif name == 'EMBEDDINGS':
        return {k: Path(v['path']) for k, v in cfg.embeddings.items()}
    elif name == 'EMBEDDING_ID_KEYS':
        return {k: v.get('id_key', 'ids') for k, v in cfg.embeddings.items()}
    elif name == 'EMBEDDING_DIMS':
        return {k: v['dim'] for k, v in cfg.embeddings.items()}
    elif name == 'PLM_DISPLAY_NAMES':
        return {k: v.get('display_name', k) for k, v in cfg.embeddings.items()}
    elif name == 'CLUSTERS':
        return {k: Path(v['path']) for k, v in cfg.clusters.items()}
    elif name == 'LABEL_FILE':
        return cfg.get_label_path()
    elif name == 'PLMS':
        return cfg.get_plms()
    elif name == 'EC_LEVELS':
        return cfg.get_ec_levels()
    elif name == 'THRESHOLDS':
        return cfg.get_thresholds()
    elif name == 'ARCHITECTURES':
        return cfg.get_architectures()
    elif name == 'TRAIN_CONFIG':
        return cfg.training
    else:
        raise AttributeError(f"Unknown config constant: {name}")


# Create module-level properties for backward compatibility
class _LegacyConfigModule:
    """Module wrapper for backward compatibility."""
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return _get_legacy_constant(name)


# Replace module with wrapper for legacy access
import sys
_original_module = sys.modules[__name__]

class _ConfigModule:
    """Combined module with both new and legacy interfaces."""
    
    def __init__(self, module):
        self._module = module
    
    def __getattr__(self, name):
        # First try the original module
        try:
            return getattr(self._module, name)
        except AttributeError:
            pass
        
        # Then try legacy constants
        try:
            return _get_legacy_constant(name)
        except AttributeError:
            raise AttributeError(f"module 'config' has no attribute '{name}'")

sys.modules[__name__] = _ConfigModule(_original_module)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test configuration
    cfg = Config()
    print(f"Project root: {cfg.project_root}")
    print(f"PLMs: {cfg.get_plms()}")
    print(f"EC levels: {cfg.get_ec_levels()}")
    print(f"Thresholds: {cfg.get_thresholds()}")
    print(f"Architectures: {cfg.get_architectures()}")
    print(f"Training config: {cfg.training}")
    print()
    cfg.validate()
