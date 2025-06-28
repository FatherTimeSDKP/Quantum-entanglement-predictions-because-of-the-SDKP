# Image Analysis
import json
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time
from itertools import product
import warnings
warnings.filterwarnings(‘ignore’, category=RuntimeWarning)

@dataclass
class SensitivityResults:
“”“Container for sensitivity analysis results”””
parameter_impacts: Dict[str, float]
interaction_effects: Dict[str, float]
component_sensitivities: Dict[str, Dict[str, float]]
correlation_matrix: np.ndarray
parameter_names: List[str]
total_variance_explained: float

@dataclass
class OptimizedSimulationResults:
“”“Container for optimized simulation results with performance metrics”””
entanglement_values: List[float]
correlation_coefficients: List[float]
numerical_mappings: Dict[str, Any]
lambda_contributions: Dict[str, float]
polarization_data: List[Dict[str, Any]]
sensitivity_analysis: Optional[SensitivityResults]
performance_metrics: Dict[str, Any]

class OptimizedQuantumEntanglementSimulator:
“””
Enhanced quantum entanglement simulator with sensitivity analysis and optimization
“””

```
def __init__(self, config: Dict[str, Any]):
    self.config = config
    self.setup_logging()
    self.validate_config()
    self._precompute_cache = {}
    
def setup_logging(self):
    """Initialize logging with performance tracking"""
    if self.config.get('parameters', {}).get('logging', False):
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    else:
        self.logger = None
        
def validate_config(self):
    """Enhanced configuration validation"""
    required_keys = ['polarization_angles_deg', 'numerical_signatures', 'lambda_weights']
    params = self.config.get('parameters', {})
    
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
            
    # Validate and normalize lambda weights
    weights = params['lambda_weights']
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        self.log(f"Normalizing lambda weights from {total_weight:.6f} to 1.0")
        for key in weights:
            weights[key] /= total_weight
            
    # Validate required weight keys
    required_weight_keys = ['C_SDN', 'VEI_delta', 'QF_delta']
    for key in required_weight_keys:
        if key not in weights:
            raise ValueError(f"Missing lambda weight: {key}")

def log(self, message: str, level: str = 'info'):
    """Enhanced logging with levels"""
    if self.logger:
        getattr(self.logger, level.lower())(message)
    elif level.lower() in ['warning', 'error']:
        print(f"[{level.upper()}] {message}")

@staticmethod
def calculate_numerical_signature_value_vectorized(signatures: List[str]) -> np.ndarray:
    """Vectorized numerical signature calculation for performance"""
    def process_single_signature(signature: str) -> float:
        clean_signature = ''.join(filter(str.isdigit, str(signature)))
        if not clean_signature:
            return 0.5
            
        digit_sum = sum(int(d) for d in clean_signature)
        positional_weight = sum(int(d) * (i + 1) for i, d in enumerate(clean_signature))
        length_factor = len(clean_signature) / 10.0
        normalized_value = (digit_sum + positional_weight * 0.1 + length_factor) % 1.0
        return normalized_value
    
    return np.array([process_single_signature(sig) for sig in signatures])

def calculate_numerical_signature_value(self, signature: str) -> float:
    """Original method maintained for compatibility"""
    if signature in self._precompute_cache:
        return self._precompute_cache[signature]
        
    clean_signature = ''.join(filter(str.isdigit, str(signature)))
    if not clean_signature:
        result = 0.5
    else:
        digit_sum = sum(int(d) for d in clean_signature)
        positional_weight = sum(int(d) * (i + 1) for i, d in enumerate(clean_signature))
        length_factor = len(clean_signature) / 10.0
        result = (digit_sum + positional_weight * 0.1 + length_factor) % 1.0
    
    self._precompute_cache[signature] = result
    return result

def apply_mapping_function(self, value: float, mapping_type: str) -> float:
    """Apply transformation mapping with caching"""
    cache_key = (value, mapping_type)
    if cache_key in self._precompute_cache:
        return self._precompute_cache[cache_key]
        
    value = max(0.0, min(1.0, value))
    
    if mapping_type == 'linear':
        result = value
    elif mapping_type == 'nonlinear':
        result = np.sin(value * np.pi) ** 2
    elif mapping_type == 'hybrid':
        result = 0.5 * value + 0.5 * np.sin(value * np.pi) ** 2
    else:
        result = value
        
    self._precompute_cache[cache_key] = result
    return result

def calculate_entanglement_measure_optimized(self, angle_deg: float, signature: str, 
                                           detailed_logging: bool = False) -> Tuple[float, Dict[str, float]]:
    """Optimized entanglement calculation with optional detailed logging"""
    if detailed_logging:
        return self.calculate_entanglement_measure_detailed(angle_deg, signature)
    
    # Fast calculation without logging
    angle_rad = math.radians(angle_deg)
    signature_value = self.calculate_numerical_signature_value(signature)
    
    # SDKP time τ_s
    use_sdkp_time = self.config['parameters'].get('use_sdkp_time', False)
    if use_sdkp_time:
        size = self.config['parameters'].get('sdkp_size', 1.0)
        density = self.config['parameters'].get('sdkp_density', 1.0)
        rotation_velocity = self.config['parameters'].get('sdkp_rotation_velocity', 1.0)
        tau_s = size * density * rotation_velocity
    else:
        tau_s = 0.0

    # Component calculations
    C_SDN = abs(np.cos(angle_rad)) * signature_value
    C_V = (2 * signature_value) - 1
    VEI_delta = 1 - abs(C_V)
    
    qf_angle = angle_rad * 2 + tau_s if use_sdkp_time else angle_rad * 2
    QF_delta = abs(np.sin(qf_angle) - signature_value)
    
    # QCC entropy
    use_qcc_entropy = self.config['parameters'].get('use_qcc_entropy', False)
    if use_qcc_entropy:
        entropy = self.config['parameters'].get('qcc_entropy', 1.0)
        epsilon_qcc = entropy / tau_s if use_sdkp_time and tau_s > 0 else entropy
    else:
        epsilon_qcc = 1.0
        
    QF_delta_weighted = QF_delta * epsilon_qcc
    
    # Apply mapping
    mapping_type = self.config['parameters'].get('mapping_type', 'linear')
    C_SDN_mapped = self.apply_mapping_function(C_SDN, mapping_type)
    VEI_delta_mapped = self.apply_mapping_function(VEI_delta, mapping_type)
    QF_delta_mapped = self.apply_mapping_function(QF_delta_weighted, mapping_type)
    
    # Weighted combination
    weights = self.config['parameters']['lambda_weights']
    entanglement_value = (
        weights['C_SDN'] * C_SDN_mapped +
        weights['VEI_delta'] * VEI_delta_mapped +
        weights['QF_delta'] * QF_delta_mapped
    )
    
    components = {
        'C_SDN_raw': C_SDN,
        'VEI_delta_raw': VEI_delta,
        'QF_delta_raw': QF_delta,
        'QF_delta_weighted': QF_delta_weighted,
        'C_SDN_mapped': C_SDN_mapped,
        'VEI_delta_mapped': VEI_delta_mapped,
        'QF_delta_mapped': QF_delta_mapped,
        'signature_value': signature_value,
        'tau_s': tau_s,
        'epsilon_qcc': epsilon_qcc
    }
    
    return entanglement_value, components

def calculate_entanglement_measure_detailed(self, angle_deg: float, signature: str) -> Tuple[float, Dict[str, float]]:
    """Your original detailed method with full logging"""
    self.log(f"\n--- Calculating Entanglement for Angle {angle_deg}° and Signature '{signature}' ---")
    
    angle_rad = math.radians(angle_deg)
    self.log(f"Step 1: Convert angle {angle_deg}° to radians: {angle_rad:.4f} rad (radians are dimensionless angles used for trig functions)")
    
    signature_value = self.calculate_numerical_signature_value(signature)
    self.log(f"Step 2: Calculated numerical signature value for '{signature}': {signature_value:.4f} (dimensionless normalized value derived from signature digits)")
    
    # SDKP time τ_s: Size × Density × Rotation Velocity
    use_sdkp_time = self.config['parameters'].get('use_sdkp_time', False)
    if use_sdkp_time:
        size = self.config['parameters'].get('sdkp_size', 1.0)  # meters (m)
        density = self.config['parameters'].get('sdkp_density', 1.0)  # kg/m³
        rotation_velocity = self.config['parameters'].get('sdkp_rotation_velocity', 1.0)  # radians/second (rad/s)
        tau_s = size * density * rotation_velocity
        self.log(f"SDKP time (τ_s) calculated: τ_s = Size (m) × Density (kg/m³) × Rotation Velocity (rad/s) = {tau_s:.4f} [units: m·kg/m³·rad/s]")
        self.log("Note: τ_s has composite units, representing a scale of time within SDKP framework")
    else:
        tau_s = 0.0
        self.log("SDKP time (τ_s) disabled; using standard angular calculations without SDKP time scaling.")

    # C_SDN — dimensionless correlation between polarization angle and signature
    C_SDN = abs(np.cos(angle_rad)) * signature_value
    self.log(f"Step 3a (C_SDN): |cos(angle_rad)| × signature_value = {C_SDN:.4f} (dimensionless correlation factor)")

    # VEI_delta — vibrational mismatch (dimensionless)
    C_V = (2 * signature_value) - 1  # maps [0,1] to [-1,1]
    VEI_delta = 1 - abs(C_V)
    self.log(f"Step 3b (VEI_delta): 1 - |2×signature_value - 1| = {VEI_delta:.4f} (dimensionless vibrational entanglement index)")

    # QF_delta — quantum number flow mismatch
    qf_angle = angle_rad * 2 + tau_s if use_sdkp_time else angle_rad * 2
    self.log(f"Step 3c (QF_delta) calculation uses angle doubled and adds τ_s phase shift if enabled:")
    self.log(f"  qf_angle = angle_rad × 2 {'+ τ_s' if use_sdkp_time else ''} = {qf_angle:.4f} radians")
    QF_delta = abs(np.sin(qf_angle) - signature_value)
    self.log(f"QF_delta = |sin(qf_angle) - signature_value| = {QF_delta:.4f} (dimensionless mismatch)")

    # QCC entropy density ε_QCC (bits per SDKP time)
    use_qcc_entropy = self.config['parameters'].get('use_qcc_entropy', False)
    if use_qcc_entropy:
        entropy = self.config['parameters'].get('qcc_entropy', 1.0)  # bits of entropy
        if use_sdkp_time and tau_s > 0:
            epsilon_qcc = entropy / tau_s
            self.log(f"QCC entropy density (ε_QCC) = entropy (bits) / τ_s = {entropy:.4f} bits / {tau_s:.4f} SDKP time units = {epsilon_qcc:.4f} bits/SDKP time")
        else:
            epsilon_qcc = entropy
            self.log(f"QCC entropy (ε_QCC) used as raw bits: {epsilon_qcc:.4f} bits (no SDKP time scaling)")
    else:
        epsilon_qcc = 1.0
        self.log("QCC entropy weighting disabled; ε_QCC set to 1 (no effect).")

    # Apply ε_QCC as weighting multiplier on QF_delta (dimensionless)
    QF_delta_weighted = QF_delta * epsilon_qcc
    self.log(f"Weighted QF_delta with ε_QCC: {QF_delta:.4f} × {epsilon_qcc:.4f} = {QF_delta_weighted:.4f} (dimensionless)")

    # Apply mapping transformations to each component
    mapping_type = self.config['parameters'].get('mapping_type', 'linear')
    self.log(f"Step 4: Applying '{mapping_type}' mapping function to components.")

    C_SDN_mapped = self.apply_mapping_function(C_SDN, mapping_type)
    self.log(f"  Mapped C_SDN: {C_SDN:.4f} -> {C_SDN_mapped:.4f}")

    VEI_delta_mapped = self.apply_mapping_function(VEI_delta, mapping_type)
    self.log(f"  Mapped VEI_delta: {VEI_delta:.4f} -> {VEI_delta_mapped:.4f}")

    QF_delta_mapped = self.apply_mapping_function(QF_delta_weighted, mapping_type)
    self.log(f"  Mapped Weighted QF_delta: {QF_delta_weighted:.4f} -> {QF_delta_mapped:.4f}")

    # Retrieve lambda weights for weighted sum
    weights = self.config['parameters']['lambda_weights']
    self.log(f"Step 5: Applying lambda weights: C_SDN={weights['C_SDN']:.2f}, VEI_delta={weights['VEI_delta']:.2f}, QF_delta={weights['QF_delta']:.2f}")

    # Final entanglement value as weighted sum of mapped components
    entanglement_value = (
        weights['C_SDN'] * C_SDN_mapped +
        weights['VEI_delta'] * VEI_delta_mapped +
        weights['QF_delta'] * QF_delta_mapped
    )
    self.log(f"  Calculation: ({weights['C_SDN']:.2f} × {C_SDN_mapped:.4f}) + "
             f"({weights['VEI_delta']:.2f} × {VEI_delta_mapped:.4f}) + "
             f"({weights['QF_delta']:.2f} × {QF_delta_mapped:.4f})")
    self.log(f"  Final Entanglement Value: {entanglement_value:.4f} (dimensionless)")

    components = {
        'C_SDN_raw': C_SDN,
        'VEI_delta_raw': VEI_delta,
        'QF_delta_raw': QF_delta,
        'QF_delta_weighted': QF_delta_weighted,
        'C_SDN_mapped': C_SDN_mapped,
        'VEI_delta_mapped': VEI_delta_mapped,
        'QF_delta_mapped': QF_delta_mapped,
        'signature_value': signature_value,
        'tau_s': tau_s,
        'epsilon_qcc': epsilon_qcc
    }

    return entanglement_value, components

def parameter_sensitivity_analysis(self, sample_size: int = 1000) -> SensitivityResults:
    """
    Comprehensive parameter sensitivity analysis using Latin Hypercube Sampling
    and Sobol sensitivity analysis
    """
    self.log("Starting comprehensive parameter sensitivity analysis...")
    start_time = time.time()
    
    # Define parameter ranges for sensitivity analysis
    param_ranges = {
        'C_SDN_weight': (0.1, 0.8),
        'VEI_delta_weight': (0.1, 0.8), 
        'QF_delta_weight': (0.1, 0.8),
        'sdkp_size': (0.1, 10.0),
        'sdkp_density': (0.1, 10.0),
        'sdkp_rotation_velocity': (0.1, 10.0),
        'qcc_entropy': (0.1, 5.0),
        'angle_deg': (0, 180),
        'signature_complexity': (1, 15)  # Number of digits in signature
    }
    
    # Generate Latin Hypercube samples
    n_params = len(param_ranges)
    lhs_samples = self._latin_hypercube_sampling(sample_size, n_params)
    
    # Scale samples to parameter ranges
    param_names = list(param_ranges.keys())
    scaled_samples = np.zeros_like(lhs_samples)
    
    for i, param_name in enumerate(param_names):
        min_val, max_val = param_ranges[param_name]
        scaled_samples[:, i] = min_val + lhs_samples[:, i] * (max_val - min_val)
    
    # Calculate entanglement values for all samples
    entanglement_values = []
    component_values = {comp: [] for comp in ['C_SDN_mapped', 'VEI_delta_mapped', 'QF_delta_mapped']}
    
    for sample in scaled_samples:
        # Create temporary config for this sample
        temp_config = self._create_temp_config_from_sample(sample, param_names)
        temp_simulator = OptimizedQuantumEntanglementSimulator(temp_config)
        
        # Calculate entanglement for sample
        angle = sample[param_names.index('angle_deg')]
        sig_complexity = int(sample[param_names.index('signature_complexity')])
        signature = '1' * sig_complexity  # Simple signature of given complexity
        
        entanglement_val, components = temp_simulator.calculate_entanglement_measure_optimized(
            angle, signature, detailed_logging=False
        )
        
        entanglement_values.append(entanglement_val)
        for comp_name in component_values:
            component_values[comp_name].append(components[comp_name])
    
    entanglement_values = np.array(entanglement_values)
    
    # Calculate Sobol indices (first-order sensitivity indices)
    parameter_impacts = {}
    for i, param_name in enumerate(param_names):
        sensitivity = self._calculate_sobol_index(scaled_samples[:, i], entanglement_values)
        parameter_impacts[param_name] = float(sensitivity)
    
    # Calculate interaction effects (second-order)
    interaction_effects = {}
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            param1, param2 = param_names[i], param_names[j]
            interaction = self._calculate_interaction_effect(
                scaled_samples[:, i], scaled_samples[:, j], entanglement_values
            )
            interaction_effects[f"{param1}_x_{param2}"] = float(interaction)
    
    # Component sensitivities
    component_sensitivities = {}
    for comp_name, comp_values in component_values.items():
        comp_values = np.array(comp_values)
        comp_sensitivities = {}
        for i, param_name in enumerate(param_names):
            sensitivity = self._calculate_sobol_index(scaled_samples[:, i], comp_values)
            comp_sensitivities[param_name] = float(sensitivity)
        component_sensitivities[comp_name] = comp_sensitivities
    
    # Correlation matrix
    all_data = np.column_stack([scaled_samples, entanglement_values])
    correlation_matrix = np.corrcoef(all_data.T)
    
    # Total variance explained
    total_variance_explained = sum(parameter_impacts.values())
    
    analysis_time = time.time() - start_time
    self.log(f"Parameter sensitivity analysis completed in {analysis_time:.2f} seconds")
    
    return SensitivityResults(
        parameter_impacts=parameter_impacts,
        interaction_effects=interaction_effects,
        component_sensitivities=component_sensitivities,
        correlation_matrix=correlation_matrix,
        parameter_names=param_names + ['entanglement'],
        total_variance_explained=total_variance_explained
    )

def _latin_hypercube_sampling(self, n_samples: int, n_dimensions: int) -> np.ndarray:
    """Generate Latin Hypercube samples"""
    samples = np.zeros((n_samples, n_dimensions))
    
    for i in range(n_dimensions):
        samples[:, i] = (np.random.permutation(n_samples) + np.random.random(n_samples)) / n_samples
        
    return samples

def _create_temp_config_from_sample(self, sample: np.ndarray, param_names: List[str]) -> Dict[str, Any]:
    """Create temporary configuration from parameter sample"""
    base_config = self.config.copy()
    
    # Normalize weights to sum to 1
    c_sdn_weight = sample[param_names.index('C_SDN_weight')]
    vei_weight = sample[param_names.index('VEI_delta_weight')]
    qf_weight = sample[param_names.index('QF_delta_weight')]
    total_weight = c_sdn_weight + vei_weight + qf_weight
    
    base_config['parameters'] = base_config.get('parameters', {}).copy()
    base_config['parameters'].update({
        'lambda_weights': {
            'C_SDN': c_sdn_weight / total_weight,
            'VEI_delta': vei_weight / total_weight,
            'QF_delta': qf_weight / total_weight
        },
        'use_sdkp_time': True,
        'sdkp_size': sample[param_names.index('sdkp_size')],
        'sdkp_density': sample[param_names.index('sdkp_density')],
        'sdkp_rotation_velocity': sample[param_names.index('sdkp_rotation_velocity')],
        'use_qcc_entropy': True,
        'qcc_entropy': sample[param_names.index('qcc_entropy')],
        'logging': False  # Disable logging for performance
    })
    
    return base_config

def _calculate_sobol_index(self, param_values: np.ndarray, output_values: np.ndarray) -> float:
    """Calculate first-order Sobol sensitivity index"""
    try:
        # Sort by parameter values
        sorted_indices = np.argsort(param_values)
        sorted_output = output_values[sorted_indices]
        
        # Calculate conditional variances
        n_bins = min(10, len(param_values) // 10)
        if n_bins < 2:
            return 0.0
            
        bin_size = len(param_values) // n_bins
        bin_vars = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(param_values)
            bin_data = sorted_output[start_idx:end_idx]
            if len(bin_data) > 1:
                bin_vars.append(np.var(bin_data))
        
        if not bin_vars:
            return 0.0
            
        # Sobol index approximation
        total_var = np.var(output_values)
        if total_var == 0:
            return 0.0
            
        conditional_var = np.mean(bin_vars)
        sobol_index = 1 - (conditional_var / total_var)
        
        return max(0.0, min(1.0, sobol_index))  # Clamp to [0,1]
        
    except Exception:
        return 0.0

def _calculate_interaction_effect(self, param1: np.ndarray, param2: np.ndarray, 
                                output: np.ndarray) -> float:
    """Calculate second-order interaction effect"""
    try:
        # Create 2D grid for interaction analysis
        n_bins = 5
        
        # Discretize parameters
        p1_bins = np.digitize(param1, np.linspace(param1.min(), param1.max(), n_bins))
        p2_bins = np.digitize(param2, np.linspace(param2.min(), param2.max(), n_bins))
        
        # Calculate interaction variance
        interaction_effects = []
        for i in range(1, n_bins + 1):
            for j in range(1, n_bins + 1):
                mask = (p1_bins == i) & (p2_bins == j)
                if np.sum(mask) > 1:
                    bin_output = output[mask]
                    interaction_effects.append(np.var(bin_output))
        
        if not interaction_effects:
            return 0.0
            
        total_var = np.var(output)
        if total_var == 0:
            return 0.0
            
        interaction_var = np.mean(interaction_effects)
        return max(0.0, interaction_var / total_var)
        
    except Exception:
        return 0.0

def run_parallel_simulation(self, use_multiprocessing: bool = True, 
                          n_workers: int = None) -> OptimizedSimulationResults:
    """
    Run simulation with parallel processing for performance optimization
    """
    self.log("Starting optimized parallel simulation...")
    start_time = time.time()
    
    angles = self.config['parameters']['polarization_angles_deg']
    signatures = self.config['parameters']['numerical_signatures']
    
    # Pre-compute signature values
    signature_values = self.calculate_numerical_signature_value_vectorized(signatures)
    
    # Create parameter combinations
    param_combinations = list(product(angles, signatures))
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(param_combinations))
    
    self.log(f"Processing {len(param_combinations)} combinations using {n_workers} workers")
    
    # Choose executor based on use_multiprocessing flag
    executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
    
    all_entanglement_values = []
    all_correlation_coefficients = []
    numerical_mappings = {}
    polarization_data = []
    
    # Process in batches for memory efficiency
    batch_size = max(1, len(param_combinations) // n_workers)
    
    with executor_class(max_workers=n_workers) as executor:
        # Submit angle-based batches
        angle_futures = {}
        
        for angle in angles:
            angle_combinations = [(a, s) for a, s in param_combinations if a == angle]
            future = executor.submit(self._process_angle_batch, angle_combinations)
            angle_futures[angle] = future
        
        # Collect results
        for angle, future in angle_futures.items():
            try:
                angle_results = future.result(timeout=300)  # 5 minute timeout
                
                angle_entanglements = angle_results['entanglements']
                angle_mappings = angle_results['mappings']
                
                all_entanglement_values.extend(angle_entanglements)
                numerical_mappings.update(angle_mappings)
                
                # Calculate correlation for this angle
                correlation = self._calculate_correlation_coefficient_fast(angle_entanglements)
                all_correlation_coefficients.append(correlation)
                
                # Create polarization data entry
                threshold = self.config['parameters'].get('entanglement_threshold', 0.75)
                angle_stats = {
                    'angle_deg': angle,
                    'entanglement_values': [float(x) for x in angle_entanglements],
                    'mean_entanglement': float(np.mean(angle_entanglements)),
                    'std_entanglement': float(np.std(angle_entanglements)),
                    'min_entanglement': float(np.min(angle_entanglements)),
                    'max_entanglement': float(np.max(angle_entanglements)),
                    'correlation': float(correlation),
                    'high_entanglement_count': sum(1 for e in angle_entanglements if e > threshold)
                }
                polarization_data.append(angle_stats)
                
            except Exception as e:
                self.log(f"Error processing angle {angle}: {e}", 'error')
    
    # Calculate lambda contributions
    lambda_contributions = self._calculate_lambda_contributions_fast(numerical_mappings)
    
    # Run sensitivity analysis if requested
    sensitivity_analysis = None
    if self.config['parameters'].get('run_sensitivity_analysis', False):
        sample_size = self.config['parameters'].get('sensitivity_sample_size', 500)
        sensitivity_analysis = self.parameter_sensitivity_analysis(sample_size)
    
    # Performance metrics
    total_time = time.time() - start_time
    performance_metrics = {
        'total_execution_time_seconds': total
```
*Automatically synced with your [v0.dev](https://v0.dev) deployments*

[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://vercel.com/donald-paul-smiths-projects/v0-image-analysis)
[![Built with v0](https://img.shields.io/badge/Built%20with-v0.dev-black?style=for-the-badge)](https://v0.dev/chat/projects/hMqnWk80fYS)

## Overview

This repository will stay in sync with your deployed chats on [v0.dev](https://v0.dev).
Any changes you make to your deployed app will be automatically pushed to this repository from [v0.dev](https://v0.dev).

## Deployment

Your project is live at:

**[https://vercel.com/donald-paul-smiths-projects/v0-image-analysis](https://vercel.com/donald-paul-smiths-projects/v0-image-analysis)**

## Build your app

Continue building your app on:

**[https://v0.dev/chat/projects/hMqnWk80fYS](https://v0.dev/chat/projects/hMqnWk80fYS)**

## How It Works

1. Create and modify your project using [v0.dev](https://v0.dev)
2. Deploy your chats from the v0 interface
3. Changes are automatically pushed to this repository
4. Vercel deploys the latest version from this repository
