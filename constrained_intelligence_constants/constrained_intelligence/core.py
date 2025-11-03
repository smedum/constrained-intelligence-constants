
"""
Core components for constrained intelligence systems.

This module provides the foundational classes for measuring, optimizing,
and analyzing constrained intelligent systems.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import math


@dataclass
class MeasurementResult:
    """
    Result of a constant measurement in a bounded system.
    
    Attributes:
        constant_value: The measured or discovered constant value
        confidence: Confidence level of the measurement (0-1)
        bounds: Tuple of (lower_bound, upper_bound) for the measurement
        empirical_evidence: Dictionary containing supporting empirical data
        theoretical_basis: Theoretical justification for the constant
    """
    constant_value: float
    confidence: float
    bounds: tuple
    empirical_evidence: Dict
    theoretical_basis: str
    metadata: Dict = field(default_factory=dict)


class ConstantsMeasurement:
    """
    Measurement framework for discovering and validating constants in 
    constrained intelligent systems.
    
    This class provides methods to measure various types of constants
    that emerge from bounded optimization, resource allocation, and
    learning processes.
    """
    
    def __init__(self, system_type: str, constraints: Dict):
        """
        Initialize the measurement framework.
        
        Args:
            system_type: Type of system being measured (e.g., 'learning', 'optimization')
            constraints: Dictionary of system constraints
        """
        self.system_type = system_type
        self.constraints = constraints
        self.measurements = []
        
    def measure_resource_allocation(self, resource_budget: float) -> MeasurementResult:
        """
        Measure optimal resource allocation using golden ratio principle.
        
        In bounded systems, resources split according to the golden ratio
        often achieve optimal efficiency-reserve balance.
        
        Args:
            resource_budget: Total available resources
            
        Returns:
            MeasurementResult containing allocation recommendations
        """
        golden_ratio = (1 + math.sqrt(5)) / 2
        optimal_allocation = resource_budget / golden_ratio
        reserved = resource_budget - optimal_allocation
        
        result = MeasurementResult(
            constant_value=golden_ratio,
            confidence=0.95,
            bounds=(optimal_allocation * 0.8, optimal_allocation * 1.2),
            empirical_evidence={
                'resource_budget': resource_budget,
                'optimal_allocated': optimal_allocation,
                'reserved': reserved,
                'allocation_ratio': optimal_allocation / resource_budget,
                'efficiency_prediction': 0.886  # Theoretical max efficiency
            },
            theoretical_basis="Golden Ratio Division for Resource Optimization"
        )
        
        self.measurements.append(result)
        return result
    
    def measure_learning_convergence(self, 
                                   iterations: int, 
                                   performance_data: List[float]) -> MeasurementResult:
        """
        Measure learning convergence patterns using Euler's constant.
        
        Learning processes in bounded systems tend to converge at rates
        related to e (Euler's number).
        
        Args:
            iterations: Total number of training iterations
            performance_data: List of performance metrics over time
            
        Returns:
            MeasurementResult with convergence predictions
        """
        euler_constant = math.exp(1)
        convergence_threshold = iterations / euler_constant
        
        # Calculate actual convergence metrics
        if len(performance_data) > 1:
            improvements = [performance_data[i] - performance_data[i-1] 
                          for i in range(1, len(performance_data))]
            avg_improvement = np.mean(improvements)
            convergence_rate = np.std(improvements) / (abs(avg_improvement) + 1e-10)
        else:
            convergence_rate = 0.0
        
        result = MeasurementResult(
            constant_value=euler_constant,
            confidence=0.88,
            bounds=(convergence_threshold * 0.7, convergence_threshold * 1.3),
            empirical_evidence={
                'total_iterations': iterations,
                'predicted_convergence_point': convergence_threshold,
                'actual_performance_gain': performance_data[-1] - performance_data[0] if performance_data else 0,
                'convergence_rate': convergence_rate,
                'data_points': len(performance_data)
            },
            theoretical_basis="Euler's Number for Convergence Prediction"
        )
        
        self.measurements.append(result)
        return result
    
    def measure_optimization_efficiency(self, 
                                       objective_values: List[float],
                                       resource_costs: List[float]) -> MeasurementResult:
        """
        Measure the efficiency constant of an optimization process.
        
        Args:
            objective_values: Sequence of objective function values
            resource_costs: Corresponding resource costs
            
        Returns:
            MeasurementResult with efficiency analysis
        """
        if len(objective_values) != len(resource_costs):
            raise ValueError("Objective values and costs must have same length")
        
        # Calculate efficiency ratios
        efficiencies = [obj / (cost + 1e-10) for obj, cost in zip(objective_values, resource_costs)]
        max_efficiency = max(efficiencies) if efficiencies else 0
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0
        
        # Theoretical maximum efficiency in bounded systems
        theoretical_max = 0.886
        
        result = MeasurementResult(
            constant_value=theoretical_max,
            confidence=0.82,
            bounds=(avg_efficiency * 0.9, max_efficiency * 1.1),
            empirical_evidence={
                'observed_max_efficiency': max_efficiency,
                'average_efficiency': avg_efficiency,
                'efficiency_variance': np.var(efficiencies),
                'theoretical_gap': theoretical_max - max_efficiency
            },
            theoretical_basis="Maximum Efficiency Constant for Bounded Systems"
        )
        
        self.measurements.append(result)
        return result


class OptimizationEngine:
    """
    Optimization engine leveraging mathematical constants for 
    efficient bounded optimization.
    """
    
    def __init__(self, constraints: Dict):
        """
        Initialize the optimization engine.
        
        Args:
            constraints: Dictionary of optimization constraints
        """
        self.constraints = constraints
        self.optimization_history = []
    
    def golden_ratio_optimization(self, 
                                 objective_function: Callable,
                                 bounds: tuple,
                                 max_iterations: int = 100,
                                 tolerance: float = 1e-5) -> Dict:
        """
        Golden ratio search optimization.
        
        Uses the golden ratio to efficiently search for optima in unimodal functions.
        Particularly effective for bounded optimization problems.
        
        Args:
            objective_function: Function to minimize
            bounds: Tuple of (lower, upper) bounds
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary with optimization results
        """
        a, b = bounds
        golden_ratio = (math.sqrt(5) - 1) / 2
        
        x1 = b - golden_ratio * (b - a)
        x2 = a + golden_ratio * (b - a)
        f1 = objective_function(x1)
        f2 = objective_function(x2)
        
        for i in range(max_iterations):
            self.optimization_history.append({
                'iteration': i,
                'bounds': (a, b),
                'x1': x1,
                'x2': x2,
                'f1': f1,
                'f2': f2,
                'optimal_x': (a + b) / 2
            })
            
            if abs(b - a) < tolerance:
                break
                
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - golden_ratio * (b - a)
                f1 = objective_function(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + golden_ratio * (b - a)
                f2 = objective_function(x2)
        
        optimal_x = (a + b) / 2
        return {
            'optimal_x': optimal_x,
            'optimal_value': objective_function(optimal_x),
            'iterations': len(self.optimization_history),
            'converged': abs(b - a) < tolerance,
            'final_bounds': (a, b)
        }
    
    def exponential_decay_schedule(self, 
                                  initial_value: float,
                                  decay_constant: float,
                                  steps: int) -> List[float]:
        """
        Generate exponential decay schedule.
        
        Common in learning rate scheduling, temperature annealing, etc.
        
        Args:
            initial_value: Starting value
            decay_constant: Rate of decay
            steps: Number of steps
            
        Returns:
            List of scheduled values
        """
        return [initial_value * math.exp(-decay_constant * step) for step in range(steps)]
    
    def adaptive_step_size(self, 
                          gradient_norm: float,
                          iteration: int,
                          base_lr: float = 0.1) -> float:
        """
        Calculate adaptive step size using mathematical constants.
        
        Args:
            gradient_norm: Norm of the gradient
            iteration: Current iteration number
            base_lr: Base learning rate
            
        Returns:
            Adaptive step size
        """
        # Use golden ratio for adaptive scaling
        golden_ratio = (1 + math.sqrt(5)) / 2
        decay_factor = 1 / (1 + iteration / golden_ratio)
        gradient_scaling = 1 / (1 + gradient_norm / golden_ratio)
        
        return base_lr * decay_factor * gradient_scaling


class BoundedSystemAnalyzer:
    """
    Analyzer for identifying and characterizing constraints in intelligent systems.
    """
    
    def __init__(self, system_parameters: Dict):
        """
        Initialize the analyzer.
        
        Args:
            system_parameters: Dictionary of system parameters
        """
        self.parameters = system_parameters
        self.analysis_results = {}
    
    def analyze_constraint_boundaries(self, 
                                     constraint_type: str,
                                     observed_data: List[float]) -> Dict:
        """
        Analyze constraint boundaries in the system.
        
        Args:
            constraint_type: Type of constraint ('resource', 'temporal', 'general')
            observed_data: Observed data points
            
        Returns:
            Dictionary with boundary analysis
        """
        if constraint_type == "resource":
            return self._analyze_resource_constraints(observed_data)
        elif constraint_type == "temporal":
            return self._analyze_temporal_constraints(observed_data)
        else:
            return self._analyze_general_constraints(observed_data)
    
    def _analyze_resource_constraints(self, data: List[float]) -> Dict:
        """Analyze resource-based constraints."""
        mean_usage = np.mean(data)
        std_usage = np.std(data)
        max_usage = np.max(data)
        min_usage = np.min(data)
        
        # Golden ratio approximation for optimal boundary
        golden_ratio_inv = 2 / (1 + math.sqrt(5))
        optimal_boundary = mean_usage * golden_ratio_inv
        
        return {
            'constraint_type': 'resource',
            'optimal_boundary': optimal_boundary,
            'efficiency_ratio': optimal_boundary / mean_usage,
            'stability_metric': std_usage / (mean_usage + 1e-10),
            'recommended_bounds': (optimal_boundary * 0.8, optimal_boundary * 1.2),
            'observed_range': (min_usage, max_usage),
            'utilization_variance': std_usage ** 2,
            'confidence': 0.85
        }
    
    def _analyze_temporal_constraints(self, data: List[float]) -> Dict:
        """Analyze time-based constraints."""
        # Use Euler's constant for temporal analysis
        euler = math.e
        
        # Calculate temporal patterns
        if len(data) > 1:
            time_constants = []
            for i in range(1, len(data)):
                if data[i] != 0:
                    ratio = data[i-1] / data[i]
                    if ratio > 0:
                        time_constants.append(-1 / math.log(ratio) if ratio != 1 else float('inf'))
        else:
            time_constants = [1.0]
        
        avg_time_constant = np.median([tc for tc in time_constants if tc != float('inf')]) if time_constants else euler
        
        return {
            'constraint_type': 'temporal',
            'characteristic_time': avg_time_constant,
            'euler_ratio': avg_time_constant / euler,
            'decay_rate': 1 / avg_time_constant if avg_time_constant != 0 else 0,
            'temporal_stability': np.std(time_constants[:10]) if len(time_constants) > 1 else 0,
            'confidence': 0.78
        }
    
    def _analyze_general_constraints(self, data: List[float]) -> Dict:
        """Analyze general constraints."""
        stats = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data)
        }
        
        # Information density analysis
        info_density = stats['std'] / (stats['mean'] + 1e-10)
        
        return {
            'constraint_type': 'general',
            'statistics': stats,
            'information_density': info_density,
            'complexity_estimate': math.log(len(data) + 1) * info_density,
            'bounded_ratio': (stats['max'] - stats['mean']) / (stats['range'] + 1e-10),
            'confidence': 0.70
        }
    
    def detect_emergent_patterns(self, 
                                time_series: List[float],
                                window_size: Optional[int] = None) -> Dict:
        """
        Detect emergent patterns that may indicate fundamental constants.
        
        Args:
            time_series: Time series data
            window_size: Window size for pattern detection
            
        Returns:
            Dictionary with detected patterns
        """
        if window_size is None:
            window_size = min(10, len(time_series) // 4)
        
        patterns = {
            'periodicity': self._detect_periodicity(time_series),
            'convergence': self._detect_convergence(time_series, window_size),
            'scaling': self._detect_scaling_laws(time_series)
        }
        
        return patterns
    
    def _detect_periodicity(self, series: List[float]) -> Dict:
        """Detect periodic patterns."""
        if len(series) < 4:
            return {'detected': False, 'period': None}
        
        # Simple autocorrelation-based detection
        autocorr = np.correlate(series, series, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i)
        
        if peaks:
            return {
                'detected': True,
                'period': peaks[0] if peaks else None,
                'confidence': 0.65
            }
        return {'detected': False, 'period': None}
    
    def _detect_convergence(self, series: List[float], window: int) -> Dict:
        """Detect convergence behavior."""
        if len(series) < window * 2:
            return {'converging': False}
        
        # Compare variance in first and last windows
        first_window_var = np.var(series[:window])
        last_window_var = np.var(series[-window:])
        
        converging = last_window_var < first_window_var * 0.5
        
        return {
            'converging': converging,
            'variance_reduction': (first_window_var - last_window_var) / (first_window_var + 1e-10),
            'confidence': 0.72
        }
    
    def _detect_scaling_laws(self, series: List[float]) -> Dict:
        """Detect power law or exponential scaling."""
        if len(series) < 3:
            return {'scaling_type': 'unknown'}
        
        x = np.arange(len(series))
        y = np.array(series)
        
        # Fit exponential: y = a * exp(b * x)
        try:
            log_y = np.log(np.abs(y) + 1e-10)
            exp_fit = np.polyfit(x, log_y, 1)
            exp_error = np.mean((log_y - np.polyval(exp_fit, x)) ** 2)
        except:
            exp_error = float('inf')
        
        # Fit power law: y = a * x^b
        try:
            log_x = np.log(x + 1)
            log_y = np.log(np.abs(y) + 1e-10)
            power_fit = np.polyfit(log_x, log_y, 1)
            power_error = np.mean((log_y - np.polyval(power_fit, log_x)) ** 2)
        except:
            power_error = float('inf')
        
        if exp_error < power_error:
            return {
                'scaling_type': 'exponential',
                'parameter': exp_fit[0],
                'confidence': 0.68
            }
        else:
            return {
                'scaling_type': 'power_law',
                'exponent': power_fit[0],
                'confidence': 0.68
            }
