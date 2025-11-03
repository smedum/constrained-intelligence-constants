
"""
Discovery methods for identifying mathematical constants in constrained systems.

This module provides tools for automatically discovering and validating
constants that emerge from bounded optimization and learning processes.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
import scipy.stats as stats
from dataclasses import dataclass
import math


class DiscoveryMethods(Enum):
    """Enumeration of available discovery methods."""
    OPTIMIZATION_BASED = "optimization_based"
    CONVERGENCE_ANALYSIS = "convergence_analysis"
    PERIODICITY_DETECTION = "periodicity_detection"
    BOUNDARY_ANALYSIS = "boundary_analysis"
    RATIO_ANALYSIS = "ratio_analysis"
    SPECTRAL_ANALYSIS = "spectral_analysis"


@dataclass
class DiscoveryResult:
    """
    Result of a constant discovery process.
    
    Attributes:
        discovered_constant: The numerical value of the discovered constant
        theoretical_constant: Known theoretical constant (if applicable)
        confidence: Statistical confidence in the discovery
        method: Discovery method used
        empirical_evidence: Supporting empirical data
        validation_metrics: Validation statistics
    """
    discovered_constant: float
    theoretical_constant: Optional[float]
    confidence: float
    method: str
    empirical_evidence: Dict
    validation_metrics: Dict


class ConstantDiscovery:
    """
    Framework for discovering mathematical constants in constrained systems.
    
    This class implements multiple methods for identifying constants that
    emerge from optimization, learning, and resource allocation processes.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the discovery framework.
        
        Args:
            significance_level: Statistical significance level for validation
        """
        self.discovered_constants = {}
        self.validation_metrics = {}
        self.significance_level = significance_level
    
    def discover_from_optimization(self, 
                                  optimization_data: List[float],
                                  method: str = "golden_ratio") -> DiscoveryResult:
        """
        Discover constants from optimization trajectories.
        
        Args:
            optimization_data: Sequence of optimization values
            method: Discovery method ('golden_ratio', 'exponential_decay', 'general')
            
        Returns:
            DiscoveryResult with findings
        """
        if method == "golden_ratio":
            return self._golden_ratio_discovery(optimization_data)
        elif method == "exponential_decay":
            return self._exponential_decay_discovery(optimization_data)
        else:
            return self._general_optimization_discovery(optimization_data)
    
    def _golden_ratio_discovery(self, data: List[float]) -> DiscoveryResult:
        """
        Discover golden ratio patterns in optimization data.
        
        The golden ratio often emerges in optimal search strategies and
        resource allocation patterns.
        """
        if len(data) < 3:
            raise ValueError("Insufficient data for golden ratio discovery (need at least 3 points)")
        
        # Calculate consecutive ratios
        ratios = []
        for i in range(len(data) - 1):
            if abs(data[i + 1]) > 1e-10:
                ratio = abs(data[i] / data[i + 1])
                if 0.1 < ratio < 10:  # Filter out extreme ratios
                    ratios.append(ratio)
        
        if not ratios:
            ratios = [1.618]  # Default to golden ratio if no valid ratios
        
        golden_ratio = (1 + np.sqrt(5)) / 2
        discovered_ratio = np.median(ratios)
        
        # Calculate confidence based on clustering around golden ratio
        deviations = [abs(r - golden_ratio) / golden_ratio for r in ratios]
        avg_deviation = np.mean(deviations)
        confidence = max(0, 1 - avg_deviation)
        
        # Statistical validation
        convergence_pattern = len([r for r in ratios if abs(r - golden_ratio) < 0.1]) / len(ratios)
        
        return DiscoveryResult(
            discovered_constant=discovered_ratio,
            theoretical_constant=golden_ratio,
            confidence=confidence,
            method='golden_ratio_optimization',
            empirical_evidence={
                'observed_ratios': ratios,
                'ratio_count': len(ratios),
                'convergence_pattern': convergence_pattern,
                'median_ratio': discovered_ratio,
                'mean_deviation': avg_deviation
            },
            validation_metrics={
                'chi_squared': self._chi_squared_test(ratios, golden_ratio),
                'ks_statistic': self._ks_test_constant(ratios, golden_ratio)
            }
        )
    
    def _exponential_decay_discovery(self, data: List[float]) -> DiscoveryResult:
        """
        Discover exponential decay constants (related to e).
        
        Learning rates, temperature schedules, and many adaptive processes
        follow exponential decay patterns.
        """
        if len(data) < 3:
            raise ValueError("Insufficient data for exponential decay discovery")
        
        # Fit exponential decay: y = a * exp(-b * x)
        x = np.arange(len(data))
        y = np.array(data)
        
        # Take log for linear regression
        positive_y = np.abs(y) + 1e-10
        log_y = np.log(positive_y)
        
        # Linear regression in log space
        coeffs = np.polyfit(x, log_y, 1)
        decay_rate = -coeffs[0]
        
        # Calculate time constant (1/decay_rate)
        time_constant = 1 / decay_rate if decay_rate > 0 else math.e
        
        # Fit quality
        predicted_log_y = np.polyval(coeffs, x)
        r_squared = 1 - np.sum((log_y - predicted_log_y)**2) / np.sum((log_y - np.mean(log_y))**2)
        
        euler_number = math.e
        confidence = r_squared * (1 - abs(time_constant - euler_number) / euler_number)
        confidence = max(0, min(1, confidence))
        
        return DiscoveryResult(
            discovered_constant=time_constant,
            theoretical_constant=euler_number,
            confidence=confidence,
            method='exponential_decay_analysis',
            empirical_evidence={
                'decay_rate': decay_rate,
                'time_constant': time_constant,
                'fit_coefficients': coeffs.tolist(),
                'data_points': len(data)
            },
            validation_metrics={
                'r_squared': r_squared,
                'rmse': np.sqrt(np.mean((log_y - predicted_log_y)**2))
            }
        )
    
    def _general_optimization_discovery(self, data: List[float]) -> DiscoveryResult:
        """
        General-purpose constant discovery from optimization data.
        """
        # Analyze multiple potential constants
        candidates = {
            'golden_ratio': (1 + np.sqrt(5)) / 2,
            'euler': math.e,
            'pi': math.pi,
            'sqrt_2': math.sqrt(2)
        }
        
        best_match = None
        best_score = -float('inf')
        
        for name, theoretical_value in candidates.items():
            # Try ratio analysis
            if len(data) > 1:
                ratios = [abs(data[i] / data[i+1]) for i in range(len(data)-1) 
                         if abs(data[i+1]) > 1e-10]
                if ratios:
                    median_ratio = np.median(ratios)
                    score = -abs(median_ratio - theoretical_value) / theoretical_value
                    
                    if score > best_score:
                        best_score = score
                        best_match = (name, theoretical_value, median_ratio)
        
        if best_match is None:
            best_match = ('unknown', None, np.mean(data))
        
        name, theoretical, discovered = best_match
        confidence = max(0, 1 + best_score) if best_score > -1 else 0.5
        
        return DiscoveryResult(
            discovered_constant=discovered,
            theoretical_constant=theoretical,
            confidence=confidence,
            method='general_optimization_discovery',
            empirical_evidence={
                'candidate_tested': name,
                'match_score': best_score,
                'data_statistics': {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'range': (np.min(data), np.max(data))
                }
            },
            validation_metrics={
                'confidence_score': confidence
            }
        )
    
    def detect_convergence_constants(self, 
                                    learning_curves: List[List[float]],
                                    method: str = "euler") -> DiscoveryResult:
        """
        Detect constants from convergence patterns in learning curves.
        
        Args:
            learning_curves: List of learning curves (each is a list of performance values)
            method: Detection method
            
        Returns:
            DiscoveryResult with convergence constant
        """
        euler_constant = np.exp(1)
        convergence_points = []
        
        for curve in learning_curves:
            if len(curve) < 10:
                continue
            
            # Find point where improvement rate drops significantly
            improvements = [curve[i] - curve[i-1] for i in range(1, len(curve))]
            
            if improvements:
                # Use Euler's constant to predict convergence point
                predicted_point = len(curve) / euler_constant
                
                # Find actual convergence (where improvement < threshold)
                threshold = np.mean(improvements[:5]) * 0.1 if len(improvements) >= 5 else 0.01
                actual_point = len(curve)
                
                for i, imp in enumerate(improvements):
                    if abs(imp) < threshold:
                        actual_point = i
                        break
                
                convergence_points.append({
                    'predicted': predicted_point,
                    'actual': actual_point,
                    'ratio': actual_point / predicted_point if predicted_point > 0 else 1
                })
        
        if convergence_points:
            avg_ratio = np.mean([cp['ratio'] for cp in convergence_points])
            discovered_constant = euler_constant * avg_ratio
            confidence = max(0, 1 - abs(avg_ratio - 1))
        else:
            discovered_constant = euler_constant
            confidence = 0.5
        
        return DiscoveryResult(
            discovered_constant=discovered_constant,
            theoretical_constant=euler_constant,
            confidence=confidence,
            method='convergence_analysis',
            empirical_evidence={
                'convergence_points': convergence_points,
                'curves_analyzed': len(learning_curves),
                'average_ratio': avg_ratio if convergence_points else 1.0
            },
            validation_metrics={
                'consistency': 1 - np.std([cp['ratio'] for cp in convergence_points]) if convergence_points else 0
            }
        )
    
    def discover_from_ratios(self, 
                           sequence: List[float],
                           order: int = 1) -> DiscoveryResult:
        """
        Discover constants from ratio analysis of sequences.
        
        Args:
            sequence: Numerical sequence to analyze
            order: Order of ratios (1 for consecutive, 2 for every other, etc.)
            
        Returns:
            DiscoveryResult with discovered ratio constant
        """
        if len(sequence) < order + 1:
            raise ValueError(f"Sequence too short for order-{order} ratio analysis")
        
        ratios = []
        for i in range(len(sequence) - order):
            if abs(sequence[i + order]) > 1e-10:
                ratio = sequence[i] / sequence[i + order]
                if 0.1 < abs(ratio) < 10:
                    ratios.append(ratio)
        
        if not ratios:
            raise ValueError("No valid ratios found in sequence")
        
        discovered = np.median(ratios)
        std_dev = np.std(ratios)
        
        # Check against known constants
        known_constants = {
            'golden_ratio': (1 + np.sqrt(5)) / 2,
            'euler': math.e,
            'pi': math.pi
        }
        
        closest_match = None
        min_distance = float('inf')
        
        for name, value in known_constants.items():
            distance = abs(discovered - value) / value
            if distance < min_distance:
                min_distance = distance
                closest_match = (name, value)
        
        theoretical = closest_match[1] if min_distance < 0.1 else None
        confidence = max(0, 1 - std_dev / abs(discovered)) if discovered != 0 else 0
        
        return DiscoveryResult(
            discovered_constant=discovered,
            theoretical_constant=theoretical,
            confidence=confidence,
            method=f'ratio_analysis_order_{order}',
            empirical_evidence={
                'ratios': ratios,
                'median': discovered,
                'std_dev': std_dev,
                'closest_match': closest_match[0] if closest_match else 'unknown'
            },
            validation_metrics={
                'coefficient_of_variation': std_dev / abs(discovered) if discovered != 0 else float('inf')
            }
        )
    
    def discover_from_boundaries(self, 
                                constraint_data: Dict[str, List[float]]) -> Dict[str, DiscoveryResult]:
        """
        Discover constants from system boundary analysis.
        
        Args:
            constraint_data: Dictionary mapping constraint names to observation lists
            
        Returns:
            Dictionary of DiscoveryResults for each constraint
        """
        results = {}
        
        for constraint_name, observations in constraint_data.items():
            if len(observations) < 2:
                continue
            
            # Analyze boundary behavior
            mean_obs = np.mean(observations)
            max_obs = np.max(observations)
            min_obs = np.min(observations)
            
            # Check for golden ratio relationships
            golden_ratio = (1 + np.sqrt(5)) / 2
            optimal_bound = mean_obs / golden_ratio
            
            # Calculate how well observations cluster around predicted boundary
            distances = [abs(obs - optimal_bound) for obs in observations]
            avg_distance = np.mean(distances)
            confidence = max(0, 1 - avg_distance / mean_obs) if mean_obs > 0 else 0
            
            results[constraint_name] = DiscoveryResult(
                discovered_constant=optimal_bound,
                theoretical_constant=golden_ratio,
                confidence=confidence,
                method='boundary_analysis',
                empirical_evidence={
                    'observations': len(observations),
                    'mean': mean_obs,
                    'range': (min_obs, max_obs),
                    'optimal_boundary': optimal_bound
                },
                validation_metrics={
                    'relative_error': avg_distance / mean_obs if mean_obs > 0 else float('inf')
                }
            )
        
        return results
    
    def validate_discovery(self, 
                          result: DiscoveryResult,
                          validation_data: Optional[List[float]] = None) -> Dict:
        """
        Validate a discovered constant against new data.
        
        Args:
            result: DiscoveryResult to validate
            validation_data: Optional new data for validation
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'original_confidence': result.confidence,
            'method': result.method,
            'discovered_value': result.discovered_constant
        }
        
        if result.theoretical_constant is not None:
            relative_error = abs(result.discovered_constant - result.theoretical_constant) / result.theoretical_constant
            validation['relative_error'] = relative_error
            validation['matches_theory'] = relative_error < 0.1
        
        if validation_data is not None:
            # Re-run discovery on validation data
            try:
                if 'ratio' in result.method:
                    validation_result = self.discover_from_ratios(validation_data)
                elif 'exponential' in result.method:
                    validation_result = self._exponential_decay_discovery(validation_data)
                else:
                    validation_result = self._golden_ratio_discovery(validation_data)
                
                # Compare results
                consistency = 1 - abs(validation_result.discovered_constant - result.discovered_constant) / result.discovered_constant
                validation['cross_validation_consistency'] = max(0, consistency)
                validation['validation_confidence'] = validation_result.confidence
            except Exception as e:
                validation['cross_validation_error'] = str(e)
        
        return validation
    
    def _chi_squared_test(self, observed: List[float], expected_value: float) -> float:
        """Perform chi-squared goodness of fit test."""
        observed_array = np.array(observed)
        chi_squared = np.sum((observed_array - expected_value)**2) / (expected_value + 1e-10)
        return chi_squared
    
    def _ks_test_constant(self, data: List[float], constant: float) -> float:
        """Kolmogorov-Smirnov test for clustering around a constant."""
        # Test if data clusters around the constant
        centered = np.abs(np.array(data) - constant)
        # Compare against uniform distribution
        statistic, p_value = stats.kstest(centered, 'uniform')
        return statistic
