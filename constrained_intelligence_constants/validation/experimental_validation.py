"""
Experimental validation suite for Constrained Intelligence Constants.

This module validates the framework against known mathematical results
and empirical benchmarks.
"""

import sys
import os
import numpy as np
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constrained_intelligence import (
    ConstantsMeasurement,
    OptimizationEngine,
    BoundedSystemAnalyzer,
    ConstantDiscovery,
    GOLDEN_RATIO,
    EULER_NUMBER,
    OPTIMAL_RESOURCE_SPLIT,
)


class ValidationResult:
    """Container for validation results."""
    def __init__(self, name: str, passed: bool, details: Dict):
        self.name = name
        self.passed = passed
        self.details = details
    
    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.name}"


def validate_golden_ratio_discovery() -> ValidationResult:
    """Validate golden ratio discovery from synthetic data."""
    print("\n" + "-"*70)
    print("Validating: Golden Ratio Discovery")
    print("-"*70)
    
    # Generate perfect golden ratio sequence
    data = [100]
    for _ in range(10):
        data.append(data[-1] / GOLDEN_RATIO)
    
    discovery = ConstantDiscovery()
    result = discovery.discover_from_optimization(data, method="golden_ratio")
    
    # Check if discovered constant is close to golden ratio
    error = abs(result.discovered_constant - GOLDEN_RATIO) / GOLDEN_RATIO
    passed = error < 0.01 and result.confidence > 0.9
    
    details = {
        'discovered': result.discovered_constant,
        'expected': GOLDEN_RATIO,
        'relative_error': error,
        'confidence': result.confidence,
        'threshold': 0.01
    }
    
    print(f"  Discovered: {result.discovered_constant:.6f}")
    print(f"  Expected: {GOLDEN_RATIO:.6f}")
    print(f"  Relative Error: {error:.4%}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return ValidationResult("Golden Ratio Discovery", passed, details)


def validate_exponential_decay() -> ValidationResult:
    """Validate exponential decay constant discovery."""
    print("\n" + "-"*70)
    print("Validating: Exponential Decay Discovery")
    print("-"*70)
    
    # Generate exponential decay sequence
    t = np.arange(20)
    data = 10 * np.exp(-t / 5)  # Decay with time constant 5
    
    discovery = ConstantDiscovery()
    result = discovery.discover_from_optimization(data, method="exponential_decay")
    
    # The discovered time constant should be close to 5
    expected_time_constant = 5.0
    error = abs(result.discovered_constant - expected_time_constant) / expected_time_constant
    passed = error < 0.15  # Allow 15% error for noisy data
    
    details = {
        'discovered_time_constant': result.discovered_constant,
        'expected_time_constant': expected_time_constant,
        'relative_error': error,
        'confidence': result.confidence
    }
    
    print(f"  Discovered Time Constant: {result.discovered_constant:.4f}")
    print(f"  Expected: {expected_time_constant:.4f}")
    print(f"  Relative Error: {error:.4%}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return ValidationResult("Exponential Decay Discovery", passed, details)


def validate_optimization_efficiency() -> ValidationResult:
    """Validate that golden ratio optimization is efficient."""
    print("\n" + "-"*70)
    print("Validating: Optimization Efficiency")
    print("-"*70)
    
    # Test on multiple objective functions
    def objective1(x):
        return (x - 5) ** 2
    
    def objective2(x):
        return x ** 2 - 4 * x + 7
    
    def objective3(x):
        return (x - 3.14159) ** 2 + 2
    
    objectives = [
        (objective1, 5.0, (0, 10)),
        (objective2, 2.0, (-5, 10)),
        (objective3, 3.14159, (0, 10))
    ]
    
    optimizer = OptimizationEngine(constraints={})
    
    errors = []
    iterations_list = []
    
    for obj_func, true_minimum, bounds in objectives:
        result = optimizer.golden_ratio_optimization(
            objective_function=obj_func,
            bounds=bounds,
            max_iterations=100
        )
        
        error = abs(result['optimal_x'] - true_minimum)
        errors.append(error)
        iterations_list.append(result['iterations'])
    
    avg_error = np.mean(errors)
    avg_iterations = np.mean(iterations_list)
    
    # Validation: average error should be small, iterations reasonable
    passed = avg_error < 0.01 and avg_iterations < 50
    
    details = {
        'average_error': avg_error,
        'average_iterations': avg_iterations,
        'individual_errors': errors,
        'individual_iterations': iterations_list
    }
    
    print(f"  Average Error: {avg_error:.6f}")
    print(f"  Average Iterations: {avg_iterations:.1f}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return ValidationResult("Optimization Efficiency", passed, details)


def validate_resource_allocation() -> ValidationResult:
    """Validate resource allocation follows golden ratio."""
    print("\n" + "-"*70)
    print("Validating: Resource Allocation")
    print("-"*70)
    
    measurer = ConstantsMeasurement(system_type="resource", constraints={})
    
    # Test multiple budget sizes
    budgets = [100, 500, 1000, 5000]
    ratios = []
    
    for budget in budgets:
        result = measurer.measure_resource_allocation(budget)
        ratio = result.empirical_evidence['allocation_ratio']
        ratios.append(ratio)
    
    avg_ratio = np.mean(ratios)
    expected_ratio = OPTIMAL_RESOURCE_SPLIT
    
    # All ratios should be close to 1/φ
    error = abs(avg_ratio - expected_ratio) / expected_ratio
    passed = error < 0.01
    
    details = {
        'average_ratio': avg_ratio,
        'expected_ratio': expected_ratio,
        'relative_error': error,
        'individual_ratios': ratios
    }
    
    print(f"  Average Allocation Ratio: {avg_ratio:.6f}")
    print(f"  Expected (1/φ): {expected_ratio:.6f}")
    print(f"  Relative Error: {error:.4%}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return ValidationResult("Resource Allocation", passed, details)


def validate_convergence_prediction() -> ValidationResult:
    """Validate learning convergence prediction."""
    print("\n" + "-"*70)
    print("Validating: Convergence Prediction")
    print("-"*70)
    
    # Simulate multiple learning curves
    num_curves = 10
    total_iterations = 100
    
    measurer = ConstantsMeasurement(system_type="learning", constraints={})
    
    predicted_convergence_points = []
    expected_point = total_iterations / EULER_NUMBER
    
    for _ in range(num_curves):
        # Generate learning curve with noise
        performance = [1 - np.exp(-i / (total_iterations / EULER_NUMBER)) + 
                      np.random.normal(0, 0.02) for i in range(20)]
        
        result = measurer.measure_learning_convergence(
            iterations=total_iterations,
            performance_data=performance
        )
        
        predicted_convergence_points.append(
            result.empirical_evidence['predicted_convergence_point']
        )
    
    avg_prediction = np.mean(predicted_convergence_points)
    error = abs(avg_prediction - expected_point) / expected_point
    
    # Prediction should be within 10% of T/e
    passed = error < 0.1
    
    details = {
        'average_prediction': avg_prediction,
        'expected_point': expected_point,
        'relative_error': error,
        'num_curves': num_curves
    }
    
    print(f"  Average Predicted Convergence: {avg_prediction:.2f}")
    print(f"  Expected (T/e): {expected_point:.2f}")
    print(f"  Relative Error: {error:.4%}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return ValidationResult("Convergence Prediction", passed, details)


def validate_boundary_analysis() -> ValidationResult:
    """Validate boundary analysis accuracy."""
    print("\n" + "-"*70)
    print("Validating: Boundary Analysis")
    print("-"*70)
    
    analyzer = BoundedSystemAnalyzer(system_parameters={})
    
    # Generate data with known mean
    np.random.seed(42)
    mean_value = 100
    data = np.random.normal(mean_value, 10, 50)
    
    result = analyzer.analyze_constraint_boundaries(
        constraint_type="resource",
        observed_data=data.tolist()
    )
    
    # Expected optimal boundary
    expected_boundary = mean_value * OPTIMAL_RESOURCE_SPLIT
    discovered_boundary = result['optimal_boundary']
    
    error = abs(discovered_boundary - expected_boundary) / expected_boundary
    passed = error < 0.1
    
    details = {
        'discovered_boundary': discovered_boundary,
        'expected_boundary': expected_boundary,
        'relative_error': error,
        'efficiency_ratio': result['efficiency_ratio']
    }
    
    print(f"  Discovered Boundary: {discovered_boundary:.4f}")
    print(f"  Expected Boundary: {expected_boundary:.4f}")
    print(f"  Relative Error: {error:.4%}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return ValidationResult("Boundary Analysis", passed, details)


def validate_ratio_consistency() -> ValidationResult:
    """Validate that discovered ratios are consistent."""
    print("\n" + "-"*70)
    print("Validating: Ratio Consistency")
    print("-"*70)
    
    discovery = ConstantDiscovery()
    
    # Generate multiple golden ratio sequences with different scales
    scales = [10, 100, 1000]
    discovered_ratios = []
    
    for scale in scales:
        data = [scale]
        for _ in range(8):
            data.append(data[-1] / GOLDEN_RATIO)
        
        result = discovery.discover_from_ratios(data, order=1)
        discovered_ratios.append(result.discovered_constant)
    
    # All discoveries should be similar
    std_dev = np.std(discovered_ratios)
    mean_ratio = np.mean(discovered_ratios)
    
    # Standard deviation should be small
    passed = std_dev < 0.1 and abs(mean_ratio - GOLDEN_RATIO) / GOLDEN_RATIO < 0.05
    
    details = {
        'discovered_ratios': discovered_ratios,
        'mean_ratio': mean_ratio,
        'std_dev': std_dev,
        'expected': GOLDEN_RATIO
    }
    
    print(f"  Mean Discovered Ratio: {mean_ratio:.6f}")
    print(f"  Standard Deviation: {std_dev:.6f}")
    print(f"  Expected: {GOLDEN_RATIO:.6f}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    
    return ValidationResult("Ratio Consistency", passed, details)


def run_all_validations() -> Dict:
    """Run all validation tests."""
    print("\n" + "="*70)
    print("CONSTRAINED INTELLIGENCE CONSTANTS - VALIDATION SUITE")
    print("="*70)
    
    validations = [
        validate_golden_ratio_discovery,
        validate_exponential_decay,
        validate_optimization_efficiency,
        validate_resource_allocation,
        validate_convergence_prediction,
        validate_boundary_analysis,
        validate_ratio_consistency,
    ]
    
    results = []
    for validation_func in validations:
        try:
            result = validation_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR in {validation_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(ValidationResult(validation_func.__name__, False, {'error': str(e)}))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for result in results:
        print(f"  {result}")
    
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    success_rate = passed_count / total_count if total_count > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"Total: {passed_count}/{total_count} validations passed")
    print(f"Success Rate: {success_rate:.1%}")
    print("="*70)
    
    if success_rate == 1.0:
        print("\n🎉 All validations passed! The framework is working correctly.")
    elif success_rate >= 0.8:
        print("\n⚠️  Most validations passed. Some minor issues detected.")
    else:
        print("\n❌ Multiple validations failed. Please review the results.")
    
    return {
        'results': results,
        'passed': passed_count,
        'total': total_count,
        'success_rate': success_rate
    }


def main():
    """Main entry point for validation."""
    results = run_all_validations()
    
    # Exit with appropriate code
    sys.exit(0 if results['success_rate'] == 1.0 else 1)


if __name__ == "__main__":
    main()
