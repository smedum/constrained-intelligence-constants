
"""
Basic usage examples for Constrained Intelligence Constants.

This script demonstrates simple, common use cases for the package.
"""

import sys
import os

# Add parent directory to path for local imports
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


def example_1_resource_allocation():
    """Example 1: Optimal resource allocation using golden ratio."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Optimal Resource Allocation")
    print("="*70)
    
    # You have 1000 units of compute to allocate
    total_budget = 1000
    
    measurer = ConstantsMeasurement(
        system_type="resource_allocation",
        constraints={"total_budget": total_budget}
    )
    
    result = measurer.measure_resource_allocation(total_budget)
    
    print(f"\nTotal Budget: {total_budget} units")
    print(f"Optimal Allocation: {result.empirical_evidence['optimal_allocated']:.2f} units")
    print(f"Reserve: {result.empirical_evidence['reserved']:.2f} units")
    print(f"Allocation Ratio: {result.empirical_evidence['allocation_ratio']:.4f}")
    print(f"Golden Ratio (1/φ): {OPTIMAL_RESOURCE_SPLIT:.4f}")
    print(f"Confidence: {result.confidence:.2%}")
    
    print("\n💡 Insight: Allocate ~61.8% for active use, reserve ~38.2% for adaptation")


def example_2_golden_ratio_optimization():
    """Example 2: Find function minimum using golden ratio search."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Golden Ratio Optimization")
    print("="*70)
    
    # Define a simple quadratic function
    def objective_function(x):
        """Function to minimize: (x - 7)²"""
        return (x - 7) ** 2
    
    optimizer = OptimizationEngine(constraints={})
    
    result = optimizer.golden_ratio_optimization(
        objective_function=objective_function,
        bounds=(0, 15),
        max_iterations=50
    )
    
    print(f"\nObjective Function: f(x) = (x - 7)²")
    print(f"Search Bounds: [0, 15]")
    print(f"Optimal x: {result['optimal_x']:.6f}")
    print(f"Optimal f(x): {result['optimal_value']:.6f}")
    print(f"Iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")
    
    print("\n💡 Insight: Golden ratio search finds the minimum efficiently")


def example_3_learning_rate_schedule():
    """Example 3: Create exponential decay learning rate schedule."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Learning Rate Schedule")
    print("="*70)
    
    optimizer = OptimizationEngine(constraints={})
    
    # Create learning rate schedule
    initial_lr = 0.1
    decay_constant = 0.05
    num_steps = 100
    
    schedule = optimizer.exponential_decay_schedule(
        initial_value=initial_lr,
        decay_constant=decay_constant,
        steps=num_steps
    )
    
    print(f"\nInitial Learning Rate: {initial_lr}")
    print(f"Decay Constant: {decay_constant}")
    print(f"Number of Steps: {num_steps}")
    print(f"\nLearning Rate Schedule:")
    print(f"  Step 0: {schedule[0]:.6f}")
    print(f"  Step 25: {schedule[25]:.6f}")
    print(f"  Step 50: {schedule[50]:.6f}")
    print(f"  Step 75: {schedule[75]:.6f}")
    print(f"  Step 99: {schedule[99]:.6f}")
    
    print("\n💡 Insight: Exponential decay based on e provides smooth convergence")


def example_4_convergence_prediction():
    """Example 4: Predict learning convergence."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Learning Convergence Prediction")
    print("="*70)
    
    # Simulate learning curve (performance improving over time)
    performance_data = [0.10, 0.25, 0.45, 0.60, 0.72, 0.81, 0.87, 0.91, 0.94, 0.95]
    total_iterations = 100
    
    measurer = ConstantsMeasurement(
        system_type="learning",
        constraints={}
    )
    
    result = measurer.measure_learning_convergence(
        iterations=total_iterations,
        performance_data=performance_data
    )
    
    print(f"\nTotal Iterations: {total_iterations}")
    print(f"Performance Data Points: {len(performance_data)}")
    print(f"Initial Performance: {performance_data[0]:.2f}")
    print(f"Current Performance: {performance_data[-1]:.2f}")
    print(f"\nPredicted Convergence Point: iteration {result.empirical_evidence['predicted_convergence_point']:.1f}")
    print(f"Expected by iteration: {int(total_iterations / EULER_NUMBER)}")
    print(f"Total Performance Gain: {result.empirical_evidence['actual_performance_gain']:.2f}")
    print(f"Confidence: {result.confidence:.2%}")
    
    print("\n💡 Insight: Learning typically converges around T/e iterations")


def example_5_constant_discovery():
    """Example 5: Discover constants from optimization data."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Constant Discovery")
    print("="*70)
    
    # Simulated optimization data (decreasing by golden ratio)
    optimization_data = [100.0, 61.8, 38.2, 23.6, 14.6, 9.0, 5.6, 3.4, 2.1]
    
    discovery = ConstantDiscovery()
    
    result = discovery.discover_from_optimization(
        optimization_data,
        method="golden_ratio"
    )
    
    print(f"\nOptimization Data: {optimization_data[:5]}...")
    print(f"\nDiscovered Constant: {result.discovered_constant:.4f}")
    print(f"Theoretical Golden Ratio: {result.theoretical_constant:.4f}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Discovery Method: {result.method}")
    
    convergence = result.empirical_evidence.get('convergence_pattern', 0)
    print(f"Convergence Pattern: {convergence:.2%} of ratios near φ")
    
    print("\n💡 Insight: Golden ratio naturally emerges from optimization processes")


def example_6_boundary_analysis():
    """Example 6: Analyze system boundaries."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Boundary Analysis")
    print("="*70)
    
    # Simulated resource usage data
    resource_usage = [42, 58, 51, 63, 55, 59, 48, 61, 54, 57]
    
    analyzer = BoundedSystemAnalyzer(system_parameters={})
    
    result = analyzer.analyze_constraint_boundaries(
        constraint_type="resource",
        observed_data=resource_usage
    )
    
    print(f"\nObserved Resource Usage: {resource_usage}")
    print(f"Mean Usage: {sum(resource_usage)/len(resource_usage):.2f}")
    print(f"\nOptimal Boundary: {result['optimal_boundary']:.2f}")
    print(f"Efficiency Ratio: {result['efficiency_ratio']:.4f}")
    print(f"Stability Metric: {result['stability_metric']:.4f}")
    print(f"Recommended Bounds: [{result['recommended_bounds'][0]:.2f}, {result['recommended_bounds'][1]:.2f}]")
    
    print("\n💡 Insight: Optimal boundaries follow mathematical constants")


def example_7_using_constants():
    """Example 7: Directly using predefined constants."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Using Predefined Constants")
    print("="*70)
    
    from constrained_intelligence import (
        CONVERGENCE_THRESHOLD_FACTOR,
        LEARNING_RATE_BOUNDARY,
        MAX_EFFICIENCY_RATIO,
    )
    
    print("\n📊 Fundamental Constants:")
    print(f"  Golden Ratio (φ): {GOLDEN_RATIO:.6f}")
    print(f"  Euler's Number (e): {EULER_NUMBER:.6f}")
    
    print("\n📊 Derived Constants for AI:")
    print(f"  Optimal Resource Split (1/φ): {OPTIMAL_RESOURCE_SPLIT:.6f}")
    print(f"  Convergence Factor (1/e): {CONVERGENCE_THRESHOLD_FACTOR:.6f}")
    print(f"  Learning Rate Boundary: {LEARNING_RATE_BOUNDARY:.6f}")
    
    print("\n📊 System Boundaries:")
    print(f"  Max Efficiency Ratio: {MAX_EFFICIENCY_RATIO:.6f}")
    
    # Example usage
    max_iterations = 1000
    check_convergence_at = int(max_iterations * CONVERGENCE_THRESHOLD_FACTOR)
    
    print(f"\n💡 Example: For {max_iterations} iterations, check convergence at iteration {check_convergence_at}")


def main():
    """Run all basic examples."""
    print("\n" + "🧠 CONSTRAINED INTELLIGENCE CONSTANTS - Basic Examples ".center(70, "="))
    
    examples = [
        example_1_resource_allocation,
        example_2_golden_ratio_optimization,
        example_3_learning_rate_schedule,
        example_4_convergence_prediction,
        example_5_constant_discovery,
        example_6_boundary_analysis,
        example_7_using_constants,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ All basic examples completed!")
    print("="*70)
    print("\n📚 Next steps:")
    print("  - Check advanced_examples.py for complex use cases")
    print("  - Explore the interactive notebook: notebook.ipynb")
    print("  - Read the documentation: README.md")
    print()


if __name__ == "__main__":
    main()
