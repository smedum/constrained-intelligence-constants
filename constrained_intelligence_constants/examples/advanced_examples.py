
"""
Advanced usage examples for Constrained Intelligence Constants.

This script demonstrates complex use cases and real-world applications.
"""

import sys
import os
import numpy as np

# Add parent directory to path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constrained_intelligence import (
    ConstantsMeasurement,
    OptimizationEngine,
    BoundedSystemAnalyzer,
    ConstantDiscovery,
    DiscoveryMethods,
    GOLDEN_RATIO,
    EULER_NUMBER,
)


def advanced_example_1_multi_objective_optimization():
    """Advanced Example 1: Multi-objective optimization with constraints."""
    print("\n" + "="*70)
    print("ADVANCED EXAMPLE 1: Multi-Objective Optimization")
    print("="*70)
    
    # Simulate a multi-objective problem: balance speed vs accuracy
    # Objective 1: Maximize accuracy (minimize negative accuracy)
    # Objective 2: Minimize computation time
    
    def objective_accuracy(allocation_ratio):
        """Accuracy increases with resources, but with diminishing returns."""
        return -(1 - np.exp(-2 * allocation_ratio))  # Negative for minimization
    
    def objective_speed(allocation_ratio):
        """Speed decreases with more resources."""
        return allocation_ratio ** 2
    
    def combined_objective(x):
        """Weighted combination using golden ratio."""
        return GOLDEN_RATIO * objective_accuracy(x) + (1/GOLDEN_RATIO) * objective_speed(x)
    
    optimizer = OptimizationEngine(constraints={})
    
    result = optimizer.golden_ratio_optimization(
        objective_function=combined_objective,
        bounds=(0, 1),
        max_iterations=100
    )
    
    optimal_ratio = result['optimal_x']
    accuracy = -objective_accuracy(optimal_ratio)
    speed = objective_speed(optimal_ratio)
    
    print(f"\nOptimal Allocation Ratio: {optimal_ratio:.4f}")
    print(f"Achieved Accuracy: {accuracy:.4f}")
    print(f"Computation Cost: {speed:.4f}")
    print(f"Trade-off ratio (φ-based weighting): {GOLDEN_RATIO:.4f} vs {1/GOLDEN_RATIO:.4f}")
    
    print("\n💡 Insight: Golden ratio provides optimal multi-objective trade-offs")


def advanced_example_2_adaptive_learning_system():
    """Advanced Example 2: Adaptive learning system with dynamic schedules."""
    print("\n" + "="*70)
    print("ADVANCED EXAMPLE 2: Adaptive Learning System")
    print("="*70)
    
    # Simulate training with adaptive learning rates
    epochs = 100
    initial_lr = 0.1
    
    optimizer = OptimizationEngine(constraints={})
    measurer = ConstantsMeasurement(system_type="learning", constraints={})
    
    # Generate performance curve
    performance = []
    learning_rates = []
    
    for epoch in range(epochs):
        # Exponential decay schedule
        lr = initial_lr * np.exp(-epoch / (epochs / EULER_NUMBER))
        learning_rates.append(lr)
        
        # Simulate performance improvement (logarithmic growth)
        perf = 1 - np.exp(-epoch / (epochs / EULER_NUMBER))
        performance.append(perf)
    
    # Analyze convergence
    result = measurer.measure_learning_convergence(
        iterations=epochs,
        performance_data=performance
    )
    
    convergence_point = result.empirical_evidence['predicted_convergence_point']
    
    print(f"\nTraining Epochs: {epochs}")
    print(f"Initial Learning Rate: {initial_lr}")
    print(f"Predicted Convergence: epoch {convergence_point:.1f}")
    print(f"Actual Performance at convergence: {performance[int(convergence_point)]:.4f}")
    print(f"Final Performance: {performance[-1]:.4f}")
    
    # Check if we converged as predicted
    convergence_accuracy = abs(convergence_point - epochs/EULER_NUMBER) / (epochs/EULER_NUMBER)
    print(f"Convergence Prediction Accuracy: {(1-convergence_accuracy)*100:.1f}%")
    
    print("\n💡 Insight: Exponential schedules based on e achieve optimal convergence")


def advanced_example_3_resource_allocation_under_uncertainty():
    """Advanced Example 3: Dynamic resource allocation with uncertainty."""
    print("\n" + "="*70)
    print("ADVANCED EXAMPLE 3: Resource Allocation Under Uncertainty")
    print("="*70)
    
    # Simulate varying workload
    np.random.seed(42)
    num_timesteps = 50
    base_load = 100
    
    # Varying load with noise
    workload = base_load + 30 * np.sin(np.linspace(0, 4*np.pi, num_timesteps)) + np.random.normal(0, 10, num_timesteps)
    
    measurer = ConstantsMeasurement(system_type="dynamic_allocation", constraints={})
    
    # Allocate resources at each timestep
    allocations = []
    reserves = []
    
    for load in workload:
        result = measurer.measure_resource_allocation(load)
        allocations.append(result.empirical_evidence['optimal_allocated'])
        reserves.append(result.empirical_evidence['reserved'])
    
    # Analysis
    avg_allocation = np.mean(allocations)
    avg_reserve = np.mean(reserves)
    allocation_ratio = avg_allocation / (avg_allocation + avg_reserve)
    
    print(f"\nTimesteps: {num_timesteps}")
    print(f"Average Workload: {np.mean(workload):.2f}")
    print(f"Workload Std Dev: {np.std(workload):.2f}")
    print(f"\nAverage Allocation: {avg_allocation:.2f}")
    print(f"Average Reserve: {avg_reserve:.2f}")
    print(f"Allocation Ratio: {allocation_ratio:.4f}")
    print(f"Target Ratio (1/φ): {1/GOLDEN_RATIO:.4f}")
    print(f"Ratio Accuracy: {abs(allocation_ratio - 1/GOLDEN_RATIO)*100:.2f}% deviation")
    
    print("\n💡 Insight: Golden ratio allocation remains optimal under uncertainty")


def advanced_example_4_hierarchical_constant_discovery():
    """Advanced Example 4: Discover constants at multiple levels."""
    print("\n" + "="*70)
    print("ADVANCED EXAMPLE 4: Hierarchical Constant Discovery")
    print("="*70)
    
    # Generate hierarchical data: outer process follows golden ratio,
    # inner process follows exponential decay
    
    discovery = ConstantDiscovery()
    
    # Level 1: Optimization trajectory
    level1_data = [100, 61.8, 38.2, 23.6, 14.6]
    result1 = discovery.discover_from_optimization(level1_data, method="golden_ratio")
    
    print(f"\nLevel 1 Discovery (Optimization):")
    print(f"  Data: {level1_data}")
    print(f"  Discovered: {result1.discovered_constant:.4f}")
    print(f"  Expected: {GOLDEN_RATIO:.4f}")
    print(f"  Confidence: {result1.confidence:.2%}")
    
    # Level 2: Learning convergence
    level2_data = []
    for i in range(20):
        level2_data.append(1 - np.exp(-i * 0.2))
    
    result2 = discovery.discover_from_optimization(level2_data, method="exponential_decay")
    
    print(f"\nLevel 2 Discovery (Convergence):")
    print(f"  Data points: {len(level2_data)}")
    print(f"  Discovered time constant: {result2.discovered_constant:.4f}")
    print(f"  Expected (e): {EULER_NUMBER:.4f}")
    print(f"  Confidence: {result2.confidence:.2%}")
    
    # Level 3: Ratio analysis
    level3_data = [10.0, 6.18, 3.82, 2.36, 1.46]
    result3 = discovery.discover_from_ratios(level3_data, order=1)
    
    print(f"\nLevel 3 Discovery (Ratios):")
    print(f"  Data: {level3_data}")
    print(f"  Discovered ratio: {result3.discovered_constant:.4f}")
    print(f"  Confidence: {result3.confidence:.2%}")
    
    print("\n💡 Insight: Constants emerge at multiple levels of system hierarchy")


def advanced_example_5_real_world_ml_pipeline():
    """Advanced Example 5: Real-world ML pipeline optimization."""
    print("\n" + "="*70)
    print("ADVANCED EXAMPLE 5: Real-World ML Pipeline")
    print("="*70)
    
    # Simulate a complete ML pipeline with resource constraints
    
    total_compute_budget = 1000  # arbitrary units
    total_time_budget = 100      # arbitrary time units
    
    measurer = ConstantsMeasurement(system_type="ml_pipeline", constraints={
        "compute": total_compute_budget,
        "time": total_time_budget
    })
    
    # 1. Allocate resources between stages
    print("\n1. Resource Allocation Across Pipeline Stages:")
    
    stages = ["data_preprocessing", "feature_engineering", "model_training", "validation"]
    
    # Allocate using golden ratio principle
    remaining_budget = total_compute_budget
    stage_allocations = {}
    
    for i, stage in enumerate(stages):
        if i < len(stages) - 1:
            result = measurer.measure_resource_allocation(remaining_budget)
            allocation = result.empirical_evidence['optimal_allocated']
            stage_allocations[stage] = allocation
            remaining_budget = result.empirical_evidence['reserved']
        else:
            stage_allocations[stage] = remaining_budget
    
    for stage, allocation in stage_allocations.items():
        print(f"  {stage}: {allocation:.2f} units ({allocation/total_compute_budget*100:.1f}%)")
    
    # 2. Learning rate schedule for training
    print("\n2. Learning Rate Schedule:")
    
    optimizer = OptimizationEngine(constraints={})
    num_epochs = 50
    initial_lr = 0.1
    
    lr_schedule = optimizer.exponential_decay_schedule(
        initial_value=initial_lr,
        decay_constant=1/EULER_NUMBER,
        steps=num_epochs
    )
    
    print(f"  Initial LR: {lr_schedule[0]:.6f}")
    print(f"  LR at epoch {num_epochs//3}: {lr_schedule[num_epochs//3]:.6f}")
    print(f"  Final LR: {lr_schedule[-1]:.6f}")
    
    # 3. Early stopping criterion
    print("\n3. Early Stopping Criterion:")
    
    convergence_check = int(num_epochs / EULER_NUMBER)
    print(f"  Check convergence at epoch: {convergence_check}")
    print(f"  Based on e-factor: {1/EULER_NUMBER:.4f}")
    
    # 4. Train-validation split
    print("\n4. Train-Validation Split:")
    
    total_data_points = 10000
    train_split = 1 / GOLDEN_RATIO
    train_size = int(total_data_points * train_split)
    val_size = total_data_points - train_size
    
    print(f"  Total Data: {total_data_points}")
    print(f"  Training: {train_size} ({train_split*100:.1f}%)")
    print(f"  Validation: {val_size} ({(1-train_split)*100:.1f}%)")
    
    print("\n💡 Insight: Mathematical constants guide the entire ML pipeline")


def advanced_example_6_emergent_pattern_detection():
    """Advanced Example 6: Detect emergent patterns in complex systems."""
    print("\n" + "="*70)
    print("ADVANCED EXAMPLE 6: Emergent Pattern Detection")
    print("="*70)
    
    # Generate complex time series with multiple patterns
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    
    # Combine multiple patterns
    exponential_decay = 10 * np.exp(-t / EULER_NUMBER)
    periodic = 2 * np.sin(2 * np.pi * t / GOLDEN_RATIO)
    noise = np.random.normal(0, 0.5, len(t))
    
    time_series = exponential_decay + periodic + noise
    
    analyzer = BoundedSystemAnalyzer(system_parameters={})
    
    patterns = analyzer.detect_emergent_patterns(time_series, window_size=20)
    
    print(f"\nTime Series Length: {len(time_series)}")
    print(f"\nDetected Patterns:")
    
    print(f"\n  Periodicity:")
    if patterns['periodicity']['detected']:
        print(f"    Detected: Yes")
        print(f"    Period: {patterns['periodicity'].get('period', 'N/A')}")
    else:
        print(f"    Detected: No")
    
    print(f"\n  Convergence:")
    print(f"    Converging: {patterns['convergence']['converging']}")
    print(f"    Variance Reduction: {patterns['convergence'].get('variance_reduction', 0):.2%}")
    
    print(f"\n  Scaling:")
    print(f"    Type: {patterns['scaling']['scaling_type']}")
    if 'parameter' in patterns['scaling']:
        print(f"    Parameter: {patterns['scaling']['parameter']:.4f}")
    
    print("\n💡 Insight: Complex systems exhibit patterns governed by mathematical constants")


def main():
    """Run all advanced examples."""
    print("\n" + "🧠 CONSTRAINED INTELLIGENCE CONSTANTS - Advanced Examples ".center(70, "="))
    
    examples = [
        advanced_example_1_multi_objective_optimization,
        advanced_example_2_adaptive_learning_system,
        advanced_example_3_resource_allocation_under_uncertainty,
        advanced_example_4_hierarchical_constant_discovery,
        advanced_example_5_real_world_ml_pipeline,
        advanced_example_6_emergent_pattern_detection,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ All advanced examples completed!")
    print("="*70)
    print("\n📚 Next steps:")
    print("  - Run validation experiments: python validation/experimental_validation.py")
    print("  - Explore the theory: THEORY.md")
    print("  - Contribute: CONTRIBUTING.md")
    print()


if __name__ == "__main__":
    main()
