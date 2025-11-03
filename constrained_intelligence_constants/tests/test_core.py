
"""
Tests for core module (ConstantsMeasurement, OptimizationEngine, BoundedSystemAnalyzer).
"""

import pytest
import numpy as np
from constrained_intelligence.core import (
    ConstantsMeasurement,
    OptimizationEngine,
    BoundedSystemAnalyzer,
    MeasurementResult,
)
from constrained_intelligence.constants import GOLDEN_RATIO, EULER_NUMBER


class TestConstantsMeasurement:
    """Tests for ConstantsMeasurement class."""
    
    def test_initialization(self):
        """Test that ConstantsMeasurement initializes correctly."""
        measurer = ConstantsMeasurement(
            system_type="test",
            constraints={"max_iter": 100}
        )
        assert measurer.system_type == "test"
        assert measurer.constraints["max_iter"] == 100
        assert len(measurer.measurements) == 0
    
    def test_measure_resource_allocation(self):
        """Test resource allocation measurement."""
        measurer = ConstantsMeasurement(system_type="resource", constraints={})
        result = measurer.measure_resource_allocation(resource_budget=1000)
        
        # Check result type
        assert isinstance(result, MeasurementResult)
        
        # Check golden ratio relationship
        allocated = result.empirical_evidence['optimal_allocated']
        reserved = result.empirical_evidence['reserved']
        
        assert abs(allocated + reserved - 1000) < 0.01
        assert abs(allocated / 1000 - 1/GOLDEN_RATIO) < 0.01
        
        # Check confidence
        assert 0 <= result.confidence <= 1
        assert result.confidence > 0.9
        
        # Check that measurement was recorded
        assert len(measurer.measurements) == 1
    
    def test_measure_learning_convergence(self):
        """Test learning convergence measurement."""
        measurer = ConstantsMeasurement(system_type="learning", constraints={})
        
        performance_data = [0.1, 0.3, 0.5, 0.7, 0.85, 0.92, 0.95]
        result = measurer.measure_learning_convergence(
            iterations=100,
            performance_data=performance_data
        )
        
        assert isinstance(result, MeasurementResult)
        assert result.constant_value == EULER_NUMBER
        
        # Check convergence prediction
        convergence_point = result.empirical_evidence['predicted_convergence_point']
        expected_point = 100 / EULER_NUMBER
        assert abs(convergence_point - expected_point) < 0.1
        
        # Check performance gain calculation
        gain = result.empirical_evidence['actual_performance_gain']
        assert abs(gain - (0.95 - 0.1)) < 0.01
    
    def test_measure_optimization_efficiency(self):
        """Test optimization efficiency measurement."""
        measurer = ConstantsMeasurement(system_type="optimization", constraints={})
        
        objective_values = [10, 8, 6, 5, 4.5, 4.2]
        resource_costs = [1, 2, 3, 4, 5, 6]
        
        result = measurer.measure_optimization_efficiency(
            objective_values=objective_values,
            resource_costs=resource_costs
        )
        
        assert isinstance(result, MeasurementResult)
        assert 'observed_max_efficiency' in result.empirical_evidence
        assert result.empirical_evidence['observed_max_efficiency'] > 0
    
    def test_invalid_efficiency_input(self):
        """Test that mismatched input lengths raise error."""
        measurer = ConstantsMeasurement(system_type="optimization", constraints={})
        
        with pytest.raises(ValueError):
            measurer.measure_optimization_efficiency(
                objective_values=[1, 2, 3],
                resource_costs=[1, 2]
            )


class TestOptimizationEngine:
    """Tests for OptimizationEngine class."""
    
    def test_initialization(self):
        """Test that OptimizationEngine initializes correctly."""
        optimizer = OptimizationEngine(constraints={"max_iter": 50})
        assert optimizer.constraints["max_iter"] == 50
        assert len(optimizer.optimization_history) == 0
    
    def test_golden_ratio_optimization_simple(self):
        """Test golden ratio optimization on simple quadratic."""
        optimizer = OptimizationEngine(constraints={})
        
        # Minimize (x - 5)^2
        def objective(x):
            return (x - 5) ** 2
        
        result = optimizer.golden_ratio_optimization(
            objective_function=objective,
            bounds=(0, 10),
            max_iterations=50
        )
        
        # Should find minimum near x=5
        assert abs(result['optimal_x'] - 5.0) < 0.01
        assert result['optimal_value'] < 0.01
        assert result['converged'] is True
        assert result['iterations'] > 0
    
    def test_golden_ratio_optimization_convergence(self):
        """Test that optimization converges within tolerance."""
        optimizer = OptimizationEngine(constraints={})
        
        def objective(x):
            return (x - 3.14159) ** 2
        
        result = optimizer.golden_ratio_optimization(
            objective_function=objective,
            bounds=(0, 10),
            max_iterations=100,
            tolerance=1e-6
        )
        
        assert result['converged'] is True
        assert abs(result['optimal_x'] - 3.14159) < 1e-3
    
    def test_exponential_decay_schedule(self):
        """Test exponential decay schedule generation."""
        optimizer = OptimizationEngine(constraints={})
        
        schedule = optimizer.exponential_decay_schedule(
            initial_value=1.0,
            decay_constant=0.1,
            steps=10
        )
        
        assert len(schedule) == 10
        assert schedule[0] == 1.0
        assert schedule[-1] < schedule[0]
        
        # Check exponential relationship
        for i in range(1, len(schedule)):
            expected = 1.0 * np.exp(-0.1 * i)
            assert abs(schedule[i] - expected) < 0.001
    
    def test_adaptive_step_size(self):
        """Test adaptive step size calculation."""
        optimizer = OptimizationEngine(constraints={})
        
        step_size = optimizer.adaptive_step_size(
            gradient_norm=2.0,
            iteration=10,
            base_lr=0.1
        )
        
        assert step_size > 0
        assert step_size <= 0.1  # Should be less than base LR
        
        # Later iterations should have smaller step size
        later_step = optimizer.adaptive_step_size(
            gradient_norm=2.0,
            iteration=100,
            base_lr=0.1
        )
        assert later_step < step_size


class TestBoundedSystemAnalyzer:
    """Tests for BoundedSystemAnalyzer class."""
    
    def test_initialization(self):
        """Test that BoundedSystemAnalyzer initializes correctly."""
        analyzer = BoundedSystemAnalyzer(system_parameters={"test": 123})
        assert analyzer.parameters["test"] == 123
        assert len(analyzer.analysis_results) == 0
    
    def test_analyze_resource_constraints(self):
        """Test resource constraint analysis."""
        analyzer = BoundedSystemAnalyzer(system_parameters={})
        
        data = [50, 55, 52, 58, 54, 56, 53, 57]
        result = analyzer.analyze_constraint_boundaries(
            constraint_type="resource",
            observed_data=data
        )
        
        assert result['constraint_type'] == 'resource'
        assert 'optimal_boundary' in result
        assert 'efficiency_ratio' in result
        assert 'stability_metric' in result
        assert 'recommended_bounds' in result
        
        # Check that optimal boundary is reasonable
        mean_val = np.mean(data)
        assert result['optimal_boundary'] > 0
        assert result['optimal_boundary'] < mean_val
    
    def test_analyze_temporal_constraints(self):
        """Test temporal constraint analysis."""
        analyzer = BoundedSystemAnalyzer(system_parameters={})
        
        # Generate exponentially decaying data
        data = [10 * np.exp(-0.1 * i) for i in range(20)]
        
        result = analyzer.analyze_constraint_boundaries(
            constraint_type="temporal",
            observed_data=data
        )
        
        assert result['constraint_type'] == 'temporal'
        assert 'characteristic_time' in result
        assert 'euler_ratio' in result
        assert 'decay_rate' in result
    
    def test_analyze_general_constraints(self):
        """Test general constraint analysis."""
        analyzer = BoundedSystemAnalyzer(system_parameters={})
        
        data = list(range(1, 21))
        result = analyzer.analyze_constraint_boundaries(
            constraint_type="general",
            observed_data=data
        )
        
        assert result['constraint_type'] == 'general'
        assert 'statistics' in result
        assert 'information_density' in result
        assert result['statistics']['mean'] == np.mean(data)
    
    def test_detect_emergent_patterns(self):
        """Test emergent pattern detection."""
        analyzer = BoundedSystemAnalyzer(system_parameters={})
        
        # Generate time series with known patterns
        t = np.linspace(0, 10, 100)
        time_series = 10 * np.exp(-t / 5) + 2 * np.sin(t)
        
        patterns = analyzer.detect_emergent_patterns(time_series.tolist())
        
        assert 'periodicity' in patterns
        assert 'convergence' in patterns
        assert 'scaling' in patterns
        
        # Should detect convergence (exponential decay)
        assert 'converging' in patterns['convergence']
    
    def test_detect_convergence(self):
        """Test convergence detection."""
        analyzer = BoundedSystemAnalyzer(system_parameters={})
        
        # Converging series
        converging_series = [10.0 - 9.0 * (1 - np.exp(-i/10)) for i in range(50)]
        patterns = analyzer.detect_emergent_patterns(converging_series, window_size=10)
        
        # Should detect convergence
        assert patterns['convergence']['converging'] == True


class TestMeasurementResult:
    """Tests for MeasurementResult dataclass."""
    
    def test_creation(self):
        """Test that MeasurementResult can be created."""
        result = MeasurementResult(
            constant_value=1.618,
            confidence=0.95,
            bounds=(1.5, 1.7),
            empirical_evidence={'test': 123},
            theoretical_basis="Test basis"
        )
        
        assert result.constant_value == 1.618
        assert result.confidence == 0.95
        assert result.bounds == (1.5, 1.7)
        assert result.empirical_evidence['test'] == 123
        assert result.theoretical_basis == "Test basis"
        assert isinstance(result.metadata, dict)


def test_integration_workflow():
    """Integration test for complete workflow."""
    # 1. Measure resource allocation
    measurer = ConstantsMeasurement(system_type="integration_test", constraints={})
    allocation_result = measurer.measure_resource_allocation(1000)
    
    # 2. Use allocation for optimization
    optimizer = OptimizationEngine(constraints={})
    allocated_budget = allocation_result.empirical_evidence['optimal_allocated']
    
    def objective(x):
        return (x - allocated_budget / 100) ** 2
    
    opt_result = optimizer.golden_ratio_optimization(
        objective_function=objective,
        bounds=(0, 20),
        max_iterations=50
    )
    
    # 3. Analyze the optimization trajectory
    analyzer = BoundedSystemAnalyzer(system_parameters={})
    trajectory = [h['optimal_x'] for h in optimizer.optimization_history[:10]]
    
    analysis = analyzer.analyze_constraint_boundaries(
        constraint_type="general",
        observed_data=trajectory
    )
    
    # Verify workflow completed successfully
    assert allocation_result.confidence > 0.9
    assert opt_result['converged'] is True
    assert analysis['constraint_type'] == 'general'
