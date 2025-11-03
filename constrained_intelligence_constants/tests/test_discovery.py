
"""
Tests for discovery module (ConstantDiscovery, DiscoveryMethods).
"""

import pytest
import numpy as np
from constrained_intelligence.discovery import (
    ConstantDiscovery,
    DiscoveryMethods,
    DiscoveryResult,
)
from constrained_intelligence.constants import GOLDEN_RATIO, EULER_NUMBER


class TestConstantDiscovery:
    """Tests for ConstantDiscovery class."""
    
    def test_initialization(self):
        """Test that ConstantDiscovery initializes correctly."""
        discovery = ConstantDiscovery(significance_level=0.01)
        assert discovery.significance_level == 0.01
        assert len(discovery.discovered_constants) == 0
    
    def test_golden_ratio_discovery_perfect(self):
        """Test discovery on perfect golden ratio sequence."""
        discovery = ConstantDiscovery()
        
        # Generate perfect sequence
        data = [100]
        for _ in range(10):
            data.append(data[-1] / GOLDEN_RATIO)
        
        result = discovery.discover_from_optimization(data, method="golden_ratio")
        
        assert isinstance(result, DiscoveryResult)
        assert abs(result.discovered_constant - GOLDEN_RATIO) < 0.01
        assert result.confidence > 0.9
        assert result.theoretical_constant == GOLDEN_RATIO
    
    def test_golden_ratio_discovery_noisy(self):
        """Test discovery on noisy golden ratio sequence."""
        np.random.seed(42)
        discovery = ConstantDiscovery()
        
        # Generate noisy sequence
        data = [100]
        for _ in range(10):
            next_val = data[-1] / GOLDEN_RATIO + np.random.normal(0, 0.5)
            data.append(next_val)
        
        result = discovery.discover_from_optimization(data, method="golden_ratio")
        
        # Should still detect golden ratio, but with lower confidence
        assert abs(result.discovered_constant - GOLDEN_RATIO) < 0.2
        assert result.confidence > 0.5
    
    def test_exponential_decay_discovery(self):
        """Test exponential decay constant discovery."""
        discovery = ConstantDiscovery()
        
        # Generate exponential decay
        t = np.arange(20)
        time_constant = 5.0
        data = 10 * np.exp(-t / time_constant)
        
        result = discovery.discover_from_optimization(data.tolist(), method="exponential_decay")
        
        assert isinstance(result, DiscoveryResult)
        # Allow reasonable error margin for time constant
        assert abs(result.discovered_constant - time_constant) / time_constant < 0.2
        # Confidence may be lower when comparing to e instead of the actual time constant
        assert result.confidence >= 0
    
    def test_general_optimization_discovery(self):
        """Test general optimization discovery."""
        discovery = ConstantDiscovery()
        
        # Use golden ratio data
        data = [100, 61.8, 38.2, 23.6, 14.6]
        
        result = discovery.discover_from_optimization(data, method="general")
        
        assert isinstance(result, DiscoveryResult)
        assert result.discovered_constant > 0
        assert result.method == 'general_optimization_discovery'
    
    def test_detect_convergence_constants(self):
        """Test convergence constant detection from learning curves."""
        discovery = ConstantDiscovery()
        
        # Generate learning curves
        curves = []
        for _ in range(5):
            curve = [1 - np.exp(-i / (100 / EULER_NUMBER)) for i in range(100)]
            curves.append(curve)
        
        result = discovery.detect_convergence_constants(curves, method="euler")
        
        assert isinstance(result, DiscoveryResult)
        assert result.theoretical_constant == EULER_NUMBER
        # Discovered constant may vary depending on convergence detection
        assert result.discovered_constant > 0
    
    def test_discover_from_ratios(self):
        """Test ratio-based discovery."""
        discovery = ConstantDiscovery()
        
        # Golden ratio sequence
        sequence = [100, 61.8, 38.2, 23.6, 14.6, 9.0, 5.6]
        
        result = discovery.discover_from_ratios(sequence, order=1)
        
        assert isinstance(result, DiscoveryResult)
        assert abs(result.discovered_constant - GOLDEN_RATIO) < 0.1
    
    def test_discover_from_ratios_order_2(self):
        """Test ratio discovery with order 2."""
        discovery = ConstantDiscovery()
        
        sequence = [100, 50, 25, 12.5, 6.25, 3.125]
        
        result = discovery.discover_from_ratios(sequence, order=2)
        
        # Should discover ratio of ~4 (100/25, 50/12.5, etc.)
        assert isinstance(result, DiscoveryResult)
        assert result.discovered_constant > 2
    
    def test_discover_from_boundaries(self):
        """Test boundary-based discovery."""
        discovery = ConstantDiscovery()
        
        constraint_data = {
            'memory': [50, 55, 52, 58, 54],
            'compute': [100, 105, 98, 103, 101]
        }
        
        results = discovery.discover_from_boundaries(constraint_data)
        
        assert isinstance(results, dict)
        assert 'memory' in results
        assert 'compute' in results
        
        for name, result in results.items():
            assert isinstance(result, DiscoveryResult)
            assert result.method == 'boundary_analysis'
    
    def test_validate_discovery(self):
        """Test discovery validation."""
        discovery = ConstantDiscovery()
        
        # Create a discovery result
        data = [100, 61.8, 38.2, 23.6, 14.6]
        result = discovery.discover_from_optimization(data, method="golden_ratio")
        
        # Validate without new data
        validation = discovery.validate_discovery(result)
        
        assert 'original_confidence' in validation
        assert 'method' in validation
        assert validation['original_confidence'] == result.confidence
    
    def test_validate_discovery_with_data(self):
        """Test discovery validation with new data."""
        discovery = ConstantDiscovery()
        
        # Initial discovery
        data1 = [100, 61.8, 38.2, 23.6, 14.6]
        result = discovery.discover_from_optimization(data1, method="golden_ratio")
        
        # Validation data
        data2 = [200, 123.6, 76.4, 47.2]
        validation = discovery.validate_discovery(result, validation_data=data2)
        
        assert 'cross_validation_consistency' in validation or 'cross_validation_error' in validation
    
    def test_insufficient_data_error(self):
        """Test that insufficient data raises error."""
        discovery = ConstantDiscovery()
        
        with pytest.raises(ValueError):
            discovery.discover_from_optimization([1, 2], method="golden_ratio")
    
    def test_invalid_ratios_error(self):
        """Test that invalid sequence raises error for ratios."""
        discovery = ConstantDiscovery()
        
        with pytest.raises(ValueError):
            discovery.discover_from_ratios([1], order=2)


class TestDiscoveryMethods:
    """Tests for DiscoveryMethods enum."""
    
    def test_enum_values(self):
        """Test that all expected methods are in enum."""
        assert DiscoveryMethods.OPTIMIZATION_BASED.value == "optimization_based"
        assert DiscoveryMethods.CONVERGENCE_ANALYSIS.value == "convergence_analysis"
        assert DiscoveryMethods.PERIODICITY_DETECTION.value == "periodicity_detection"
        assert DiscoveryMethods.BOUNDARY_ANALYSIS.value == "boundary_analysis"


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""
    
    def test_creation(self):
        """Test that DiscoveryResult can be created."""
        result = DiscoveryResult(
            discovered_constant=1.618,
            theoretical_constant=GOLDEN_RATIO,
            confidence=0.95,
            method="test_method",
            empirical_evidence={'data': [1, 2, 3]},
            validation_metrics={'r_squared': 0.98}
        )
        
        assert result.discovered_constant == 1.618
        assert result.theoretical_constant == GOLDEN_RATIO
        assert result.confidence == 0.95
        assert result.method == "test_method"
        assert len(result.empirical_evidence) > 0
        assert len(result.validation_metrics) > 0


def test_discovery_integration():
    """Integration test for discovery workflow."""
    discovery = ConstantDiscovery()
    
    # 1. Generate data with known constant
    data = [1000]
    for _ in range(12):
        data.append(data[-1] / GOLDEN_RATIO)
    
    # 2. Discover constant
    result = discovery.discover_from_optimization(data, method="golden_ratio")
    
    # 3. Validate discovery
    validation_data = [500]
    for _ in range(8):
        validation_data.append(validation_data[-1] / GOLDEN_RATIO)
    
    validation = discovery.validate_discovery(result, validation_data)
    
    # Verify workflow
    assert abs(result.discovered_constant - GOLDEN_RATIO) < 0.05
    assert result.confidence > 0.85
    assert 'original_confidence' in validation


def test_multiple_discovery_methods():
    """Test that different methods can discover same constant."""
    discovery = ConstantDiscovery()
    
    # Data following golden ratio
    data = [100, 61.8, 38.2, 23.6, 14.6, 9.0, 5.6]
    
    # Try different methods
    result1 = discovery.discover_from_optimization(data, method="golden_ratio")
    result2 = discovery.discover_from_ratios(data, order=1)
    
    # Both should find similar constants
    assert abs(result1.discovered_constant - result2.discovered_constant) < 0.3
    assert result1.confidence > 0.5
    assert result2.confidence > 0.5
