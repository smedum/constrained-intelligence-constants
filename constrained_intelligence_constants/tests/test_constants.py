
"""
Tests for constants module.
"""

import math
import pytest
from constrained_intelligence.constants import (
    GOLDEN_RATIO,
    EULER_NUMBER,
    PI,
    NATURAL_LOG_BASE,
    OPTIMAL_RESOURCE_SPLIT,
    CONVERGENCE_THRESHOLD_FACTOR,
    LEARNING_RATE_BOUNDARY,
    MAX_EFFICIENCY_RATIO,
    MINIMAL_COMPLEXITY_CONSTANT,
    INFORMATION_DENSITY_LIMIT,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    VALIDATION_SIGNIFICANCE_LEVEL,
    get_constant_info,
    list_all_constants,
)


class TestFundamentalConstants:
    """Tests for fundamental mathematical constants."""
    
    def test_golden_ratio_value(self):
        """Test that golden ratio has correct value."""
        expected = (1 + math.sqrt(5)) / 2
        assert abs(GOLDEN_RATIO - expected) < 1e-10
        assert abs(GOLDEN_RATIO - 1.618033988749) < 1e-9
    
    def test_golden_ratio_property(self):
        """Test golden ratio mathematical property: φ² = φ + 1."""
        assert abs(GOLDEN_RATIO ** 2 - (GOLDEN_RATIO + 1)) < 1e-10
    
    def test_euler_number_value(self):
        """Test that Euler's number has correct value."""
        expected = math.exp(1)
        assert abs(EULER_NUMBER - expected) < 1e-10
        assert abs(EULER_NUMBER - 2.718281828459) < 1e-9
    
    def test_pi_value(self):
        """Test that pi has correct value."""
        assert abs(PI - math.pi) < 1e-10
        assert abs(PI - 3.14159265359) < 1e-9
    
    def test_natural_log_base(self):
        """Test that natural log base equals Euler's number."""
        assert NATURAL_LOG_BASE == EULER_NUMBER
        assert abs(NATURAL_LOG_BASE - math.e) < 1e-10


class TestDerivedConstants:
    """Tests for derived constants."""
    
    def test_optimal_resource_split(self):
        """Test optimal resource split calculation."""
        expected = 1 / GOLDEN_RATIO
        assert abs(OPTIMAL_RESOURCE_SPLIT - expected) < 1e-10
        assert 0.6 < OPTIMAL_RESOURCE_SPLIT < 0.7
    
    def test_convergence_threshold_factor(self):
        """Test convergence threshold factor."""
        expected = 1 / EULER_NUMBER
        assert abs(CONVERGENCE_THRESHOLD_FACTOR - expected) < 1e-10
        assert 0.3 < CONVERGENCE_THRESHOLD_FACTOR < 0.4
    
    def test_learning_rate_boundary(self):
        """Test learning rate boundary."""
        expected = 1 / (2 * PI)
        assert abs(LEARNING_RATE_BOUNDARY - expected) < 1e-10
        assert 0.15 < LEARNING_RATE_BOUNDARY < 0.17


class TestSystemBoundaryConstants:
    """Tests for system boundary constants."""
    
    def test_max_efficiency_ratio(self):
        """Test maximum efficiency ratio is in valid range."""
        assert 0 < MAX_EFFICIENCY_RATIO < 1
        assert 0.8 < MAX_EFFICIENCY_RATIO < 0.95
    
    def test_minimal_complexity_constant(self):
        """Test minimal complexity constant."""
        expected = math.e ** (1 / math.e)
        assert abs(MINIMAL_COMPLEXITY_CONSTANT - expected) < 1e-10
        assert MINIMAL_COMPLEXITY_CONSTANT > 1
    
    def test_information_density_limit(self):
        """Test information density limit."""
        expected = math.log(2) * 2
        assert abs(INFORMATION_DENSITY_LIMIT - expected) < 1e-10
        assert INFORMATION_DENSITY_LIMIT > 1


class TestThresholdConstants:
    """Tests for threshold constants."""
    
    def test_confidence_thresholds(self):
        """Test confidence thresholds are properly ordered."""
        assert 0 < MEDIUM_CONFIDENCE_THRESHOLD < HIGH_CONFIDENCE_THRESHOLD < 1
        assert HIGH_CONFIDENCE_THRESHOLD == 0.9
        assert MEDIUM_CONFIDENCE_THRESHOLD == 0.7
    
    def test_validation_significance_level(self):
        """Test validation significance level."""
        assert 0 < VALIDATION_SIGNIFICANCE_LEVEL < 0.1
        assert VALIDATION_SIGNIFICANCE_LEVEL == 0.05


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_get_constant_info_golden_ratio(self):
        """Test getting info for golden ratio."""
        info = get_constant_info('GOLDEN_RATIO')
        
        assert 'value' in info
        assert 'description' in info
        assert abs(info['value'] - GOLDEN_RATIO) < 1e-10
    
    def test_get_constant_info_euler(self):
        """Test getting info for Euler's number."""
        info = get_constant_info('EULER_NUMBER')
        
        assert 'value' in info
        assert info['value'] == EULER_NUMBER
    
    def test_get_constant_info_unknown(self):
        """Test getting info for unknown constant."""
        info = get_constant_info('UNKNOWN_CONSTANT')
        
        assert 'value' in info
        assert info['value'] is None
        assert 'Unknown' in info['description']
    
    def test_list_all_constants(self):
        """Test listing all constants."""
        constants = list_all_constants()
        
        assert isinstance(constants, dict)
        assert 'GOLDEN_RATIO' in constants
        assert 'EULER_NUMBER' in constants
        assert 'PI' in constants
        assert 'OPTIMAL_RESOURCE_SPLIT' in constants
        
        # Check values
        assert constants['GOLDEN_RATIO'] == GOLDEN_RATIO
        assert constants['EULER_NUMBER'] == EULER_NUMBER
        
        # Should have at least 10 constants
        assert len(constants) >= 10


class TestConstantRelationships:
    """Tests for relationships between constants."""
    
    def test_golden_ratio_reciprocal(self):
        """Test relationship between φ and 1/φ."""
        # φ - 1 = 1/φ
        assert abs((GOLDEN_RATIO - 1) - (1/GOLDEN_RATIO)) < 1e-10
        assert abs((GOLDEN_RATIO - 1) - OPTIMAL_RESOURCE_SPLIT) < 1e-10
    
    def test_euler_convergence(self):
        """Test Euler number through limit definition."""
        # e = lim(n→∞) (1 + 1/n)^n
        n = 100000
        approximation = (1 + 1/n) ** n
        assert abs(EULER_NUMBER - approximation) < 0.01
    
    def test_pi_circle_relationship(self):
        """Test pi through circle circumference."""
        # For radius 1, circumference = 2π
        radius = 1
        circumference = 2 * PI * radius
        assert abs(circumference - 2 * math.pi) < 1e-10


class TestConstantPrecision:
    """Tests for constant precision."""
    
    def test_golden_ratio_precision(self):
        """Test golden ratio has sufficient precision."""
        # Should match to at least 10 decimal places
        reference = 1.6180339887
        assert abs(GOLDEN_RATIO - reference) < 1e-9
    
    def test_euler_precision(self):
        """Test Euler's number has sufficient precision."""
        reference = 2.7182818285
        assert abs(EULER_NUMBER - reference) < 1e-9
    
    def test_pi_precision(self):
        """Test pi has sufficient precision."""
        reference = 3.1415926536
        assert abs(PI - reference) < 1e-9


def test_constant_immutability():
    """Test that constants cannot be accidentally modified."""
    # Python doesn't enforce true immutability for module-level variables,
    # but we can verify they're the expected types
    assert isinstance(GOLDEN_RATIO, float)
    assert isinstance(EULER_NUMBER, float)
    assert isinstance(PI, float)
    
    # Attempting to modify should not affect the original module constant
    original_value = GOLDEN_RATIO
    local_copy = GOLDEN_RATIO
    local_copy = 999
    
    # Re-import to verify module constant unchanged
    from constrained_intelligence.constants import GOLDEN_RATIO as phi
    assert phi == original_value


def test_all_constants_defined():
    """Test that all documented constants are defined."""
    required_constants = [
        'GOLDEN_RATIO',
        'EULER_NUMBER',
        'PI',
        'OPTIMAL_RESOURCE_SPLIT',
        'CONVERGENCE_THRESHOLD_FACTOR',
        'LEARNING_RATE_BOUNDARY',
        'MAX_EFFICIENCY_RATIO',
        'MINIMAL_COMPLEXITY_CONSTANT',
        'INFORMATION_DENSITY_LIMIT',
        'HIGH_CONFIDENCE_THRESHOLD',
        'MEDIUM_CONFIDENCE_THRESHOLD',
        'VALIDATION_SIGNIFICANCE_LEVEL',
    ]
    
    all_constants = list_all_constants()
    
    for const_name in required_constants:
        assert const_name in all_constants, f"Missing constant: {const_name}"
        assert all_constants[const_name] is not None


def test_constant_documentation():
    """Test that constants have documentation."""
    from constrained_intelligence import constants
    
    # Module should have docstring
    assert constants.__doc__ is not None
    assert len(constants.__doc__) > 100
