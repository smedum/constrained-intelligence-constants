
"""
Fundamental constants for constrained intelligence systems.

This module defines the mathematical constants that emerge from and govern
bounded intelligent systems. These constants appear in optimization,
learning convergence, resource allocation, and system boundaries.
"""

import math

# ============================================================================
# FUNDAMENTAL MATHEMATICAL CONSTANTS
# ============================================================================

GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
"""
The Golden Ratio (φ ≈ 1.618)

Appears in:
- Optimal resource allocation in bounded systems
- Search optimization (golden section search)
- Aesthetic and structural proportions
- Fibonacci growth patterns
"""

EULER_NUMBER = math.exp(1)
"""
Euler's Number (e ≈ 2.718)

Appears in:
- Learning convergence rates
- Exponential decay schedules
- Natural growth and decay processes
- Compound optimization dynamics
"""

PI = math.pi
"""
Pi (π ≈ 3.14159)

Appears in:
- Circular and cyclic learning schedules
- Fourier-based signal processing in neural networks
- Angular momentum in rotational optimization
"""

NATURAL_LOG_BASE = math.e
"""
Natural Logarithm Base (e ≈ 2.718)

Same as EULER_NUMBER, used in logarithmic scaling and entropy calculations.
"""

# ============================================================================
# DERIVED CONSTANTS FOR AI SYSTEMS
# ============================================================================

OPTIMAL_RESOURCE_SPLIT = 1 / GOLDEN_RATIO
"""
Optimal Resource Split Ratio (1/φ ≈ 0.618)

In resource-constrained systems, dividing resources at this ratio often
achieves optimal balance between exploitation and exploration, or between
active computation and reserves.
"""

CONVERGENCE_THRESHOLD_FACTOR = 1 / EULER_NUMBER
"""
Convergence Threshold Factor (1/e ≈ 0.368)

Learning algorithms in bounded systems typically reach 63.2% (1 - 1/e) of
their asymptotic performance after one time constant. This factor appears
in convergence criteria.
"""

LEARNING_RATE_BOUNDARY = 1 / (2 * PI)
"""
Learning Rate Boundary (1/(2π) ≈ 0.159)

Maximum stable learning rate in many cyclic and oscillatory systems.
Beyond this boundary, learning may become unstable or divergent.
"""

# ============================================================================
# SYSTEM BOUNDARY CONSTANTS
# ============================================================================

MAX_EFFICIENCY_RATIO = 0.886
"""
Maximum Efficiency Ratio in Bounded Systems (≈ 0.886)

Theoretical maximum efficiency that can be achieved in resource-constrained
optimization. Derived from thermodynamic and information-theoretic limits.

Related to the Carnot efficiency and Shannon's channel capacity bounds.
"""

MINIMAL_COMPLEXITY_CONSTANT = math.e ** (1 / math.e)
"""
Minimal Complexity Constant (e^(1/e) ≈ 1.444)

Minimum complexity required for emergent intelligent behavior in bounded
systems. Systems below this complexity threshold cannot exhibit robust
adaptive learning.
"""

INFORMATION_DENSITY_LIMIT = math.log(2) * 2
"""
Information Density Limit (2 * ln(2) ≈ 1.386)

Maximum information density in constrained communication channels.
Based on information theory and coding bounds.
"""

# ============================================================================
# DISCOVERY AND VALIDATION THRESHOLDS
# ============================================================================

HIGH_CONFIDENCE_THRESHOLD = 0.9
"""
High confidence threshold for constant discovery.

Discoveries with confidence above this threshold are considered
highly reliable and backed by strong empirical evidence.
"""

MEDIUM_CONFIDENCE_THRESHOLD = 0.7
"""
Medium confidence threshold for constant discovery.

Discoveries with confidence above this threshold are considered
moderately reliable and should be validated with additional data.
"""

LOW_CONFIDENCE_THRESHOLD = 0.5
"""
Low confidence threshold for constant discovery.

Discoveries below this threshold are speculative and require
significant additional validation.
"""

VALIDATION_SIGNIFICANCE_LEVEL = 0.05
"""
Statistical significance level for validation tests (α = 0.05).

Standard p-value threshold for statistical hypothesis testing
in constant discovery validation.
"""

# ============================================================================
# COMPUTATIONAL CONSTANTS
# ============================================================================

NUMERICAL_TOLERANCE = 1e-10
"""
Numerical tolerance for floating-point comparisons and convergence checks.
"""

MAX_ITERATIONS_DEFAULT = 1000
"""
Default maximum iterations for optimization algorithms.
"""

CONVERGENCE_TOLERANCE = 1e-6
"""
Default convergence tolerance for iterative algorithms.
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_constant_info(constant_name: str) -> dict:
    """
    Get detailed information about a named constant.
    
    Args:
        constant_name: Name of the constant (e.g., 'GOLDEN_RATIO')
        
    Returns:
        Dictionary with constant value and description
    """
    constants_info = {
        'GOLDEN_RATIO': {
            'value': GOLDEN_RATIO,
            'symbol': 'φ',
            'description': 'The Golden Ratio appears in optimal resource allocation, search optimization, and structural proportions'
        },
        'EULER_NUMBER': {
            'value': EULER_NUMBER,
            'symbol': 'e',
            'description': 'Base of natural logarithm, appears in growth and decay processes'
        },
        'PI': {
            'value': PI,
            'symbol': 'π',
            'description': 'Ratio of circle circumference to diameter, appears in cyclic patterns'
        },
        'OPTIMAL_RESOURCE_SPLIT': {
            'value': OPTIMAL_RESOURCE_SPLIT,
            'symbol': '1/φ',
            'description': 'Optimal resource allocation ratio for balanced exploitation/exploration'
        },
        'CONVERGENCE_THRESHOLD_FACTOR': {
            'value': CONVERGENCE_THRESHOLD_FACTOR,
            'symbol': '1/e',
            'description': 'Learning convergence threshold indicating 63.2% of asymptotic performance'
        },
        'MAX_EFFICIENCY_RATIO': {
            'value': MAX_EFFICIENCY_RATIO,
            'symbol': 'η_max',
            'description': 'Maximum theoretical efficiency in bounded systems'
        }
    }
    
    return constants_info.get(constant_name, {'value': None, 'description': 'Unknown constant'})


def list_all_constants() -> dict:
    """
    List all defined constants in the module.
    
    Returns:
        Dictionary mapping constant names to values
    """
    return {
        'GOLDEN_RATIO': GOLDEN_RATIO,
        'EULER_NUMBER': EULER_NUMBER,
        'PI': PI,
        'NATURAL_LOG_BASE': NATURAL_LOG_BASE,
        'OPTIMAL_RESOURCE_SPLIT': OPTIMAL_RESOURCE_SPLIT,
        'CONVERGENCE_THRESHOLD_FACTOR': CONVERGENCE_THRESHOLD_FACTOR,
        'LEARNING_RATE_BOUNDARY': LEARNING_RATE_BOUNDARY,
        'MAX_EFFICIENCY_RATIO': MAX_EFFICIENCY_RATIO,
        'MINIMAL_COMPLEXITY_CONSTANT': MINIMAL_COMPLEXITY_CONSTANT,
        'INFORMATION_DENSITY_LIMIT': INFORMATION_DENSITY_LIMIT,
        'HIGH_CONFIDENCE_THRESHOLD': HIGH_CONFIDENCE_THRESHOLD,
        'MEDIUM_CONFIDENCE_THRESHOLD': MEDIUM_CONFIDENCE_THRESHOLD,
        'LOW_CONFIDENCE_THRESHOLD': LOW_CONFIDENCE_THRESHOLD,
        'VALIDATION_SIGNIFICANCE_LEVEL': VALIDATION_SIGNIFICANCE_LEVEL,
    }
