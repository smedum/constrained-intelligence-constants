
# 🤝 Contributing to Constrained Intelligence Constants

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive environment.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of mathematical optimization and/or machine learning

### Find Something to Work On

1. **Check existing issues**: Browse [open issues](https://github.com/yourusername/constrained-intelligence-constants/issues)
2. **Good first issues**: Look for issues tagged `good-first-issue`
3. **Feature requests**: Check `enhancement` tagged issues
4. **Bugs**: Issues tagged `bug` need fixing
5. **Documentation**: Issues tagged `documentation` need writing

### Or Propose Something New

- Open a [new issue](https://github.com/yourusername/constrained-intelligence-constants/issues/new) to discuss your idea
- Wait for feedback from maintainers before starting major work
- For small fixes, feel free to directly submit a PR

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/constrained-intelligence-constants.git
cd constrained-intelligence-constants
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- The package in editable mode
- Testing tools (pytest, pytest-cov)
- Linting tools (black, flake8, mypy)
- Documentation tools (sphinx)

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

## How to Contribute

### Types of Contributions

#### 🐛 Bug Fixes

1. Check if the bug is already reported
2. If not, open an issue describing the bug
3. Fork, create a branch, and fix the bug
4. Add a test that would have caught the bug
5. Submit a pull request

#### ✨ New Features

1. Open an issue to discuss the feature
2. Wait for approval from maintainers
3. Implement the feature with tests
4. Update documentation
5. Submit a pull request

#### 📚 Documentation

1. Find gaps in documentation
2. Improve clarity, add examples
3. Fix typos, improve wording
4. Submit a pull request

#### 🧪 Tests

1. Identify untested code
2. Write comprehensive tests
3. Improve test coverage
4. Submit a pull request

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Use double quotes for strings
- **Imports**: Organize as: stdlib, third-party, local
- **Type hints**: Use type hints for all public functions

### Code Formatting

We use `black` for automatic formatting:

```bash
black constrained_intelligence/
```

### Linting

Run linters before committing:

```bash
# Style checking
flake8 constrained_intelligence/

# Type checking
mypy constrained_intelligence/
```

### Example Code Structure

```python
"""
Module docstring describing purpose.
"""

import math
from typing import List, Dict, Optional

import numpy as np
from scipy import stats

from .constants import GOLDEN_RATIO


class MyClass:
    """
    Class docstring with description.
    
    Attributes:
        param1: Description of param1
        param2: Description of param2
    """
    
    def __init__(self, param1: float, param2: str):
        """
        Initialize MyClass.
        
        Args:
            param1: Description of param1
            param2: Description of param2
        """
        self.param1 = param1
        self.param2 = param2
    
    def my_method(self, arg: int) -> List[float]:
        """
        Method docstring with description.
        
        Args:
            arg: Description of arg
            
        Returns:
            Description of return value
            
        Raises:
            ValueError: When arg is negative
        """
        if arg < 0:
            raise ValueError("arg must be non-negative")
        
        return [float(i) for i in range(arg)]
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=constrained_intelligence tests/

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::test_golden_ratio_optimization
```

### Writing Tests

1. **Location**: Place tests in `tests/` directory
2. **Naming**: Test files: `test_*.py`, Test functions: `test_*`
3. **Structure**: Arrange-Act-Assert pattern
4. **Coverage**: Aim for >80% code coverage

#### Example Test

```python
import pytest
from constrained_intelligence import OptimizationEngine


def test_golden_ratio_optimization():
    """Test golden ratio optimization finds correct minimum."""
    # Arrange
    optimizer = OptimizationEngine(constraints={})
    objective = lambda x: (x - 5.0) ** 2
    bounds = (0.0, 10.0)
    
    # Act
    result = optimizer.golden_ratio_optimization(
        objective_function=objective,
        bounds=bounds,
        max_iterations=100
    )
    
    # Assert
    assert abs(result['optimal_x'] - 5.0) < 0.001
    assert result['converged'] is True
    assert result['iterations'] <= 100


def test_optimization_with_invalid_bounds():
    """Test that invalid bounds raise ValueError."""
    optimizer = OptimizationEngine(constraints={})
    objective = lambda x: x ** 2
    
    with pytest.raises(ValueError):
        optimizer.golden_ratio_optimization(objective, (10, 0))  # Invalid: low > high
```

### Test Categories

1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test component interactions
3. **Validation Tests**: Verify against known mathematical results
4. **Performance Tests**: Ensure algorithms meet performance requirements

## Documentation

### Docstring Format

We use **Google-style** docstrings:

```python
def function(arg1: int, arg2: str) -> bool:
    """
    Short one-line description.
    
    Longer description with more details about what the function does,
    how it works, and any important notes.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When something is wrong
        
    Example:
        >>> function(42, "hello")
        True
    """
    pass
```

### Documentation Types

1. **API Documentation**: Docstrings for all public functions/classes
2. **User Guides**: Tutorials and how-tos in markdown
3. **Theory Documentation**: Mathematical foundations
4. **Examples**: Working code examples

### Building Documentation

```bash
cd docs/
make html
# Open docs/_build/html/index.html
```

## Pull Request Process

### Before Submitting

1. ✅ Code follows style guidelines
2. ✅ All tests pass
3. ✅ New tests added for new functionality
4. ✅ Documentation updated
5. ✅ CHANGELOG.md updated
6. ✅ Commits are clear and descriptive

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add spectral analysis method to discovery module

- Implement FFT-based periodicity detection
- Add validation tests
- Update documentation

Closes #42
```

**Format**:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding/updating tests
- `refactor:` Code restructuring
- `perf:` Performance improvement
- `chore:` Maintenance tasks

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Manually tested

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or documented)

## Related Issues
Closes #issue_number
```

### Review Process

1. **Automated Checks**: CI/CD must pass
2. **Code Review**: At least one maintainer approval required
3. **Discussion**: Address reviewer feedback
4. **Approval**: Maintainer approves and merges

### After Merge

- Your contribution will be included in the next release
- You'll be added to CONTRIBUTORS.md
- Thank you! 🎉

## Development Workflow

### Typical Workflow

```bash
# 1. Update main branch
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes
# ... edit code ...

# 4. Run tests
pytest

# 5. Format code
black constrained_intelligence/

# 6. Commit changes
git add .
git commit -m "feat: add my feature"

# 7. Push to your fork
git push origin feature/my-feature

# 8. Open pull request on GitHub
```

### Keep Your Fork Updated

```bash
# Add upstream remote (once)
git remote add upstream https://github.com/original/constrained-intelligence-constants.git

# Fetch and merge updates
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## Questions?

- **General Questions**: [GitHub Discussions](https://github.com/yourusername/constrained-intelligence-constants/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/yourusername/constrained-intelligence-constants/issues)
- **Security Issues**: Email constrained-intelligence-security@example.com

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Annual contributor highlights

Thank you for contributing to Constrained Intelligence Constants! 🙏

---

**Need help?** Don't hesitate to ask questions in issues or discussions. We're here to help!
