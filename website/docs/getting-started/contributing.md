---
sidebar_position: 5
---

# Contributing

Thank you for your interest in contributing to Amigo! We welcome contributions from the community.

## Ways to Contribute

### Report Bugs

Found a bug? Please [create an issue](https://github.com/your-org/amigo/issues/new) with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your environment (OS, Python version, Amigo version)
- Minimal code example demonstrating the issue

### Suggest Enhancements

Have ideas for new features? [Start a discussion](https://github.com/your-org/amigo/discussions) or create an issue describing:

- The problem you're trying to solve
- Your proposed solution
- Alternative approaches you've considered
- How this would benefit other users

### Improve Documentation

Documentation improvements are always welcome:

- Fix typos or clarify explanations
- Add examples or tutorials
- Improve API documentation
- Translate documentation

### Contribute Code

Ready to contribute code? Great! Follow the development workflow below.

## Development Workflow

### 1. Set Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/your-username/amigo.git
cd amigo

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Make Changes

- Write clear, documented code
- Follow the coding style (PEP 8 for Python)
- Add tests for new functionality
- Update documentation as needed

### 4. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_component.py

# Run with coverage
pytest --cov=amigo tests/
```

### 5. Commit Changes

```bash
git add .
git commit -m "Add: brief description of changes"
```

Use clear commit messages:
- `Add: new feature`
- `Fix: bug description`
- `Docs: documentation update`
- `Test: test additions/modifications`
- `Refactor: code restructuring`

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots/examples if applicable

## Coding Standards

### Python Style

Follow [PEP 8](https://pep8.org/) guidelines:

```python
# Good
class MyComponent(am.Component):
    """Component docstring explaining purpose."""
    
    def __init__(self):
        super().__init__()
        self.add_input("x", value=0.0)
    
    def compute(self):
        """Compute outputs from inputs."""
        x = self.inputs["x"]
        self.outputs["y"] = 2 * x
```

### Documentation

Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/):

```python
def example_function(param1, param2):
    """
    Brief description of function.
    
    Longer description if needed, explaining behavior
    and important details.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
    
    Returns
    -------
    type
        Description of return value
    
    Examples
    --------
    >>> example_function(1, 2)
    3
    """
    return param1 + param2
```

### Testing

Write tests for all new functionality:

```python
import pytest
import amigo as am

def test_component_creation():
    """Test that components can be created."""
    comp = am.Component()
    assert comp is not None

def test_input_addition():
    """Test adding inputs to components."""
    comp = am.Component()
    comp.add_input("x", value=1.0)
    assert "x" in comp.inputs
```

## Code Review Process

1. **Automated Checks**: CI/CD runs tests and linters
2. **Maintainer Review**: Core maintainers review code
3. **Feedback**: Address review comments
4. **Approval**: Once approved, code is merged

## Git Commit Guidelines

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Build process or auxiliary tool changes

### Example

```
feat: Add support for vector-valued constraints

Implemented vector constraint handling in Component class.
Constraints can now be added with shape parameter to define
multi-dimensional constraint vectors.

Closes #123
```

## Community Guidelines

### Code of Conduct

Be respectful, inclusive, and constructive:

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/your-org/amigo/discussions)
- **Bugs**: Create an [issue](https://github.com/your-org/amigo/issues)
- **Chat**: Join our community chat (if available)

## Recognition

Contributors are recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Documentation credits

Thank you for contributing to Amigo! 🎉

