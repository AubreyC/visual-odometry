# Python Style Rule

## PEP 8 Compliance (MANDATORY)
- MUST follow PEP 8 style guide exactly
- MUST use 4 spaces for indentation, never tabs
- MUST limit lines to 88 characters maximum

## Formatting Rules (MANDATORY)
- MUST use double quotes for all strings (never single quotes)
- MUST format all code with Black-compatible Ruff formatter
- MUST sort imports automatically with Ruff
- MUST place one blank line between import groups
- MUST place two blank lines before class definitions
- MUST place one blank line before method definitions

## Type Hints (MANDATORY)
- MUST provide type hints for ALL function parameters
- MUST provide type hints for ALL function return values
- MUST use `None` as return type for procedures that don't return values
- MUST use built-in generic types for Python 3.9+ (e.g., `list[str]`, not `List[str]`)
- MUST use `Optional[T]` for parameters/values that can be None
- MUST type hint complex variables and data structures
- MUST NOT use `Any` unless absolutely necessary
- MUST NOT skip type hints on any function
- MUST NOT use `List[str]` instead of `list[str]` (for Python 3.9+)
- MUST NOT import from `typing` when built-in types exist

## Import Organization (MANDATORY)
MUST organize imports in this exact order:
```python
# 1. Standard library imports (alphabetically)
import math
from typing import List, Optional

# 2. Third-party imports (alphabetically)
import numpy as np

# 3. Local imports (alphabetically)
from .utils import helper_function
```
- MUST NOT use wildcard imports (`from module import *`)

## Documentation (MANDATORY)
- MUST include docstrings for all public functions, classes, and methods
- MUST use triple double quotes for docstrings
- MUST follow Google-style docstring format
- MUST include type information in docstring Args section using format: `param_name (Type): description`
- MUST include type information in Returns section using format: `ReturnType: description`

## CODE GENERATION TEMPLATE

### Function Template:
```python
def function_name(param1: Type1, param2: Optional[Type2] = None) -> ReturnType:
    """Brief description of what function does.

    Args:
        param1 (Type1): Description of param1.
        param2 (Optional[Type2]): Description of param2. Defaults to None.

    Returns:
        ReturnType: Description of return value.
    """
    # Implementation here
    pass
```

### Class Template:
```python
class ClassName:
    """Brief description of class purpose."""

    def __init__(self, param1: Type1, param2: Type2) -> None:
        """Initialize ClassName.

        Args:
            param1 (Type1): Description of param1.
            param2 (Type2): Description of param2.
        """
        self.param1 = param1
        self.param2 = param2

    def method_name(self, param: Type) -> ReturnType:
        """Brief description of method.

        Args:
            param (Type): Description of parameter.

        Returns:
            ReturnType: Description of return value.
        """
        # Implementation here
        pass
```

## CONFIGURATION REFERENCE

AI agents MUST respect the project's `pyproject.toml` configuration:
- Line length: 88
- Target version: Python 3.8+
- Enabled rules: E, W, F, I, N, UP, B, C4, SIM
- Quote style: double
- Indent style: space

## EXCEPTION HANDLING RULES

- MUST use specific exception types, never bare `except`
- MUST include meaningful error messages
- MUST use context managers for resource management
- MUST NOT suppress exceptions without logging
- MUST NOT use bare `except:` clauses

## NAMING CONVENTIONS

| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `calculate_distance()` |
| Variables | snake_case | `user_name` |
| Constants | UPPER_CASE | `MAX_SIZE = 100` |
| Classes | PascalCase | `CameraCalibration` |
| Methods | snake_case | `get_camera_matrix()` |
| Modules | snake_case | `camera_utils.py` |
| Private | Leading underscore | `_internal_method()` |

## NUMERIC LITERALS

- MUST use underscores for large numbers: `1_000_000`
- MUST use scientific notation for very large/small floats: `1.23e6`
- MUST NOT use unnecessary leading zeros: `0.5` not `0.5000`

## STRING FORMATTING

- MUST use f-strings for formatting: `f"Value: {variable}"`
- MUST use `.format()` only when f-strings are insufficient
- MUST NOT use `%` formatting

## BOOLEAN EXPRESSIONS

- MUST use `is` and `is not` for identity comparison
- MUST use `==` and `!=` for value comparison
- MUST write boolean expressions clearly without unnecessary complexity

## GENERAL CODING PRACTICES

- MUST NOT use mutable default arguments
- MUST NOT define multiple classes/functions per file unless logically grouped
