# Python Development Environment Setup with Ruff

## 1. Install Dependencies

### Using pip:
```bash
pip install pytest>=7.0 pytest-cov>=3.0 pytest-timeout>=2.1 pytest-mock>=3.6 \
            ruff>=0.1.0 mypy>=0.950
```

### Using requirements-dev.txt:
```bash
pip install -r requirements-dev.txt
```

## 2. What Ruff Replaces

**Ruff replaces:**
- Black (formatting)
- isort (import sorting)
- Flake8 (linting)
- pyupgrade, autoflake, and more

**Keep:**
- pytest (testing framework)
- pytest-cov, pytest-timeout, pytest-mock (pytest plugins)
- mypy (for static type checking - Ruff doesn't do full type checking yet)

## 3. Configuration Files

### pyproject.toml (Single config file)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = [
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing",
    "-v"
]
# Timeout for tests (in seconds) - requires pytest-timeout plugin
timeout = 300
timeout_method = "thread"  # or "signal" on Unix systems

[tool.ruff]
line-length = 88
target-version = "py38"

# Exclude directories
exclude = [
    ".git",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    "venv",
]

[tool.ruff.lint]
# Enable specific rule sets
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]

# Ignore specific rules
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.format]
# Use Black-compatible formatting
quote-style = "double"
indent-style = "space"

[tool.ruff.lint.isort]
known-first-party = ["src"]

[tool.mypy]
python_version = "3.9"  # Use 3.9+ for built-in generic types (list[str] instead of List[str])
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = false

# If you must use Python 3.8 or earlier, set python_version = "3.8" 
# and import from typing: from typing import List, Dict, Set, Tuple
```

## 4. Pre-commit Hooks (Optional but Recommended)

### Install pre-commit:
```bash
pip install pre-commit
```

### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      # Run the linter
      - id: ruff
        args: [--fix]
      # Run the formatter
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Install hooks:
```bash
pre-commit install
```

## 5. Common Commands

### Format and fix code:
```bash
# Format code (replaces black)
ruff format .

# Lint and auto-fix issues (replaces flake8 + isort)
ruff check . --fix

# Just check without fixing
ruff check .
```

### Type checking:
```bash
mypy src/
```

### Run tests:
```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_example.py

# With verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_function_name"
```

## 6. Project Structure
```
project/
├── src/
│   └── your_package/
│       ├── __init__.py
│       └── module.py
├── tests/
│   ├── __init__.py
│   └── test_module.py
├── .vscode/
│   ├── settings.json
│   ├── extensions.json
│   └── launch.json
├── pyproject.toml
├── .pre-commit-config.yaml
├── requirements.txt
└── requirements-dev.txt
```

## 7. requirements-dev.txt
```
pytest>=7.0
pytest-cov>=3.0
pytest-timeout>=2.1
pytest-mock>=3.6
ruff>=0.1.0
mypy>=0.950
pre-commit>=3.0
```

## 8. VSCode Setup

### Recommended Extensions:
- **Python** (ms-python.python) - Base Python support
- **Pylance** (ms-python.vscode-pylance) - Fast language server
- **Ruff** (charliermarsh.ruff) - Linting and formatting
- **Mypy Type Checker** (ms-python.mypy-type-checker) - Type checking
- **Test Explorer** (littlefoxteam.vscode-python-test-adapter) - Visual test runner

### .vscode/settings.json
```json
{
  // Python
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  
  // Formatting with Ruff
  "editor.formatOnSave": true,
  "editor.formatOnPaste": false,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  
  // Ruff configuration
  "ruff.lint.args": [],
  "ruff.format.args": [],
  
  // Mypy
  "mypy-type-checker.args": [
    "--config-file=pyproject.toml"
  ],
  
  // Testing
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": [
    "tests",
    "-v",
    "--cov=src"
  ],
  
  // Editor
  "editor.rulers": [88],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    "htmlcov": true,
    ".coverage": true,
    "*.egg-info": true,
    ".ruff_cache": true
  }
}
```

### .vscode/extensions.json
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "charliermarsh.ruff",
    "ms-python.mypy-type-checker",
    "littlefoxteam.vscode-python-test-adapter"
  ]
}
```

### .vscode/launch.json
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal"
    },
    {
      "name": "Python: pytest",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": [
        "${file}",
        "-v"
      ],
      "console": "integratedTerminal"
    }
  ]
}
```

## 9. Makefile (Optional convenience)
```makefile
.PHONY: format lint test clean all

format:
	ruff format .

lint:
	ruff check . --fix
	mypy src/

check:
	ruff check .
	mypy src/

test:
	pytest --cov=src --cov-report=html --cov-report=term-missing

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: format lint test
```

## 10. Quick Start
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements-dev.txt

# 3. Set up pre-commit (optional)
pre-commit install

# 4. In VSCode: Install recommended extensions
# Press Ctrl+Shift+P, type "Python: Select Interpreter"
# Choose the venv/bin/python interpreter

# 5. Run all checks
make all  # or run commands individually
```