# Error Handling and Security

This guide covers essential exception handling, input validation, and security practices for Python projects.

## Custom Exceptions

### Exception Hierarchy
```python
class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass

class ProcessingError(RuntimeError):
    """Raised when data processing fails."""
    pass

class ConfigurationError(ValueError):
    """Raised when configuration is invalid."""
    pass
```

### Usage Example
```python
class DataProcessor:
    def __init__(self, scale_x: float, scale_y: float, offset_x: float, offset_y: float):
        if scale_x <= 0 or scale_y <= 0:
            raise ValidationError(f"Scale factors must be positive: scale_x={scale_x}, scale_y={scale_y}")
        if offset_x < 0 or offset_y < 0:
            raise ValidationError(f"Offsets must be non-negative: offset_x={offset_x}, offset_y={offset_y}")

        self.scale_x, self.scale_y, self.offset_x, self.offset_y = scale_x, scale_y, offset_x, offset_y

    def transform(self, data: np.ndarray) -> np.ndarray:
        if data.shape[1] != 3:
            raise ValueError(f"Data must have shape (N, 3), got {data.shape}")
        if np.any(data[:, 2] <= 0):
            raise ProcessingError("Invalid data values (third column must be positive)")

        a, b, c = data.T
        return np.column_stack([
            self.scale_x * a / c + self.offset_x,
            self.scale_y * b / c + self.offset_y
        ])
```

## Input Validation

### Common Validation Functions
```python
def validate_array_shape(array: np.ndarray, expected_shape: tuple, name: str) -> None:
    """Validate array shape."""
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")

def validate_positive(values: np.ndarray, name: str) -> None:
    """Validate all values are positive."""
    if np.any(values <= 0):
        raise ValueError(f"{name} must contain positive values")

def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """Validate value is within range."""
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
```

### Validation Example
```python
def validate_transform_params(scale_x: float, scale_y: float, offset_x: float, offset_y: float) -> None:
    """Validate transformation parameters."""
    for param, name in [(scale_x, 'scale_x'), (scale_y, 'scale_y'), (offset_x, 'offset_x'), (offset_y, 'offset_y')]:
        validate_range(param, 0, 10000, name)

    if scale_x <= 0 or scale_y <= 0:
        raise ValidationError("Scale factors must be positive")
```

## Security Best Practices

### Safe File Handling
```python
def load_config(filepath: str) -> dict:
    """Load JSON config with security checks."""
    import os
    if not filepath.endswith('.json') or '..' in filepath:
        raise ValueError("Invalid config file path")

    if os.path.getsize(filepath) > 1024 * 1024:  # 1MB limit
        raise ValueError("Config file too large")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ValueError(f"Failed to load config: {e}")

    # Validate required numeric fields
    required = ['scale_x', 'scale_y', 'offset_x', 'offset_y']
    for field in required:
        if field not in config or not isinstance(config[field], (int, float)):
            raise ValueError(f"Invalid or missing field: {field}")
        if not (0 < config[field] < 10000):
            raise ValueError(f"Field {field} out of valid range")

    return config
```

### Safe Numerical Operations
```python
def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe division handling division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / b
        return np.where(np.isfinite(result), result, 0.0)

def safe_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors safely."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    with np.errstate(invalid='ignore'):
        normalized = vectors / norms
        return np.where(np.isfinite(normalized), normalized, 0.0)

def clamp_values(array: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clamp array values to range."""
    return np.clip(array, min_val, max_val)
```

## Error Recovery Patterns

### Graceful Degradation
```python
class RobustProcessor:
    """Processor with fallback methods."""

    def __init__(self):
        self.methods = ['advanced', 'standard', 'basic']

    def process(self, data: np.ndarray) -> tuple:
        """Process data with automatic fallback."""
        for method in self.methods:
            try:
                processor = self._get_processor(method)
                result, metadata = processor.process(data)
                if result is not None:
                    return result, metadata
            except Exception as e:
                print(f"Warning: {method} processing failed: {e}")
                continue
        raise RuntimeError("All processing methods failed")

    def _get_processor(self, method: str):
        """Get processor instance for method."""
        if method == 'advanced':
            return AdvancedProcessor()
        elif method == 'standard':
            return StandardProcessor()
        elif method == 'basic':
            return BasicProcessor()
```

### Retry Decorator
```python
import time
from functools import wraps

def retry_on_failure(max_attempts: int = 3, delay: float = 0.1, backoff: float = 2.0):
    """Retry decorator for transient failures."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed, retrying in {current_delay:.2f}s: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

class RobustSolver:
    """Solver with retry on convergence failures."""

    @retry_on_failure(max_attempts=3)
    def solve_system(self, initial_guess: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Solve system with retry on failure."""
        if len(data) < 3:
            raise ValueError("Insufficient data for solving")

        # Add perturbation to help convergence
        perturbed = initial_guess + np.random.normal(0, 0.01, initial_guess.shape)
        result = self._solve(perturbed, data)

        if not self._is_valid_solution(result):
            raise RuntimeError("Solution converged to invalid result")

        return result
```

## Logging and Monitoring

### Structured Logging Setup
```python
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)

class LoggedMixin:
    """Mixin that adds logging to operations."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_start(self, operation: str, **params):
        """Log operation start."""
        self.logger.info(f"Starting {operation}", extra={'params': params})

    def log_end(self, operation: str, success: bool, **results):
        """Log operation completion."""
        level = logging.INFO if success else logging.ERROR
        status = "succeeded" if success else "failed"
        self.logger.log(level, f"{operation} {status}", extra={'results': results})

class PerformanceMonitor:
    """Track performance and error metrics."""

    def __init__(self):
        self.total_ops = 0
        self.failed_ops = 0
        self.avg_time = 0.0
        self.recent_errors = []

    def record(self, operation: str, success: bool, exec_time: float, error: str = None):
        """Record operation result."""
        self.total_ops += 1
        if not success:
            self.failed_ops += 1
            self.recent_errors.append({
                'op': operation, 'error': error, 'time': time.time()
            })
            self.recent_errors = self.recent_errors[-10:]  # Keep last 10

        # Update rolling average
        self.avg_time = ((self.avg_time * (self.total_ops - 1)) + exec_time) / self.total_ops

    def error_rate(self) -> float:
        """Get current error rate."""
        return self.failed_ops / self.total_ops if self.total_ops > 0 else 0.0

    def should_alert(self) -> bool:
        """Check if error rate exceeds threshold."""
        return self.error_rate() > 0.1
```

## Testing Error Handling

### Exception Testing Examples
```python
import pytest
import numpy as np
from processor import DataProcessor

class TestProcessorErrors:
    def test_invalid_scale_factors(self):
        """Test invalid scale factors raise errors."""
        with pytest.raises(ValidationError, match="positive"):
            DataProcessor(scale_x=-1.0, scale_y=2.0, offset_x=1.0, offset_y=1.0)

    def test_invalid_offsets(self):
        """Test invalid offsets raise errors."""
        with pytest.raises(ValidationError, match="non-negative"):
            DataProcessor(scale_x=2.0, scale_y=2.0, offset_x=-1.0, offset_y=1.0)

    @pytest.mark.parametrize("bad_input", [
        np.array([[1, 2]]),  # Wrong shape
        "not_array",  # Wrong type
    ])
    def test_invalid_transform_input(self, bad_input):
        """Test invalid transform inputs."""
        processor = DataProcessor(2.0, 2.0, 1.0, 1.0)
        with pytest.raises((ValueError, TypeError)):
            processor.transform(bad_input)

    def test_invalid_data_values(self):
        """Test invalid data values."""
        processor = DataProcessor(2.0, 2.0, 1.0, 1.0)
        data = np.array([[1.0, 1.0, -1.0]])
        with pytest.raises(ProcessingError, match="positive"):
            processor.transform(data)
```

This concise error handling approach ensures robust, safe code with clear validation and graceful failure handling.
