# **Comprehensive Guide to Running Tests**  

> This README extends the **[main project README](https://github.com/GHOST-Science-Club/tree-classification-irim/blob/main/README.md)**.  
> For a quick test guide, refer to that file. *Familiarity with its content is required here.*  

---

## Pytest

By default, `pytest` uses CLI options from `pyproject.toml`, but you can override them by passing custom arguments (see [pytest docs](https://docs.pytest.org/en/stable/how-to/usage.html) for details).  

### **Running Tests with Markers**  
Pytest allows *custom markers* to run a **subset of tests**. Available markers and descriptions are listed in [`pyproject.toml`](https://github.com/GHOST-Science-Club/tree-classification-irim/blob/main/pyproject.toml).  

To run tests under specific markers:  
```bash
pytest -m [marker1] [marker2]
```

### **Running Specific Tests**  
Run tests from a specific file:  
```bash
pytest tests/foo.py tests/bar.py
```

Run specific tests using `-k`:  
```bash
pytest -k 'test_001 or test_some_other_test'
```

### **Additional Pytest Commands**
- **Run tests with detailed output:**  
  ```bash
  pytest -v  # Verbose mode, default in the config file
  ```
- **Generate a test coverage report:**  
  ```bash
  pytest --cov=.--cov-report=html # Default in the config file
  ```
- **Run only failed tests from the last session:**  
  ```bash
  pytest --lf
  ```

See also: [Working with custom markers](https://docs.pytest.org/en/stable/example/markers.html).  

---

## Tox

Tox runs tests in isolated environments and uses `tox.ini` for configuration, which can be **partially overridden** (see [tox CLI docs](https://tox.wiki/en/4.24.2/cli_interface.html)).  

### **Running a Specific Environment**  
Run tests in a specific environment:  
```bash
tox -e [environment_name]
```

### **Useful Tox Commands**
- **List available test environments:**  
  ```bash
  tox -l
  ```
- **Recreate environments (force a clean state):**  
  ```bash
  tox --recreate
  ```
- **Pass custom arguments to pytest within tox:**  
  ```bash
  tox -- -k 'test_example'
  ```
