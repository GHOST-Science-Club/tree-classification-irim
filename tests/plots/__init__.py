from pathlib import Path
import sys

path = Path().cwd() / "src" / "visualization_functions.py"
print(path)

sys.path.append(str(path))
