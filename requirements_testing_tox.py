import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "unix-requirement.txt;sys_platform=='linux'"]
)

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "requirement.txt;sys_platform!='linux'"]
)

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"]
)
