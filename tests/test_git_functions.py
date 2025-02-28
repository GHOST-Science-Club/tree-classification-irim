import subprocess
import string
import pytest
from src.git_functions import get_git_branch, generate_short_hash


@pytest.mark.git_functions
def test_get_git_branch_env_variable(monkeypatch):
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature-branch")
    
    assert get_git_branch() == "feature-branch", "Invalid branch name"


@pytest.mark.git_functions
def test_get_git_branch_git_command(monkeypatch):
    monkeypatch.delenv("GITHUB_HEAD_REF", raising=False)
    
    monkeypatch.setattr(subprocess, "check_output", lambda e: b"main")
    
    assert get_git_branch() == "main", "Invalid branch name"


@pytest.mark.git_functions
def test_get_git_branch_git_command_failure(monkeypatch):
    monkeypatch.delenv("GITHUB_HEAD_REF", raising=False)
    
    def raise_(ex):
        raise ex
    monkeypatch.setattr(subprocess, "check_output", lambda e: raise_(subprocess.CalledProcessError(1, "git")))

    assert get_git_branch() == "unknown-branch", "Invalid branch name"


@pytest.mark.git_functions
def test_generate_short_hash():
    hash_value = generate_short_hash()
    
    assert isinstance(hash_value, str), "Run hash is not a string"
    assert len(hash_value) == 6, "Run hash has incorrect length"
    assert all(char in string.ascii_lowercase + string.digits for char in hash_value), "Run hash has incorrect elements"
