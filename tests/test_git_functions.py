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
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda e: raise_(subprocess.CalledProcessError(1, "git"))
    )

    assert get_git_branch() == "unknown-branch", "Invalid branch name"


@pytest.mark.git_functions
def test_generate_short_hash():
    hash_value = generate_short_hash()

    allow_char = string.ascii_lowercase + string.digits

    error_msg = {
        "not-str": "Run hash is not a string",
        "bad-length": "Run hash has incorrect length",
        "bad-elm": "Run hash has incorrect elements"
    }

    assert isinstance(hash_value, str), error_msg["not-str"]
    assert len(hash_value) == 6, error_msg["bad-length"]
    assert all(char in allow_char for char in hash_value), error_msg["bad-elm"]
