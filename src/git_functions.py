import os
import random
import string
import subprocess


def get_git_branch():
    github_ref_name = os.getenv("GITHUB_HEAD_REF")
    if github_ref_name:
        return github_ref_name

    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
        return branch
    except subprocess.CalledProcessError:
        return "unknown-branch"


def generate_short_hash():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
