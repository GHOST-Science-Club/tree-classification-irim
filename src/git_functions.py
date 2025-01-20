import os
import random
import string
import subprocess


def get_git_branch():
    github_ref = os.getenv('GITHUB_REF')
    if github_ref and github_ref.startswith('refs/heads/'):
        return github_ref.replace('refs/heads/', '')

    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
        return branch
    except subprocess.CalledProcessError:
        return "unknown-branch"


def generate_short_hash():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
