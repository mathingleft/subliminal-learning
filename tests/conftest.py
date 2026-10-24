import subprocess

import pytest


@pytest.fixture
def clean_git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.com")
    (repo / "README").write_text("test repo\n")
    git("add", "README")
    git("commit", "-q", "-m", "init")
    return repo
