"""Contracts that keep dependency resolution reproducible across every install path."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_project_metadata_is_the_single_dependency_source_of_truth():
    config = _pyproject()

    assert config["project"]["dependencies"]
    assert {"dev", "evaluation", "frontend", "local-models"} <= set(config["dependency-groups"])


def test_local_model_stack_is_optional_for_hosted_embedding_runtimes():
    config = _pyproject()
    runtime = set(config["project"]["dependencies"])
    local_models = set(config["dependency-groups"]["local-models"])

    assert not any(dependency.startswith("torch") for dependency in runtime)
    assert any(dependency.startswith("torch") for dependency in local_models)


def test_torch_is_resolved_from_the_cpu_wheel_index():
    uv = _pyproject()["tool"]["uv"]

    assert uv["sources"]["torch"] == {"index": "pytorch-cpu"}
    cpu_index = next(index for index in uv["index"] if index["name"] == "pytorch-cpu")
    assert cpu_index["url"] == "https://download.pytorch.org/whl/cpu"
    assert cpu_index["explicit"] is True


def test_ragas_compatibility_pair_is_explicitly_pinned():
    evaluation = _pyproject()["dependency-groups"]["evaluation"]

    assert "ragas==0.4.3" in evaluation
    assert "langchain-community==0.4.1" in evaluation


def test_universal_lock_contains_the_compatible_ragas_version():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = {(package["name"], package["version"]) for package in lock["package"]}

    assert ("ragas", "0.4.3") in versions
    assert ("langchain-community", "0.4.1") in versions


def test_universal_lock_contains_no_cuda_runtime_packages():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    names = {package["name"] for package in lock["package"]}

    assert not {name for name in names if name.startswith(("cuda-", "nvidia-"))}


def test_container_builds_install_from_the_frozen_lock():
    for name in ("Dockerfile", "Dockerfile.frontend"):
        dockerfile = (ROOT / name).read_text(encoding="utf-8")
        assert "COPY pyproject.toml uv.lock" in dockerfile
        assert "uv sync --locked" in dockerfile
        assert "pip install" not in dockerfile


def test_automation_uses_the_frozen_lock_instead_of_unbounded_pip_solves():
    paths = [
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / ".github" / "workflows" / "nightly-eval.yml",
        ROOT / ".github" / "workflows" / "heavy-benchmark.yml",
    ]

    for path in paths:
        workflow = path.read_text(encoding="utf-8")
        assert "astral-sh/setup-uv" in workflow, path
        assert "uv sync --locked" in workflow, path
        assert "pip install -r requirements.txt" not in workflow, path


def test_model_backed_workflows_install_the_optional_local_model_stack():
    """A clean CI runner must install the models used by each evaluation job."""
    paths = [
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / ".github" / "workflows" / "nightly-eval.yml",
        ROOT / ".github" / "workflows" / "heavy-benchmark.yml",
    ]

    for path in paths:
        workflow = path.read_text(encoding="utf-8")
        assert "--group local-models" in workflow, (
            f"{path.name} runs local embeddings or reranking but does not install "
            "the local-models dependency group"
        )
