"""Static guards for cloud image size, process safety, and release automation."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_api_image_forces_cpu_torch_and_one_worker():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "download.pytorch.org/whl/cpu" in dockerfile
    assert "requirements-api.txt" in dockerfile
    assert "--workers 1" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "requirements.txt" not in dockerfile


def test_api_runtime_requirements_exclude_non_runtime_tooling():
    requirements = (ROOT / "requirements-api.txt").read_text(encoding="utf-8").lower()

    for excluded in ("pytest", "ruff", "ragas", "datasets", "streamlit"):
        assert excluded not in requirements


def test_frontend_image_does_not_install_model_dependencies():
    dockerfile = (ROOT / "Dockerfile.frontend").read_text(encoding="utf-8").lower()

    assert "sentence-transformers" not in dockerfile
    assert "pip install torch" not in dockerfile
    assert "user 10001:10001" in dockerfile
    assert "scripts.container_healthcheck" in dockerfile
    assert "curl" not in dockerfile


def test_release_workflow_builds_both_images_after_contract_tests():
    path = ROOT / ".github" / "workflows" / "release.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))

    jobs = workflow["jobs"]
    assert jobs["images"]["needs"] == "release-contract"
    components = {
        item["component"] for item in jobs["images"]["strategy"]["matrix"]["include"]
    }
    assert components == {"api", "frontend"}
    assert jobs["smoke"]["needs"] == "images"


def test_production_environment_template_contains_no_real_secret():
    template = (ROOT / "deploy" / "environment.example").read_text(encoding="utf-8")

    assert "RAGUARD_ENVIRONMENT=production" in template
    assert "HF_HOME=/models" in template
    assert "replace-" in template


def test_render_blueprint_has_the_required_remote_resources():
    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    services = {service["name"]: service for service in blueprint["services"]}

    api = services["raguard-api-fahim03"]
    frontend = services["raguard-ui-fahim03"]
    admission = services["raguard-admission-fahim03"]

    assert api["plan"] == "pro plus"
    assert api["numInstances"] == 1
    assert api["disk"] == {
        "name": "raguard-model-cache",
        "mountPath": "/models",
        "sizeGB": 5,
    }
    assert api["healthCheckPath"] == "/health"
    assert frontend["dockerfilePath"] == "./Dockerfile.frontend"
    assert admission["type"] == "keyvalue"
    assert admission["ipAllowList"] == []


def test_render_blueprint_prompts_for_secrets_instead_of_committing_them():
    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    api = next(service for service in blueprint["services"] if service["name"].startswith("raguard-api"))
    env = {item["key"]: item for item in api["envVars"]}

    assert env["DATABASE_URL"] == {"key": "DATABASE_URL", "sync": False}
    assert env["GOOGLE_API_KEY"] == {"key": "GOOGLE_API_KEY", "sync": False}
    assert env["ADMIN_API_KEY"]["generateValue"] is True
    assert "value" not in env["DATABASE_URL"]
