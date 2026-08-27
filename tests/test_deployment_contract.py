"""Static guards for cloud image size, process safety, and release automation."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml
from yaml.constructor import ConstructorError

ROOT = Path(__file__).resolve().parents[1]


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects keys GitHub would reject."""


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_contract_yaml(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)


def test_deployment_yaml_contains_no_duplicate_keys():
    paths = [ROOT / "render.yaml", *(ROOT / ".github" / "workflows").glob("*.yml")]

    for path in paths:
        assert load_contract_yaml(path) is not None, path


def test_api_image_forces_cpu_torch_and_one_worker():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cpu"}
    assert "uv sync --locked --no-default-groups" in dockerfile
    assert "--workers 1" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "requirements" not in dockerfile


def test_api_runtime_requirements_exclude_non_runtime_tooling():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = "\n".join(project["project"]["dependencies"]).lower()

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
    workflow = load_contract_yaml(path)

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
    blueprint = load_contract_yaml(ROOT / "render.yaml")
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
    assert "maxShutdownDelaySeconds" not in api
    assert api["healthCheckPath"] == "/health"
    assert frontend["dockerfilePath"] == "./Dockerfile.frontend"
    assert admission["type"] == "keyvalue"
    assert admission["ipAllowList"] == []


def test_render_blueprint_prompts_for_secrets_instead_of_committing_them():
    blueprint = load_contract_yaml(ROOT / "render.yaml")
    api = next(service for service in blueprint["services"] if service["name"].startswith("raguard-api"))
    env = {item["key"]: item for item in api["envVars"]}

    assert env["DATABASE_URL"] == {"key": "DATABASE_URL", "sync": False}
    assert env["GOOGLE_API_KEY"] == {"key": "GOOGLE_API_KEY", "sync": False}
    assert env["ADMIN_API_KEY"]["generateValue"] is True
    assert "value" not in env["DATABASE_URL"]
