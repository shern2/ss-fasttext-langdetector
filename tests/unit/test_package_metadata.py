import tomllib
from pathlib import Path


def test_numpy_is_declared_as_runtime_dependency():
    """The vendored FastText package imports NumPy at runtime."""
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]
    dependency_names = {dependency.split(">", 1)[0].split("<", 1)[0].split("=", 1)[0] for dependency in project["dependencies"]}
    assert "numpy" in dependency_names
