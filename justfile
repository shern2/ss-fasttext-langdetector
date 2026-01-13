# ==========================================
# GLOBAL CONFIGURATION
# ==========================================

# UI Colors (ANSI escape codes)
RED    := `tput setaf 1 2>/dev/null || true`
GREEN  := `tput setaf 2 2>/dev/null || true`
YELLOW := `tput setaf 3 2>/dev/null || true`
CYAN   := `tput setaf 6 2>/dev/null || true`
GRAY   := `tput setaf 8 2>/dev/null || true`
RESET  := `tput sgr0 2>/dev/null || true`

# --- Environment ---
env := "dev"

# --- Recipes ---
# List all available commands
default:
    @just --list

# ==========================================
# 🐣 Setup
# ==========================================

# Setup the development environment
[group('Dependency Management / Setup')]
setup: (_check_dependency "uv")
    #!/usr/bin/env bash
    set -euo pipefail

    echo "🐣 {{YELLOW}}Setting up development environment...{{RESET}}"
    just install

    if command -v pre-commit >/dev/null 2>&1; then
        echo "🔗 {{YELLOW}}Setting up pre-commit hooks...{{RESET}}"
        pre-commit install --install-hooks
    else
        echo "⚠️ {{YELLOW}}Warning: pre-commit not found, skipping hook installation{{RESET}}"
    fi

    echo ""
    echo "💡 {{CYAN}}Note: Ensure you have a .env file if your tests require environment variables.{{RESET}}"

# Install dependencies from uv.lock
[group('Dependency Management / Setup')]
install: (_check_dependency "uv")
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🚀 {{YELLOW}}Syncing dependencies...{{RESET}}"
    uv sync --all-extras --dev


# ==========================================
# 🧪 Testing
# ==========================================

# Run tests with a specific python version (e.g., just test 3.12)
[group('Testing')]
test version="3.12" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🧪 {{YELLOW}}Running tests on Python {{version}}...{{RESET}}"
    [ -f .env ] && set -a && source .env && set +a || true
    uv run --python {{version}} pytest {{args}}

# Run only unit tests
[group('Testing')]
test-unit *args:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔗 {{YELLOW}}Running unit tests...{{RESET}}"
    [ -f .env ] && set -a && source .env && set +a || true
    ${UV_RUNNER:-uv run} pytest tests/unit/ {{args}}

# Run unit tests specifically for AI agents
[group('Testing')]
agent-test-unit *args:
    # note: pointing to '.venv/bin/python -m' is necessary for AI agents to work in Cursor sandbox
    @UV_RUNNER=".venv/bin/python -m" just test-unit {{args}}

# Run integration tests
[group('Testing')]
test-integration *args:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔗 {{YELLOW}}Running integration tests...{{RESET}}"
    [ -f .env ] && set -a && source .env && set +a || true
    ${UV_RUNNER:-uv run} pytest tests/integration/ {{args}}

# Run integration tests specifically for AI agents
[group('Testing')]
agent-test-integration *args:
    # note: pointing to '.venv/bin/python -m' is necessary for AI agents to work in Cursor sandbox
    @UV_RUNNER=".venv/bin/python -m" just test-integration {{args}}


# ==========================================
# 🐍 Utilities
# ==========================================

# Clean up cache files
[group('Utils')]
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🗑️ {{YELLOW}}Cleaning artifacts...{{RESET}}"
    rm -rf .ruff_cache .pytest_cache .mypy_cache .coverage htmlcov dist
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true


# ==========================================
# 🚀 Publishing
# ==========================================

# Publish to PyPI (e.g., just publish v0.0.2)
[group('Publishing')]
publish version="":
    #!/usr/bin/env bash
    set -euo pipefail

    VERSION="{{version}}"
    if [ -z "$VERSION" ]; then
        VERSION=$(uv version | awk '{print $NF}')
    else
        echo "🏷️ {{YELLOW}}Setting version to $VERSION...{{RESET}}"
        uv version "$VERSION"
    fi

    echo "📦 {{YELLOW}}Publishing $VERSION to PyPI...{{RESET}}"
    rm -rf dist/*
    uv build
    uv publish

# Bump version and tag (e.g., just bump patch)
[group('Publishing')]
bump bump="patch" verify="true":
    #!/usr/bin/env bash
    set -euo pipefail

    if [[ "{{verify}}" == "true" ]]; then
        if [[ -n $(git status --porcelain) ]]; then
            echo "❌ {{RED}}Error: Working directory is not clean. Commit changes first.{{RESET}}"
            echo "   (Use 'just bump {{bump}} false' to bypass this check)"
            exit 1
        fi
    else
        echo "⚠️  {{YELLOW}}Warning: Skipping git status verification (--no-verify){{RESET}}"
    fi

    echo "📈 {{YELLOW}}Bumping {{bump}} version...{{RESET}}"
    uv version --bump {{bump}}
    NEW_VERSION=$(uv version | awk '{print $NF}')

    echo "🏷️ {{YELLOW}}Tagging v$NEW_VERSION...{{RESET}}"
    git add pyproject.toml uv.lock
    git commit -m "chore(release): bump version to $NEW_VERSION"
    git tag "v$NEW_VERSION"
    git push origin "v$NEW_VERSION"
    git push origin $(git branch --show-current)


# ==========================================
# 🔒 Internal Helpers
# ==========================================

[private]
_check_dependency bin:
    @command -v {{bin}} >/dev/null 2>&1 || (echo "❌ {{RED}}Error: {{bin}} is not installed{{RESET}}" && exit 1)
