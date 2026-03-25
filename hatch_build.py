"""
Hatchling build hook: compiles fasttext from source and vendors the output
(fasttext Python sources + fasttext_pybind .so) into the package.

This makes ss-fasttext-langdetector a self-contained native platform wheel —
consumers get a pre-compiled binary from PyPI with no source compilation needed.
"""

import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

FASTTEXT_REPO = "https://github.com/facebookresearch/fastText"
FASTTEXT_REV = "1142dc4c4ecbc19cc16eee5cdd28472e689267e6"

PKG_DIR = Path(__file__).parent / "src" / "ss_fasttext_langdetector"
VENDOR_DIR = PKG_DIR / "_vendor"


class CustomBuildHook(BuildHookInterface):
    """Compiles fasttext from source and vendors the output into the package."""

    PLUGIN_NAME = "custom"

    def initialize(self, _version: str, build_data: dict) -> None:
        """Compile fasttext and populate force_include with vendored files.

        Skipped for editable installs (local dev) — fasttext is provided there
        via the git source in [tool.uv.sources] + [dependency-groups].
        """
        if _version == "editable":
            return

        if not (VENDOR_DIR / "fasttext").exists():
            self._compile_and_vendor()

        # Force-include everything in _vendor/ into the wheel
        for path in VENDOR_DIR.rglob("*"):
            if path.is_file():
                wheel_path = str(
                    Path("ss_fasttext_langdetector") / path.relative_to(PKG_DIR)
                )
                build_data["force_include"][str(path)] = wheel_path

        # Set explicit platform wheel tag — pure_python=False alone is unreliable
        # because hatchling may compute the tag during the metadata phase before
        # our hook fires. Setting 'tag' directly overrides whatever hatchling computed.
        # auditwheel repair will re-tag the Linux wheel to manylinux_2_28_x86_64.
        build_data["pure_python"] = False
        python_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        build_data["tag"] = f"{python_ver}-{python_ver}-{plat}"

    def _compile_and_vendor(self) -> None:
        VENDOR_DIR.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            ft_dir = Path(tmpdir) / "fastText"

            self.app.display_info(f"[build hook] Cloning fastText @ {FASTTEXT_REV[:12]}...")
            subprocess.run(
                ["git", "clone", "--no-checkout", "--depth=50", FASTTEXT_REPO, str(ft_dir)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "checkout", FASTTEXT_REV],
                check=True,
                cwd=ft_dir,
                capture_output=True,
            )

            self.app.display_info("[build hook] Compiling fasttext_pybind...")
            subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                check=True,
                cwd=ft_dir,
            )

            # Vendor the fasttext Python wrapper sources
            # The repo layout at this rev: python/fasttext_module/fasttext/
            ft_py_src = ft_dir / "python" / "fasttext_module" / "fasttext"
            shutil.copytree(ft_py_src, VENDOR_DIR / "fasttext", dirs_exist_ok=True)

            # Vendor the compiled pybind extension
            # setup.py copies the .so into python/fasttext_module/ (inplace)
            so_files = list((ft_dir / "python" / "fasttext_module").glob("fasttext_pybind*.so"))
            if not so_files:
                raise RuntimeError("fasttext_pybind.so not found after build — check fasttext setup.py output")
            for so in so_files:
                self.app.display_info(f"[build hook] Vendoring {so.name}")
                shutil.copy2(so, VENDOR_DIR / so.name)

        self.app.display_info("[build hook] fasttext vendored successfully.")

    def clean(self, _versions: list[str]) -> None:
        """Remove the vendored fasttext directory."""
        if VENDOR_DIR.exists():
            shutil.rmtree(VENDOR_DIR)
