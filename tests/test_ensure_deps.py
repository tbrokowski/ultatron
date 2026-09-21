"""Dependency bootstrap checks that never install packages or require a GPU."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import ensure_deps


class EnsureDepsTests(unittest.TestCase):
    def test_installer_preserves_container_versions(self):
        versions = {"torch": "2.11.0a0+nv", "torchvision": "0.25.0a0", "numpy": "1.26.4"}
        constraint_paths = []

        def check_install(command):
            self.assertIn("transformers>=4.56.0,<5.0", command)
            path = Path(command[command.index("--constraint") + 1])
            constraint_paths.append(path)
            self.assertEqual(
                set(path.read_text().splitlines()),
                {f"{name}=={value}" for name, value in versions.items()},
            )

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(ensure_deps, "REPO_ROOT", Path(directory)), \
                patch.object(ensure_deps, "version", side_effect=versions.__getitem__), \
                patch.object(ensure_deps.subprocess, "check_call", side_effect=check_install) as install:
            ensure_deps._pip_install(["transformers>=4.56.0,<5.0"])
        install.assert_called_once()
        self.assertFalse(constraint_paths[0].exists())

    def test_ready_environment_does_not_install(self):
        with patch.object(ensure_deps, "missing_imports", return_value=[]), \
                patch.object(ensure_deps, "_pip_install") as install, \
                patch.object(ensure_deps, "check_torch_numpy") as check:
            ensure_deps.ensure_deps()
        install.assert_not_called()
        check.assert_called_once()

    def test_installs_only_missing_packages(self):
        with patch.object(ensure_deps, "missing_imports", side_effect=[["yaml"], []]), \
                patch.object(ensure_deps, "_pip_install") as install, \
                patch.object(ensure_deps, "check_torch_numpy"):
            ensure_deps.ensure_deps()
        install.assert_called_once_with(["pyyaml>=6.0"])
