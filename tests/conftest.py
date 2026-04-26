from __future__ import annotations

import os
from pathlib import Path


_TORCH_REQUIRED = {
    "test_mbu_netpp_prepare_nasa_super.py",
    "test_mbu_netpp_semi_supervised.py",
    "test_mbu_netpp_suite_runner.py",
    "test_mbu_netpp_train_resume.py",
}


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # type: ignore[no-untyped-def]
    if os.environ.get("RUN_TORCH_TESTS") == "1":
        return False
    return collection_path.name in _TORCH_REQUIRED
