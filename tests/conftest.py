"""Tests marked needs_data use the downloaded FPL / odds / fixture files in data/, which aren't in
git (255 MB). They're skipped when data/ is missing, e.g. in CI; locally test.py runs everything."""
import os

import pytest

HAVE_DATA = os.path.exists("data/boot.json") and os.path.exists("data/gw_2024-25.csv")


def pytest_collection_modifyitems(config, items):
    if HAVE_DATA:
        return
    skip = pytest.mark.skip(reason="needs downloaded data/ (run ./run.sh once)")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)
