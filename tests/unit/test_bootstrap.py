"""Bootstrap smoke tests; financial-domain tests arrive in later tasks."""

import asyncio
from importlib import import_module
from importlib.metadata import version

import pytest


def test_package_can_be_imported() -> None:
    package = import_module("portfolio_analyzer")
    assert package.__name__ == "portfolio_analyzer"


def test_package_distribution_is_installed() -> None:
    installed_version = version("portfolio-PnL-onchain-analyzer")
    assert installed_version == "0.1.0"


@pytest.mark.asyncio
async def test_pytest_asyncio_runs_async_tests() -> None:
    async def compute() -> int:
        await asyncio.sleep(0)
        return 42

    assert await compute() == 42
