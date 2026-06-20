"""Pytest configuration for root-level tests (base pseudo-module).

Base tests run when:
1. TEST_ALL is set, OR
2. Any base-level file (src/interaction_finder/*.py) is tainted by changes
   from anywhere in the codebase (base or submodules)
"""

import os

import pytest

from tests.conftest_helpers import skip_unless_tainted

pytest_collection_modifyitems = skip_unless_tainted("base", is_base=True)


@pytest.fixture(autouse=True)
def _dummy_openai_key():
    """Ensure a placeholder OPENAI_API_KEY so agents construct under TestModel.

    Building a pydantic-ai Agent instantiates a provider client, which raises
    without a key -- even when the test overrides the model with TestModel and
    never makes a real call. A real key (CI secret, local dev) is left intact.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-test-placeholder"
    yield
