"""Pytest configuration for root-level tests (base pseudo-module).

Base tests run when:
1. TEST_ALL is set, OR
2. Any base-level file (src/interaction_finder/*.py) is tainted by changes
   from anywhere in the codebase (base or submodules)
"""

from tests.conftest_helpers import skip_unless_tainted

pytest_collection_modifyitems = skip_unless_tainted("base", is_base=True)
