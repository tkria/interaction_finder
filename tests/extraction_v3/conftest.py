"""
Shared fixtures for extraction graph V3 integration tests.

Provides known papers dataset, ground truth pairs, and test configuration.
"""

import json
import pytest
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

from interaction_finder.settings import IfetcherConfig


@pytest.fixture
def known_papers_dataset():
    """
    Load known papers with ground truth gene-disease pairs.

    Returns list of dicts with url, disease, gene fields from
    known-papers-pulmonary-arterial-hypertension.jsonl.
    """
    dataset_path = Path("known-papers-pulmonary-arterial-hypertension.jsonl")
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")

    papers = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))

    return papers


@pytest.fixture
def test_paper_urls(known_papers_dataset) -> List[str]:
    """
    Select ~dozen papers marked as '# Found' for testing.

    These papers have clear gene-disease relationships that should be
    extracted with high recall.
    """
    # Select specific papers with known entities
    # (Previously filtered from known_papers_dataset for papers marked "# Found" or "# Almost")
    selected_urls = [
        "https://doi.org/10.1002/humu.20285",  # BMPR2
        "https://doi.org/10.1007/s10038-006-0104-3",  # ACVRL1
        "https://doi.org/10.1016/j.ajhg.2008.05.002",  # COX6B1
        "https://doi.org/10.1016/j.ajhg.2009.05.005",  # FOXF1
        "https://doi.org/10.1016/j.ajhg.2010.12.010",  # SARS2
        "https://doi.org/10.1016/j.ajhg.2011.05.012",  # FBN1
        "https://doi.org/10.1016/j.ajhg.2011.05.017",  # NAA10
        "https://doi.org/10.1002/ajmg.a.62488",  # TBX5
        "https://doi.org/10.1002/humu.23210",  # COX5A
        "https://doi.org/10.1002/humu.24179",  # ALDH1A2
        "https://doi.org/10.1016/j.ajhg.2008.09.013",  # SLC29A3 (bonus)
    ]

    return selected_urls


@pytest.fixture
def ground_truth_pairs() -> List[Tuple[str, str]]:
    """
    Known gene-disease pairs that should be extracted.

    Returns list of (gene, disease) tuples representing ground truth
    relationships from the selected papers.
    """
    return [
        ("BMPR2", "pulmonary arterial hypertension"),
        ("ACVRL1", "pulmonary arterial hypertension"),
        ("COX6B1", "pulmonary arterial hypertension"),
        ("FOXF1", "pulmonary arterial hypertension"),
        ("SARS2", "pulmonary arterial hypertension"),
        ("FBN1", "pulmonary arterial hypertension"),
        ("NAA10", "pulmonary arterial hypertension"),
        ("TBX5", "pulmonary arterial hypertension"),
        ("COX5A", "pulmonary arterial hypertension"),
        ("ALDH1A2", "pulmonary arterial hypertension"),
        ("SLC29A3", "pulmonary arterial hypertension"),
        # Alternative disease names that might be extracted
        ("BMPR2", "PAH"),
        ("ACVRL1", "PAH"),
        ("FOXF1", "PAH"),
    ]


@pytest.fixture
def test_config(tmp_path) -> IfetcherConfig:
    """
    Create test configuration for V3 pipeline.

    Uses gpt-4o-mini for cost efficiency during testing.
    """
    config_dict = {
        "task": {
            "entity_kinds": [
                {"kind_name": "gene", "description": "Gene or protein name"},
                {"kind_name": "disease", "description": "Disease or condition"},
            ],
            "relation": "gene-disease interaction",
            "context": "Biomedical research on pulmonary arterial hypertension",
        },
        "agents": {
            "_": {"llm": "openai:gpt-4o-mini"},  # Use cheap model for tests
        },
        "workflow": {
            "extraction": {
                "parallelism": 2,  # Limit parallelism for testing
            },
            "v3": {
                "semantic_cache_enabled": True,
                "extraction_parallelism": 2,
                "assessment_parallelism": 3,
                "evaluation_parallelism": 5,
                "enable_tier3_candidates": True,
            },
        },
        "paths": {
            "cache_dir": str(tmp_path / "cache"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }

    # Create mock config (simplified for testing)
    config = MagicMock(spec=IfetcherConfig)
    config.task = MagicMock()
    config.task.entity_kinds = [
        MagicMock(kind_name="gene", description="Gene or protein name"),
        MagicMock(kind_name="disease", description="Disease or condition"),
    ]
    config.task.relation = config_dict["task"]["relation"]
    config.task.context = config_dict["task"]["context"]
    config.task.get_kind_names.return_value = ["gene", "disease"]

    config.agents = config_dict["agents"]
    config.workflow = config_dict["workflow"]
    config.paths = config_dict["paths"]

    return config


@pytest.fixture
def bmpr2_paper_urls() -> List[str]:
    """URLs for papers mentioning BMPR2-PAH (for cross-document test)."""
    return [
        "https://doi.org/10.1002/humu.20285",  # BMPR2 paper 1
        "https://doi.org/10.1002/humu.9398",  # BMPR2 paper 2
    ]


@pytest.fixture
def sample_paper_url() -> str:
    """Single paper URL for quick smoke tests."""
    return "https://doi.org/10.1002/humu.20285"  # BMPR2 paper
