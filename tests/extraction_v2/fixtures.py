"""
Test fixtures for extraction graph V2 testing.

Known documents with verified gene-disease relationships
for testing ResourceQuote integration.
"""

from typing import List
from interaction_finder.resources import ResourcePool


def create_test_resource_pool(documents: List[tuple[str, str, str]]) -> ResourcePool:
    """
    Create test resource pool from document tuples.

    Args:
        documents: List of (url, title, content) tuples

    Returns:
        ResourcePool with loaded documents
    """
    pool = ResourcePool()
    for url, title, content in documents:
        pool.add(url, title, content)
    return pool


# Test document with known BRCA1-breast cancer relationship
BRCA1_DOCUMENT = """
BRCA1 mutations significantly increase breast cancer risk.

The BRCA1 gene plays a crucial role in DNA repair mechanisms. Studies have shown
that BRCA1 mutations cause breast cancer in approximately 60-80% of carriers.
Women with BRCA1 mutations have a substantially higher lifetime risk of developing
breast cancer compared to the general population.

Recent research indicates that BRCA1 deficiency leads to genomic instability,
which is a key factor in breast cancer development. The protein encoded by BRCA1
is essential for homologous recombination repair of DNA double-strand breaks.

Clinical trials have demonstrated that BRCA1 testing should be recommended for
women with a family history of breast cancer. Early detection through genetic
screening can help identify individuals at high risk for breast cancer.
"""

# Expected entities in BRCA1_DOCUMENT
BRCA1_EXPECTED_ENTITIES = {
    "BRCA1": {
        "kind": "gene",
        "occurrences": 7,  # Number of times BRCA1 appears
        "contexts": [
            "BRCA1 mutations significantly increase breast cancer risk",
            "The BRCA1 gene plays a crucial role in DNA repair",
            "BRCA1 mutations cause breast cancer",
            "Women with BRCA1 mutations have a substantially higher",
            "BRCA1 deficiency leads to genomic instability",
            "The protein encoded by BRCA1 is essential",
            "BRCA1 testing should be recommended",
        ],
    },
    "breast cancer": {
        "kind": "disease",
        "occurrences": 4,
        "contexts": [
            "BRCA1 mutations significantly increase breast cancer risk",
            "BRCA1 mutations cause breast cancer in approximately",
            "higher lifetime risk of developing breast cancer compared",
            "family history of breast cancer",
        ],
    },
}

# Expected relationship evidence
BRCA1_RELATIONSHIP_EVIDENCE = [
    "BRCA1 mutations significantly increase breast cancer risk",
    "BRCA1 mutations cause breast cancer in approximately 60-80% of carriers",
    "individuals at high risk for breast cancer",
]

# Test document with multiple gene-disease relationships
MULTI_GENE_DOCUMENT = """
TP53 mutations are associated with Li-Fraumeni syndrome and various cancers.
The TP53 gene encodes the p53 protein, which is crucial for preventing cancer.

BRCA2 mutations also increase breast cancer risk, similar to BRCA1.
Both BRCA1 and BRCA2 are involved in homologous recombination repair.

KRAS mutations are frequently found in pancreatic cancer and lung cancer.
The KRAS oncogene promotes cell proliferation when mutated.

APC mutations are the primary cause of familial adenomatous polyposis (FAP).
Patients with APC mutations have a very high risk of colorectal cancer.
"""

# Expected entities in MULTI_GENE_DOCUMENT
MULTI_GENE_EXPECTED = {
    "genes": ["TP53", "BRCA2", "BRCA1", "KRAS", "APC"],
    "diseases": [
        "Li-Fraumeni syndrome",
        "cancer",
        "breast cancer",
        "pancreatic cancer",
        "lung cancer",
        "colorectal cancer",
        "familial adenomatous polyposis",
    ],
    "relationships": [
        ("TP53", "Li-Fraumeni syndrome"),
        ("TP53", "cancer"),
        ("BRCA2", "breast cancer"),
        ("BRCA1", "breast cancer"),
        ("KRAS", "pancreatic cancer"),
        ("KRAS", "lung cancer"),
        ("APC", "colorectal cancer"),
        ("APC", "familial adenomatous polyposis"),
    ],
}

# Document with no clear relationships (negative test case)
NO_RELATIONSHIP_DOCUMENT = """
The weather today is sunny and warm. Scientists are studying climate change
patterns around the world. 

Gene expression varies across different tissues and developmental stages.
Proteins play important roles in cellular processes and metabolic pathways.

Research methodologies continue to evolve with new technologies.
Data analysis requires careful attention to statistical significance.
"""

# Document with entity mentions but no clear relationships
WEAK_RELATIONSHIP_DOCUMENT = """
BRCA1 is a gene located on chromosome 17. The human genome contains
approximately 20,000-25,000 protein-coding genes.

Breast cancer is a type of malignancy that affects breast tissue.
Cancer occurs when cells grow and divide uncontrollably.

Many factors can influence health outcomes, including genetics,
environment, and lifestyle choices.
"""

# Test URLs for ResourcePool
TEST_URLS = [
    "https://test.example.com/brca1_paper.html",
    "https://test.example.com/multi_gene_study.html",
    "https://test.example.com/negative_control.html",
]
