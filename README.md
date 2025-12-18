# interaction-finder

Automated discovery of biological associations from scientific literature using AI-driven pipelines.

## Installation

Requires Python 3.13+. Run directly with [uv](https://docs.astral.sh/uv/) (requires SSH access to the repository):

```bash
uvx --from "git+ssh://git@github.com/tecosaur/interaction_finder.git" interaction-finder --help
```

Or clone and install locally:

```bash
git clone git@github.com:tecosaur/interaction_finder.git
cd interaction-finder
uv sync
```

Set your OpenAI API key (or other LLM provider):

```bash
export OPENAI_API_KEY=your-key
```

## Usage

The pipeline has three stages: **keywords** (find bridging terms from reviews), **search** (query literature databases), and **extract** (identify entity pairs with evidence). Running a later stage automatically executes preceding ones.

Extract gene-disease associations for a research topic:

```bash
uv run interaction-finder extract "diabetes genetics" \
  -e gene -e disease -o results.json
```

The `-e` flag specifies entity types to extract (e.g., `gene`, `disease`, `phenotype`, `drug`). Pairs are formed between the specified types.

Generate an HTML report:

```bash
uv run interaction-finder report results.json
```

## Quick Test Run

For a faster test with minimal API calls, limit search rounds and results:

```bash
uv run interaction-finder extract "diabetes genetics" \
  -e gene -e disease -o results.json \
  -O stage.search.max_rounds=1 \
  -O stage.search.results_per_query=2
```

## Search Backends

- `pubmed` — NCBI PubMed (default, best for biomedical literature)
- `perplexica` — Local Perplexica instance (broader web search)
- `openai` — OpenAI web search API

```bash
uv run interaction-finder extract "topic" -e gene -b perplexica -o results.json
```

## Configuration

Create a `config.toml` to customize LLM models:

```toml
[agents._]
llm = "openai:gpt-4o-mini"

[agents.extraction.judge]
llm = "openai:gpt-4o"
```

Override via CLI with `-O key=value`. See `uv run interaction-finder config schema` for all options.
