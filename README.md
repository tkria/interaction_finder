# interaction-finder

Automated discovery of biological associations from scientific literature using AI-driven pipelines.

![](screenshot.png)

## Installation

Requires Python 3.13+. Run directly with [uv](https://docs.astral.sh/uv/):

```bash
uvx --from "git+https://github.com/tecosaur/interaction_finder.git" interaction-finder --help
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

## The browser UI

The quickest way in is the browser UI. Launch it locally and it opens in your default browser:

```bash
uv run interaction-finder ui
```

Start a run from a topic and set of entity kinds, watch progress (and any warnings, such as a PubMed rate limit) stream in live, then read the result as an extraction report: each entity pair carries a trust badge, the source text with mentions and evidence quotes highlighted, and the reasoning behind every quality assessment. You can also open an existing checkpoint, edit the config, and set API keys without leaving the page.

## Command-line usage

The pipeline has three stages: **keywords** (find bridging terms from reviews), **search** (query literature databases), and **extract** (identify entity pairs with evidence). Running a later stage automatically executes preceding ones.

Extract gene-disease associations for a research topic:

```bash
uv run interaction-finder extract "Inhibitors of IL-23" \
  -e antibody -e target -o results.json
```

The `-e` flag specifies entity types to extract (e.g., `gene`, `disease`, `phenotype`, `drug`). Pairs are formed between the specified types.

Generate an HTML report:

```bash
uv run interaction-finder report results.json
```

## Quick Test Run

For a faster test with minimal API calls, limit search rounds and results:

```bash
uv run interaction-finder extract "Inhibitors of IL-23" \
  -e antibody -e target -o results.json \
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

[agents.extraction.pair_judge]
llm = "openai:gpt-4o"
```

Override via CLI with `-O key=value`. See `uv run interaction-finder config schema` for all options.
