"""Tests for resolving article redistribution licences.

Network access is mocked throughout: these tests pin the classification rules
and the identifier/caching behaviour, not OpenAlex's current answers.
"""

import json

import httpx
import pytest

from interaction_finder.licences import (
    ArticleLicence,
    Verdict,
    resolve_licences,
    summarise,
)

EMAIL = "test@example.org"


def client_returning(payloads: dict[str, dict], calls: list[str] | None = None):
    """httpx.Client answering from `payloads`, keyed by "kind:value" identifier.

    Serves both request shapes the resolver uses: a list request filtered by an
    OR-joined set of identifiers, and a single-entity fetch. `calls` records the
    identifiers asked for, in order, whichever shape carried them.
    """

    def respond(identifier: str) -> dict | None:
        payload = payloads.get(identifier)
        if payload is None:
            return None
        # Echo the identifier back the way OpenAlex does, so the resolver can
        # match a returned work to the identifier it requested.
        kind, _, value = identifier.partition(":")
        echoed = dict(payload)
        if kind == "doi":
            echoed.setdefault("doi", f"https://doi.org/{value}")
        else:
            echoed.setdefault("ids", {kind: value})
        return echoed

    def handler(request: httpx.Request) -> httpx.Response:
        if "filter" in request.url.params:
            expression = request.url.params["filter"]
            kind, _, joined = expression.partition(":")
            identifiers = [f"{kind}:{value}" for value in joined.split("|")]
            if calls is not None:
                calls.extend(identifiers)
            results = [r for r in map(respond, identifiers) if r is not None]
            return httpx.Response(200, json={"results": results})
        # Identifiers contain slashes (doi:10.1/x), so take everything after
        # the /works/ prefix rather than splitting on the last slash.
        identifier = request.url.path.split("/works/", 1)[-1]
        if calls is not None:
            calls.append(identifier)
        payload = respond(identifier)
        if payload is None:
            return httpx.Response(404, json={})
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def work(oa_status=None, licences=()):
    """Minimal OpenAlex work payload with the fields classification reads."""
    return {
        "open_access": {"oa_status": oa_status},
        "locations": [{"license": licence} for licence in licences],
    }


class TestClassification:
    """Which licences permit redistribution, and which do not."""

    @pytest.mark.parametrize("licence", ["cc-by", "cc-by-sa", "cc0", "public-domain"])
    def test_permissive_licences_are_redistributable(self, licence):
        client = client_returning({"doi:10.1/x": work("gold", [licence])})
        result = resolve_licences({"u": {"doi": "10.1/x"}}, email=EMAIL, client=client)
        assert result["u"].verdict is Verdict.REDISTRIBUTABLE
        assert result["u"].licence == licence

    @pytest.mark.parametrize("licence", ["cc-by-nc", "cc-by-nc-nd", "cc-by-nd"])
    def test_conditional_licences_are_share_alike(self, licence):
        client = client_returning({"doi:10.1/x": work("hybrid", [licence])})
        result = resolve_licences({"u": {"doi": "10.1/x"}}, email=EMAIL, client=client)
        assert result["u"].verdict is Verdict.SHARE_ALIKE

    def test_bronze_without_licence_is_restricted(self):
        """Free to read at the publisher is not permission to redistribute."""
        client = client_returning({"doi:10.1/x": work("bronze", [])})
        result = resolve_licences({"u": {"doi": "10.1/x"}}, email=EMAIL, client=client)
        assert result["u"].verdict is Verdict.RESTRICTED
        assert "no open licence" in result["u"].reason

    def test_green_without_licence_is_restricted(self):
        """A repository manuscript copy carries no redistribution grant."""
        client = client_returning({"doi:10.1/x": work("green", [])})
        result = resolve_licences({"u": {"doi": "10.1/x"}}, email=EMAIL, client=client)
        assert result["u"].verdict is Verdict.RESTRICTED

    def test_closed_is_restricted(self):
        client = client_returning({"doi:10.1/x": work("closed", [])})
        result = resolve_licences({"u": {"doi": "10.1/x"}}, email=EMAIL, client=client)
        assert result["u"].verdict is Verdict.RESTRICTED

    def test_unrecognised_licence_is_restricted_not_assumed(self):
        client = client_returning({"doi:10.1/x": work("gold", ["other-oa"])})
        result = resolve_licences({"u": {"doi": "10.1/x"}}, email=EMAIL, client=client)
        assert result["u"].verdict is Verdict.RESTRICTED

    def test_permissive_wins_over_conditional_across_locations(self):
        """A CC-BY copy anywhere permits sharing, whatever other copies say."""
        client = client_returning(
            {"doi:10.1/x": work("hybrid", ["cc-by-nc-nd", "cc-by"])}
        )
        result = resolve_licences({"u": {"doi": "10.1/x"}}, email=EMAIL, client=client)
        assert result["u"].verdict is Verdict.REDISTRIBUTABLE


class TestIdentifiers:
    """How an article is located in OpenAlex."""

    def test_doi_is_preferred(self):
        calls: list[str] = []
        client = client_returning({"doi:10.1/x": work("gold", ["cc-by"])}, calls)
        resolve_licences(
            {"https://pubmed.ncbi.nlm.nih.gov/12345/": {"doi": "10.1/x"}},
            email=EMAIL,
            client=client,
        )
        assert calls == ["doi:10.1/x"]

    def test_works_without_a_contact_address(self):
        """OpenAlex serves anonymous callers; no email should be required."""
        client = client_returning({"doi:10.1/x": work("gold", ["cc-by"])})
        result = resolve_licences({"u": {"doi": "10.1/x"}}, client=client)
        assert result["u"].verdict is Verdict.REDISTRIBUTABLE

    def test_pmid_is_read_from_the_url_when_no_doi(self):
        calls: list[str] = []
        client = client_returning({"pmid:12345": work("gold", ["cc-by"])}, calls)
        result = resolve_licences(
            {"https://pubmed.ncbi.nlm.nih.gov/12345/": {}}, email=EMAIL, client=client
        )
        assert calls == ["pmid:12345"]
        assert result["https://pubmed.ncbi.nlm.nih.gov/12345/"].verdict is (
            Verdict.REDISTRIBUTABLE
        )

    def test_unidentifiable_article_is_unknown(self):
        client = client_returning({})
        result = resolve_licences({"file:///local.pdf": {}}, email=EMAIL, client=client)
        assert result["file:///local.pdf"].verdict is Verdict.UNKNOWN
        assert "no DOI or PMID" in result["file:///local.pdf"].reason

    def test_failed_lookup_is_unknown_not_restricted(self):
        """A lookup failure must be distinguishable from a licence refusal."""
        client = client_returning({})  # every request 404s
        result = resolve_licences(
            {"u": {"doi": "10.1/missing"}}, email=EMAIL, client=client
        )
        assert result["u"].verdict is Verdict.UNKNOWN

    def test_every_input_gets_a_verdict(self):
        client = client_returning({"doi:10.1/a": work("gold", ["cc-by"])})
        articles = {"a": {"doi": "10.1/a"}, "b": {"doi": "10.1/b"}, "c": {}}
        result = resolve_licences(articles, email=EMAIL, client=client)
        assert set(result) == set(articles)


class TestCaching:
    """Repeated runs over one corpus should not re-query."""

    def test_cache_is_written_and_reused(self, tmp_path):
        cache = tmp_path / "licences.json"
        calls: list[str] = []
        payloads = {"doi:10.1/x": work("gold", ["cc-by"])}
        resolve_licences(
            {"u": {"doi": "10.1/x"}},
            email=EMAIL,
            cache_path=cache,
            client=client_returning(payloads, calls),
        )
        assert cache.exists()
        resolve_licences(
            {"u": {"doi": "10.1/x"}},
            email=EMAIL,
            cache_path=cache,
            client=client_returning(payloads, calls),
        )
        assert calls == ["doi:10.1/x"], "second run should hit the cache"

    def test_corrupt_cache_is_ignored(self, tmp_path):
        cache = tmp_path / "licences.json"
        cache.write_text("{not json")
        client = client_returning({"doi:10.1/x": work("gold", ["cc-by"])})
        result = resolve_licences(
            {"u": {"doi": "10.1/x"}}, email=EMAIL, cache_path=cache, client=client
        )
        assert result["u"].verdict is Verdict.REDISTRIBUTABLE
        assert json.loads(cache.read_text())  # rewritten cleanly


class TestSummarise:
    def test_counts_every_verdict_kind(self):
        counts = summarise(
            [
                ArticleLicence(Verdict.REDISTRIBUTABLE, "cc-by", "gold", ""),
                ArticleLicence(Verdict.RESTRICTED, None, "bronze", ""),
                ArticleLicence(Verdict.RESTRICTED, None, "closed", ""),
            ]
        )
        assert counts["redistributable"] == 1
        assert counts["restricted"] == 2
        assert counts["unknown"] == 0


class TestBatching:
    """Many articles resolve in few requests, without spending the allowance."""

    def batch_client(self, works: dict[str, dict], calls: list[httpx.Request]):
        """Client answering list+filter requests from `works` (keyed by DOI)."""

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if "/works/" in request.url.path:  # single-entity fallback
                identifier = request.url.path.split("/works/", 1)[-1]
                doi = identifier.removeprefix("doi:")
                if doi in works:
                    return httpx.Response(200, json=works[doi])
                return httpx.Response(404, json={})
            asked = request.url.params.get("filter", "").removeprefix("doi:")
            results = [works[d] for d in asked.split("|") if d in works]
            return httpx.Response(
                200,
                json={"results": results},
                headers={"x-ratelimit-remaining": "900"},
            )

        return httpx.Client(transport=httpx.MockTransport(handler))

    def works_for(self, dois: list[str]) -> dict[str, dict]:
        return {
            doi: {
                "doi": f"https://doi.org/{doi}",
                "open_access": {"oa_status": "gold"},
                "locations": [{"license": "cc-by"}],
            }
            for doi in dois
        }

    def test_many_articles_resolve_in_one_request(self):
        dois = [f"10.1/{n}" for n in range(60)]
        calls: list[httpx.Request] = []
        client = self.batch_client(self.works_for(dois), calls)
        result = resolve_licences(
            {f"u{n}": {"doi": doi} for n, doi in enumerate(dois)}, client=client
        )
        assert len(calls) == 1, "60 articles should take one list request"
        assert all(v.verdict is Verdict.REDISTRIBUTABLE for v in result.values())

    def test_batches_are_split_at_the_size_limit(self):
        from interaction_finder.licences import _BATCH_SIZE

        dois = [f"10.1/{n}" for n in range(_BATCH_SIZE + 5)]
        calls: list[httpx.Request] = []
        client = self.batch_client(self.works_for(dois), calls)
        resolve_licences(
            {f"u{n}": {"doi": doi} for n, doi in enumerate(dois)}, client=client
        )
        assert len(calls) == 2

    def test_article_missing_from_a_batch_falls_back_to_single_fetch(self):
        """Single-entity fetches are unbilled, so stragglers are retried singly."""
        dois = ["10.1/a", "10.1/b"]
        works = self.works_for(["10.1/a"])  # b absent from list results
        works["10.1/b"] = {
            "doi": "https://doi.org/10.1/b",
            "open_access": {"oa_status": "bronze"},
            "locations": [],
        }
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if "/works/" in request.url.path:
                return httpx.Response(200, json=works["10.1/b"])
            return httpx.Response(200, json={"results": [works["10.1/a"]]})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        result = resolve_licences(
            {"ua": {"doi": "10.1/a"}, "ub": {"doi": "10.1/b"}}, client=client
        )
        assert len(calls) == 2, "one list request, then one single fetch"
        assert result["ua"].verdict is Verdict.REDISTRIBUTABLE
        assert result["ub"].verdict is Verdict.RESTRICTED

    def test_cached_articles_are_not_refetched(self, tmp_path):
        dois = [f"10.1/{n}" for n in range(5)]
        cache = tmp_path / "lic.json"
        articles = {f"u{n}": {"doi": doi} for n, doi in enumerate(dois)}
        first: list[httpx.Request] = []
        resolve_licences(
            articles,
            cache_path=cache,
            client=self.batch_client(self.works_for(dois), first),
        )
        second: list[httpx.Request] = []
        resolve_licences(
            articles,
            cache_path=cache,
            client=self.batch_client(self.works_for(dois), second),
        )
        assert first and not second

    def test_pmids_and_dois_are_requested_separately(self):
        """Identifier kinds cannot share one filter, so each gets its own."""
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"results": []})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        resolve_licences(
            {
                "a": {"doi": "10.1/a"},
                "b": {"url": "https://pubmed.ncbi.nlm.nih.gov/999/"},
            },
            client=client,
        )
        filters = [
            c.url.params.get("filter") for c in calls if "filter" in c.url.params
        ]
        assert any(f.startswith("doi:") for f in filters)
        assert any(f.startswith("pmid:") for f in filters)


class TestErrorsAreNotCachedPermanently:
    """A transient failure must not pin an article to UNKNOWN forever."""

    def test_failed_lookup_is_retried_on_the_next_run(self, tmp_path):
        cache = tmp_path / "lic.json"
        work = {
            "doi": "https://doi.org/10.1/x",
            "open_access": {"oa_status": "gold"},
            "locations": [{"license": "cc-by"}],
        }
        failing = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(500, json={}))
        )
        first = resolve_licences(
            {"u": {"doi": "10.1/x"}}, cache_path=cache, client=failing
        )
        assert first["u"].verdict is Verdict.UNKNOWN

        calls: list[httpx.Request] = []

        def healthy(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"results": [work]})

        second = resolve_licences(
            {"u": {"doi": "10.1/x"}},
            cache_path=cache,
            client=httpx.Client(transport=httpx.MockTransport(healthy)),
        )
        assert calls, "a cached error must not suppress the retry"
        assert second["u"].verdict is Verdict.REDISTRIBUTABLE
