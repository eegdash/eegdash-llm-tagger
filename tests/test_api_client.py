"""
Tests for the API client.
"""
import pytest
from unittest.mock import patch, Mock

from eegdash_tagger.client.api_client import EEGDashTaggerClient, TaggingResult


class TestEEGDashTaggerClient:
    """Tests for API client."""

    @pytest.fixture
    def client(self, monkeypatch):
        # A key exported in the developer's shell must not leak into tests.
        monkeypatch.delenv("TAGGER_API_KEY", raising=False)
        return EEGDashTaggerClient("http://localhost:8000")

    def test_health_check(self, client):
        """Test health check."""
        with patch('requests.Session.get') as mock_get:
            mock_get.return_value.json.return_value = {
                "status": "healthy",
                "cache_entries": 5,
                "config_hash": "abc123"
            }
            mock_get.return_value.raise_for_status = Mock()

            result = client.health_check()

            assert result["status"] == "healthy"
            mock_get.assert_called_once()

    def test_tag_dataset(self, client):
        """Test tagging a dataset."""
        with patch('requests.Session.post') as mock_post:
            mock_post.return_value.json.return_value = {
                "dataset_id": "ds001234",
                "pathology": ["Healthy"],
                "modality": ["Visual"],
                "type": ["Perception"],
                "confidence": {"pathology": 0.9},
                "reasoning": {},
                "from_cache": False
            }
            mock_post.return_value.raise_for_status = Mock()

            result = client.tag_dataset("ds001234", "https://github.com/test/ds001234")

            assert isinstance(result, TaggingResult)
            assert result.dataset_id == "ds001234"
            assert result.pathology == ["Healthy"]
            assert result.from_cache == False

    def test_tag_dataset_with_force_refresh(self, client):
        """Test tagging with force refresh."""
        with patch('requests.Session.post') as mock_post:
            mock_post.return_value.json.return_value = {
                "dataset_id": "ds001234",
                "pathology": ["Healthy"],
                "modality": ["Visual"],
                "type": ["Perception"],
                "confidence": {},
                "reasoning": {},
                "from_cache": False
            }
            mock_post.return_value.raise_for_status = Mock()

            client.tag_dataset("ds001234", "https://github.com/test/ds001234", force_refresh=True)

            # Verify force_refresh was sent
            call_args = mock_post.call_args
            assert call_args[1]["json"]["force_refresh"] == True

    def test_get_cached_tags_found(self, client):
        """Test getting cached tags when they exist."""
        with patch('requests.Session.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "dataset_id": "ds001234",
                "pathology": ["Healthy"],
                "modality": ["Visual"],
                "type": ["Perception"],
                "confidence": {},
                "reasoning": {},
                "from_cache": True
            }
            mock_get.return_value.raise_for_status = Mock()

            result = client.get_cached_tags("ds001234")

            assert result is not None
            assert result.from_cache == True

    def test_get_cached_tags_not_found(self, client):
        """Test getting cached tags when they don't exist."""
        with patch('requests.Session.get') as mock_get:
            mock_get.return_value.status_code = 404

            result = client.get_cached_tags("ds999999")

            assert result is None

    def test_batch_tag_datasets(self, client):
        """Test batch tagging multiple datasets."""
        with patch.object(client, 'get_cached_tags') as mock_cached, \
             patch.object(client, 'tag_dataset') as mock_tag:

            # First dataset is cached
            mock_cached.side_effect = [
                TaggingResult(
                    dataset_id="ds001",
                    pathology=["Healthy"],
                    modality=["Visual"],
                    type=["Perception"],
                    confidence={},
                    reasoning={},
                    from_cache=True
                ),
                None  # Second not cached
            ]

            mock_tag.return_value = TaggingResult(
                dataset_id="ds002",
                pathology=["Epilepsy"],
                modality=["Resting State"],
                type=["Clinical/Intervention"],
                confidence={},
                reasoning={},
                from_cache=False
            )

            datasets = [
                {"dataset_id": "ds001", "source_url": "https://github.com/test/ds001"},
                {"dataset_id": "ds002", "source_url": "https://github.com/test/ds002"}
            ]

            results = client.batch_tag_datasets(datasets, skip_cached=True)

            assert len(results) == 2
            assert results[0].dataset_id == "ds001"
            assert results[0].from_cache == True
            assert results[1].dataset_id == "ds002"

    def test_batch_tag_handles_errors(self, client):
        """Test batch tagging handles errors gracefully."""
        with patch.object(client, 'get_cached_tags', return_value=None), \
             patch.object(client, 'tag_dataset') as mock_tag:

            mock_tag.side_effect = Exception("API error")

            datasets = [
                {"dataset_id": "ds001", "source_url": "https://github.com/test/ds001"}
            ]

            results = client.batch_tag_datasets(datasets)

            assert len(results) == 1
            assert results[0].pathology == ["Unknown"]
            assert results[0].error == "API error"

    def test_get_cache_stats(self, client):
        """Test getting cache stats."""
        with patch('requests.Session.get') as mock_get:
            mock_get.return_value.json.return_value = {
                "total_entries": 10,
                "config_hash": "abc123",
                "unique_datasets": 5,
                "datasets": ["ds001", "ds002"]
            }
            mock_get.return_value.raise_for_status = Mock()

            stats = client.get_cache_stats()

            assert stats["total_entries"] == 10
            assert stats["unique_datasets"] == 5

    def test_clear_cache(self, client):
        """Test clearing cache."""
        with patch('requests.Session.delete') as mock_delete:
            mock_delete.return_value.raise_for_status = Mock()

            client.clear_cache()

            mock_delete.assert_called_once()

    def test_delete_cache_entry_found(self, client):
        """Test deleting cache entry that exists."""
        with patch('requests.Session.delete') as mock_delete:
            mock_delete.return_value.status_code = 200
            mock_delete.return_value.raise_for_status = Mock()

            result = client.delete_cache_entry("ds001:hash:config:model")

            assert result == True

    def test_delete_cache_entry_not_found(self, client):
        """Test deleting cache entry that doesn't exist."""
        with patch('requests.Session.delete') as mock_delete:
            mock_delete.return_value.status_code = 404

            result = client.delete_cache_entry("nonexistent:key")

            assert result == False


@pytest.mark.parametrize("method,patch_target,call", [
    ("tag_dataset", "requests.Session.post", lambda c: c.tag_dataset("ds1", "https://github.com/OpenNeuroDatasets/ds1")),
    ("clear_cache", "requests.Session.delete", lambda c: c.clear_cache()),
    ("delete_cache_entry", "requests.Session.delete", lambda c: c.delete_cache_entry("k")),
])
@pytest.mark.parametrize("key,expected", [("k123", {"X-API-Key": "k123"}), (None, {})])
def test_gated_routes_send_api_key(monkeypatch, method, patch_target, call, key, expected):
    """tag/cache-delete routes are gated server-side; the key must go with them."""
    monkeypatch.delenv("TAGGER_API_KEY", raising=False)
    client = EEGDashTaggerClient("http://localhost:8000", api_key=key)
    with patch(patch_target) as mock_req:
        mock_req.return_value.status_code = 200
        mock_req.return_value.json.return_value = {
            "dataset_id": "ds1", "pathology": [], "modality": [], "type": [], "from_cache": False,
        }
        call(client)
    assert mock_req.called
    sent = {k: v for k, v in client._session.headers.items() if k == "X-API-Key"}
    assert sent == expected


def test_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("TAGGER_API_KEY", "from-env")
    assert EEGDashTaggerClient("http://x")._session.headers["X-API-Key"] == "from-env"


def test_key_is_sent_on_the_wire_for_reads_too(monkeypatch):
    """The Caddy edge gates GETs as well, so the key must be on every request."""
    import requests as _requests
    monkeypatch.delenv("TAGGER_API_KEY", raising=False)
    client = EEGDashTaggerClient("http://localhost:8000", api_key="k9")
    seen = {}

    def fake_send(self, request, **kwargs):
        seen["key"] = request.headers.get("X-API-Key")
        resp = _requests.Response()
        resp.status_code = 200
        resp._content = b'{"status": "healthy"}'
        return resp

    with patch.object(_requests.Session, "send", fake_send):
        client.health_check()
    assert seen["key"] == "k9"


@pytest.mark.parametrize("status", [401, 403, 503])
def test_batch_stops_on_auth_errors(monkeypatch, status):
    import requests as _requests
    monkeypatch.delenv("TAGGER_API_KEY", raising=False)
    client = EEGDashTaggerClient("http://localhost:8000")
    resp = _requests.Response()
    resp.status_code = status
    err = _requests.HTTPError(response=resp)
    with patch.object(EEGDashTaggerClient, "tag_dataset", side_effect=err) as tag:
        with pytest.raises(_requests.HTTPError):
            client.batch_tag_datasets(
                [{"dataset_id": "a", "source_url": "u"}, {"dataset_id": "b", "source_url": "u"}],
                skip_cached=False, verbose=False,
            )
    assert tag.call_count == 1
