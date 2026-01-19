"""
Tests for the API client.
"""
import pytest
from unittest.mock import patch, Mock

from eegdash_tagger.client.api_client import EEGDashTaggerClient, TaggingResult


class TestEEGDashTaggerClient:
    """Tests for API client."""

    @pytest.fixture
    def client(self):
        return EEGDashTaggerClient("http://localhost:8000")

    def test_health_check(self, client):
        """Test health check."""
        with patch('requests.get') as mock_get:
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
        with patch('requests.post') as mock_post:
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
        with patch('requests.post') as mock_post:
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
        with patch('requests.get') as mock_get:
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
        with patch('requests.get') as mock_get:
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
        with patch('requests.get') as mock_get:
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
        with patch('requests.delete') as mock_delete:
            mock_delete.return_value.raise_for_status = Mock()

            client.clear_cache()

            mock_delete.assert_called_once()

    def test_delete_cache_entry_found(self, client):
        """Test deleting cache entry that exists."""
        with patch('requests.delete') as mock_delete:
            mock_delete.return_value.status_code = 200
            mock_delete.return_value.raise_for_status = Mock()

            result = client.delete_cache_entry("ds001:hash:config:model")

            assert result == True

    def test_delete_cache_entry_not_found(self, client):
        """Test deleting cache entry that doesn't exist."""
        with patch('requests.delete') as mock_delete:
            mock_delete.return_value.status_code = 404

            result = client.delete_cache_entry("nonexistent:key")

            assert result == False
