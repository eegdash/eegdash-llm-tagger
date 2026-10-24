"""
OpenRouter.ai LLM-based tagger for EEG datasets.

This module provides an LLM-powered implementation of the Tagger protocol
using OpenRouter.ai API to classify datasets with GPT-4/Claude models.
"""

import json
import math
from importlib.resources import files
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

from .taxonomy import normalize_labels
from .provenance import request_provenance

from .tagger import Tagger, ParsedMetadata, TaggingResult


class TaggingError(RuntimeError):
    """The model request failed or did not return a usable tagging record.

    This is distinct from a valid model abstention (``Unknown`` labels). Callers
    must not cache or persist a failed call as a completed classification.
    """


class OpenRouterTagger:
    """
    LLM-based tagger using OpenRouter.ai API.

    This tagger uses few-shot learning with labeled examples to classify
    datasets according to the structured reasoning framework in prompt.md.

    Attributes:
        api_key: OpenRouter.ai API key
        model: Model identifier (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
        verbose: If True, print progress and debug information
        few_shot_examples: Cached labeled examples for in-context learning
    """

    # Any OpenAI-compatible chat endpoint works (OpenRouter, a local proxy,
    # ...): set LLM_API_BASE, e.g. http://127.0.0.1:8317/v1.
    ENDPOINT = (
        os.getenv("LLM_API_BASE", "https://openrouter.ai/api/v1").rstrip("/")
        + "/chat/completions"
    )

    # Metadata fields relevant for classification (whitelist)
    # These are the only fields sent to the LLM to reduce noise and tokens
    RELEVANT_METADATA_KEYS = {
        "title",
        "dataset_description",
        "readme",
        "participants_overview",
        "tasks",
        "events",
        "task_details",
        "paper_abstract",
    }

    # Fields to include from few-shot examples (labels + relevant metadata)
    FEW_SHOT_KEYS = {
        "pathology",
        "modality",
        "type",
        "metadata",  # Will be filtered by RELEVANT_METADATA_KEYS
    }

    @classmethod
    def _default_resource_path(cls, filename: str) -> Path:
        # Pip installs wheels unpacked. Source/editable checkouts retain the
        # canonical prompt/example files used as wheel build inputs.
        bundled = files("eegdash_tagger").joinpath("resources", filename)
        if bundled.is_file():
            return Path(str(bundled))
        root = Path(__file__).resolve().parents[3]
        return root / ("data/processed" if filename.endswith(".json") else "") / filename

    @classmethod
    def get_default_few_shot_path(cls) -> Path:
        """Return the bundled examples, or canonical source file in a checkout."""
        return cls._default_resource_path("few_shot_examples.json")

    @classmethod
    def get_default_prompt_path(cls) -> Path:
        """Return the bundled prompt, or canonical source file in a checkout."""
        return cls._default_resource_path("prompt.md")

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-5.2",
        verbose: bool = False,
        few_shot_path: Optional[Path] = None,
        prompt_path: Optional[Path] = None,
        max_tokens: int = 4000
    ):
        """
        Initialize OpenRouterTagger.

        Args:
            api_key: OpenRouter.ai API key. If None, reads from OPENROUTER_API_KEY env var
            model: Model identifier (default: "openai/gpt-5.2")
            verbose: Enable verbose output
            few_shot_path: Path to few_shot_examples.json. If None, uses package default
            prompt_path: Path to prompt.md. If None, uses package default
            max_tokens: Maximum tokens for LLM response (default: 4000)

        Raises:
            ValueError: If API key is not provided and OPENROUTER_API_KEY env var is not set
            FileNotFoundError: If few-shot examples or prompt file cannot be found
        """
        self.api_key = api_key or os.getenv('LLM_API_KEY') or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.model = model
        self.verbose = verbose
        self.max_tokens = max_tokens
        base_url = os.getenv("LLM_API_BASE")
        self.endpoint = base_url.rstrip("/") + "/chat/completions" if base_url else self.ENDPOINT

        # Store prompt path for later use
        self.prompt_path = Path(prompt_path) if prompt_path is not None else self.get_default_prompt_path()
        if not self.prompt_path.is_file():
            raise FileNotFoundError(f"Prompt not found at: {self.prompt_path}")

        # Load few-shot examples
        if few_shot_path is None:
            few_shot_path = self.get_default_few_shot_path()

        few_shot_path = Path(few_shot_path)
        if not few_shot_path.is_file():
            raise FileNotFoundError(f"Few-shot examples not found at: {few_shot_path}")

        with open(few_shot_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Extract the array from the file structure
            self.few_shot_examples = data['few_shot_examples']

        # Filter few-shot examples to only include relevant fields
        self.few_shot_examples = [
            self._filter_few_shot_example(ex) for ex in self.few_shot_examples
        ]

        if self.verbose:
            print(f"Loaded {len(self.few_shot_examples)} few-shot examples")
            print(f"Using model: {self.model}")
            print(f"Prompt path: {self.prompt_path}")
            print(f"Max tokens: {self.max_tokens}")

    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter metadata to only include relevant fields for classification.

        Args:
            metadata: Full metadata dictionary

        Returns:
            Filtered metadata with only classification-relevant fields
        """
        return {
            k: v for k, v in metadata.items()
            if k in self.RELEVANT_METADATA_KEYS and v
        }

    def _filter_few_shot_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter a few-shot example to only include relevant fields.

        Args:
            example: Full few-shot example

        Returns:
            Filtered example with labels and relevant metadata only
        """
        filtered = {}

        # Include label fields
        for key in ["pathology", "modality", "type"]:
            if key in example:
                filtered[key] = example[key]

        # Filter nested metadata
        if "metadata" in example:
            filtered["metadata"] = self._filter_metadata(example["metadata"])

        return filtered

    def tag(self, meta: ParsedMetadata) -> TaggingResult:
        """Return labels or raise :class:`TaggingError` on a failed model call.

        A successful response containing ``Unknown`` remains a valid abstention.
        Confidence is the model's score, not a calibrated correctness estimate.
        """
        output = self.tag_with_details(meta)
        confidence = output["confidence"]
        reasoning = output.get("reasoning", {})
        def snippet(key: str) -> str:
            value = reasoning.get(key, "N/A")
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            return text[:100]
        return TaggingResult(
            pathology=output["pathology"], modality=output["modality"], type=output["type"],
            confidence=sum(confidence[axis] for axis in ("pathology", "modality", "type")) / 3,
            rationale=f"Few-shot: {snippet('few_shot_analysis')}... | Metadata: {snippet('metadata_analysis')}...",
        )

    def _build_system_prompt(self) -> str:
        """
        Build system prompt from prompt.md.

        Returns:
            System message with full instructions
        """
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _build_user_message(self, meta: ParsedMetadata) -> str:
        """
        Build user message with few-shot examples and dataset to classify.

        Args:
            meta: Metadata to classify

        Returns:
            JSON string with few_shot_examples and single dataset
        """
        # Filter metadata to only include relevant fields
        # Note: dataset_id is intentionally excluded to prevent bias
        filtered_metadata = self._filter_metadata(dict(meta))

        # Build message with single dataset (not array)
        message = {
            "few_shot_examples": self.few_shot_examples,
            "dataset": filtered_metadata  # Single dataset, no dataset_id
        }

        return json.dumps(message, indent=2, ensure_ascii=False)

    def _build_request_payload(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

    def _call_api(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        """
        Make API call to OpenRouter.ai.

        Args:
            system_prompt: System message with instructions
            user_message: User message with data to classify

        Returns:
            API response data

        Raises:
            requests.RequestException: If API call fails
            ValueError: If response format is invalid
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = self._build_request_payload(system_prompt, user_message)
        endpoint = self.endpoint

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=int(os.getenv("LLM_TIMEOUT", "300"))
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                # Internal capture is overwritten locally, never trusted from
                # the provider. It contains no authorization headers.
                data["_eegdash_request"] = {"payload": payload, "endpoint": endpoint}
            return data

        except requests.RequestException as e:
            if self.verbose:
                print(f"API request failed: {type(e).__name__}", file=sys.stderr)
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response status: {e.response.status_code}", file=sys.stderr)
            raise

    def _clean_json_response(self, content: str) -> str:
        """
        Clean LLM response to extract pure JSON.

        Removes markdown code fences and leading/trailing whitespace.

        Args:
            content: Raw LLM response

        Returns:
            Cleaned JSON string
        """
        content = content.strip()

        # Remove markdown code fences if present
        if content.startswith("```"):
            # Remove opening fence
            lines = content.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = '\n'.join(lines).strip()

        return content

    def _decode_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the record envelope before label normalization."""
        try:
            content = response_data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("message content must be text")
            def reject_constant(value):
                raise ValueError(f"Non-JSON numeric constant: {value}")
            output = json.loads(self._clean_json_response(content), parse_constant=reject_constant)
            if not isinstance(output, dict):
                raise ValueError("response must be a JSON object")
            for axis in ("pathology", "modality", "type"):
                raw = output[axis]
                if not isinstance(raw, (str, list)) or (isinstance(raw, list) and not all(isinstance(v, str) for v in raw)):
                    raise ValueError(f"{axis} must contain strings")
            confidence = output["confidence"]
            if not isinstance(confidence, dict):
                raise ValueError("confidence must be an object")
            for axis in ("pathology", "modality", "type"):
                score = confidence[axis]
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"confidence.{axis} must be finite and between zero and one")
            if "reasoning" in output and not isinstance(output["reasoning"], dict):
                raise ValueError("reasoning must be an object when supplied")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise TaggingError(f"Invalid model response: {exc}") from exc
        return output

    def _parse_response(self, response_data: Dict[str, Any], meta: ParsedMetadata) -> TaggingResult:
        """Parse a response; retained for callers of the earlier private helper."""
        output = self._decode_response(response_data)
        for axis in ("pathology", "modality", "type"):
            output[axis] = normalize_labels(axis, output[axis])
        reasoning = output.get("reasoning", {})
        snippets = [str(reasoning.get(key, "N/A"))[:100] for key in ("few_shot_analysis", "metadata_analysis")]
        return TaggingResult(
            pathology=output["pathology"], modality=output["modality"], type=output["type"],
            confidence=sum(output["confidence"][axis] for axis in ("pathology", "modality", "type")) / 3,
            rationale=f"Few-shot: {snippets[0]}... | Metadata: {snippets[1]}...",
        )

    def tag_with_details(self, meta: ParsedMetadata, dataset_id: str = "unknown") -> Dict[str, Any]:
        """Return a detailed record, or raise :class:`TaggingError` on failure.

        ``dataset_id`` is added after the request and is not sent as a field.
        Required labels/confidence are validated, then aliases and vocabulary
        rules are applied. Reasoning values are preserved; this does not verify
        their meaning or require that every reasoning value is a string.
        No failure record is disguised as a successful ``Unknown`` response.
        """
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(meta)
        captured_payload = self._build_request_payload(system_prompt, user_message)
        captured_endpoint = self.endpoint
        try:
            response_data = self._call_api(system_prompt, user_message)
        except (requests.RequestException, ValueError) as exc:
            raise TaggingError("Model request failed") from exc
        if isinstance(response_data, dict) and "_eegdash_request" in response_data:
            capture = response_data.pop("_eegdash_request")
            captured_payload, captured_endpoint = capture["payload"], capture["endpoint"]
        output = self._decode_response(response_data)
        for axis in ("pathology", "modality", "type"):
            output[axis] = normalize_labels(axis, output[axis])
        output["dataset_id"] = dataset_id
        output["model"] = captured_payload["model"]
        output["status"] = "ok"
        output["input_metadata"], output["tagger_meta"] = request_provenance(
            captured_payload, captured_endpoint, response_data,
        )
        return output
