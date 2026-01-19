"""
OpenRouter.ai LLM-based tagger for EEG datasets.

This module provides an LLM-powered implementation of the Tagger protocol
using OpenRouter.ai API to classify datasets with GPT-4/Claude models.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

from .tagger import Tagger, ParsedMetadata, TaggingResult


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

    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

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
    def get_default_few_shot_path(cls) -> Path:
        """Get the default path to few_shot_examples.json bundled with the package."""
        return Path(__file__).parent.parent.parent.parent / "data" / "processed" / "few_shot_examples.json"

    @classmethod
    def get_default_prompt_path(cls) -> Path:
        """Get the default path to prompt.md bundled with the package."""
        return Path(__file__).parent.parent.parent.parent / "prompt.md"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4",
        verbose: bool = False,
        few_shot_path: Optional[Path] = None,
        prompt_path: Optional[Path] = None,
        max_tokens: int = 4000
    ):
        """
        Initialize OpenRouterTagger.

        Args:
            api_key: OpenRouter.ai API key. If None, reads from OPENROUTER_API_KEY env var
            model: Model identifier (default: "openai/gpt-4")
            verbose: Enable verbose output
            few_shot_path: Path to few_shot_examples.json. If None, uses package default
            prompt_path: Path to prompt.md. If None, uses package default
            max_tokens: Maximum tokens for LLM response (default: 4000)

        Raises:
            ValueError: If API key is not provided and OPENROUTER_API_KEY env var is not set
            FileNotFoundError: If few-shot examples or prompt file cannot be found
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.model = model
        self.verbose = verbose
        self.max_tokens = max_tokens

        # Store prompt path for later use
        self.prompt_path = prompt_path if prompt_path else self.get_default_prompt_path()

        # Load few-shot examples
        if few_shot_path is None:
            few_shot_path = self.get_default_few_shot_path()

        if not few_shot_path.exists():
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
        """
        Tag a dataset using LLM.

        Args:
            meta: Parsed metadata from dataset

        Returns:
            TaggingResult with pathology, modality, type, confidence, and rationale
        """
        try:
            # Build prompt messages
            system_prompt = self._build_system_prompt()
            user_message = self._build_user_message(meta)

            if self.verbose:
                print(f"\nCalling {self.model} API...")
                print(f"System prompt length: {len(system_prompt)} chars")
                print(f"User message length: {len(user_message)} chars")

            # Call API
            response_data = self._call_api(system_prompt, user_message)

            # Parse response
            result = self._parse_response(response_data, meta)

            if self.verbose:
                print("✓ Successfully tagged dataset")

            return result

        except Exception as e:
            if self.verbose:
                print(f"✗ Error tagging dataset: {e}", file=sys.stderr)

            # Return fallback result
            return TaggingResult(
                pathology=["Unknown"],
                modality=["Unknown"],
                type=["Unknown"],
                confidence=0.0,
                rationale=f"API call failed: {str(e)}"
            )

    def _build_system_prompt(self) -> str:
        """
        Build system prompt from prompt.md.

        Returns:
            System message with full instructions
        """
        if not self.prompt_path.exists():
            # Fallback to minimal instructions
            return """You are an expert EEG/MEG dataset curator for the EEGDash catalog.

Your task is to classify datasets with:
1. Pathology - one or two labels
2. Modality of experiment - one or two labels
3. Type of experiment - one or two labels
4. Confidence scores for each category (0-1)

Use few-shot examples as ground truth for labeling patterns.
Base reasoning on actual metadata phrases.
Return strict JSON format only."""

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

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            # Limit max_tokens to control costs and fit within credit limits
            # The response should be a relatively small JSON object
            "max_tokens": self.max_tokens,
        }

        # Enforce JSON mode for all models
        # OpenRouter supports response_format for many models
        # If a model doesn't support it, the API will ignore it gracefully
        # and the prompt instructions will still enforce JSON output
        payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(
                self.ENDPOINT,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            if self.verbose:
                print(f"API request failed: {e}", file=sys.stderr)
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response status: {e.response.status_code}", file=sys.stderr)
                    print(f"Response body: {e.response.text[:500]}", file=sys.stderr)
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

    def _parse_response(self, response_data: Dict[str, Any], meta: ParsedMetadata) -> TaggingResult:
        """
        Parse API response into TaggingResult.

        Args:
            response_data: Raw API response
            meta: Original metadata (for fallback)

        Returns:
            TaggingResult with extracted labels and confidence

        Raises:
            ValueError: If response format is invalid
        """
        try:
            # Extract message content
            content = response_data["choices"][0]["message"]["content"]

            if self.verbose:
                print(f"Raw response length: {len(content)} chars")

            # Clean and parse JSON content
            cleaned_content = self._clean_json_response(content)
            llm_output = json.loads(cleaned_content)

            # New format: flat object (no "results" array)
            # Extract labels (arrays)
            pathology = llm_output.get("pathology", ["Unknown"])
            modality = llm_output.get("modality", ["Unknown"])
            exp_type = llm_output.get("type", ["Unknown"])

            # Extract confidence scores
            confidence_scores = llm_output.get("confidence", {})
            avg_confidence = sum([
                confidence_scores.get("pathology", 0.0),
                confidence_scores.get("modality", 0.0),
                confidence_scores.get("type", 0.0),
            ]) / 3.0

            # Extract reasoning
            reasoning = llm_output.get("reasoning", {})
            rationale = (
                f"Few-shot: {reasoning.get('few_shot_analysis', 'N/A')[:100]}... | "
                f"Metadata: {reasoning.get('metadata_analysis', 'N/A')[:100]}..."
            )

            return TaggingResult(
                pathology=pathology,
                modality=modality,
                type=exp_type,
                confidence=avg_confidence,
                rationale=rationale
            )

        except (KeyError, json.JSONDecodeError, IndexError) as e:
            if self.verbose:
                print(f"Error parsing response: {e}", file=sys.stderr)
                print(f"Response data: {response_data}", file=sys.stderr)

            # Return fallback
            return TaggingResult(
                pathology=["Unknown"],
                modality=["Unknown"],
                type=["Unknown"],
                confidence=0.0,
                rationale=f"Failed to parse LLM response: {str(e)}"
            )

    def tag_with_details(self, meta: ParsedMetadata, dataset_id: str = "unknown") -> Dict[str, Any]:
        """
        Tag dataset and return detailed results including reasoning.

        This method returns the full LLM output including confidence breakdown
        and reasoning, which is useful for the llm_output.json format.

        Note: dataset_id is NOT sent to the LLM (to prevent bias) but is
        added back to the result after the API call.

        Args:
            meta: Parsed metadata from dataset
            dataset_id: Dataset identifier

        Returns:
            Dict with dataset_id, pathology, modality, type, confidence, and reasoning
        """
        try:
            # Build and call API
            system_prompt = self._build_system_prompt()
            user_message = self._build_user_message(meta)
            response_data = self._call_api(system_prompt, user_message)

            # Extract and clean full result (new flat format, no "results" array)
            content = response_data["choices"][0]["message"]["content"]
            cleaned_content = self._clean_json_response(content)
            llm_output = json.loads(cleaned_content)

            # Add dataset_id back (it was masked when sending to LLM)
            llm_output["dataset_id"] = dataset_id

            return llm_output

        except Exception as e:
            if self.verbose:
                print(f"Error in tag_with_details: {e}", file=sys.stderr)

            return {
                "dataset_id": dataset_id,
                "pathology": ["Unknown"],
                "modality": ["Unknown"],
                "type": ["Unknown"],
                "confidence": {
                    "pathology": 0.0,
                    "modality": 0.0,
                    "type": 0.0
                },
                "reasoning": {
                    "few_shot_analysis": f"Error: {str(e)}",
                    "metadata_analysis": "N/A",
                    "paper_abstract_analysis": "N/A",
                    "decision_summary": "API call failed"
                }
            }
