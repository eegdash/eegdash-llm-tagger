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

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4",
        verbose: bool = False,
        few_shot_path: Optional[Path] = None
    ):
        """
        Initialize OpenRouterTagger.

        Args:
            api_key: OpenRouter.ai API key. If None, reads from OPENROUTER_API_KEY env var
            model: Model identifier (default: "openai/gpt-4")
            verbose: Enable verbose output
            few_shot_path: Path to few_shot_examples.json. If None, uses default location

        Raises:
            ValueError: If API key is not provided and OPENROUTER_API_KEY env var is not set
            FileNotFoundError: If few-shot examples file cannot be found
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.model = model
        self.verbose = verbose

        # Load few-shot examples
        if few_shot_path is None:
            # Default path relative to project root
            few_shot_path = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "few_shot_examples.json"

        if not few_shot_path.exists():
            raise FileNotFoundError(f"Few-shot examples not found at: {few_shot_path}")

        with open(few_shot_path, 'r', encoding='utf-8') as f:
            self.few_shot_examples = json.load(f)

        if self.verbose:
            print(f"Loaded {len(self.few_shot_examples)} few-shot examples")
            print(f"Using model: {self.model}")

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
        # Load prompt.md
        prompt_path = Path(__file__).parent.parent.parent.parent / "prompt.md"

        if not prompt_path.exists():
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

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _build_user_message(self, meta: ParsedMetadata) -> str:
        """
        Build user message with few-shot examples and test dataset.

        Args:
            meta: Metadata to classify

        Returns:
            JSON string with few_shot_examples and test dataset
        """
        # Format test dataset
        test_dataset = {
            "dataset_id": "test",
            "metadata": {
                "title": meta.get("title", ""),
                "dataset_description": meta.get("dataset_description", ""),
                "readme": meta.get("readme", ""),
                "participants_overview": meta.get("participants_overview", ""),
                "tasks": meta.get("tasks", []),
                "events": meta.get("events", []),
            }
        }

        # Build message
        message = {
            "few_shot_examples": self.few_shot_examples,
            "datasets": [test_dataset]  # Use "datasets" array as prompt.md expects
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
        }

        # Add response_format for models that support it
        if "gpt-4" in self.model.lower() or "gpt-3.5" in self.model.lower():
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

            # Parse JSON content
            llm_output = json.loads(content)

            # Extract first result
            if "results" not in llm_output or not llm_output["results"]:
                raise ValueError("No results in LLM output")

            first_result = llm_output["results"][0]

            # Extract labels (arrays)
            pathology = first_result.get("pathology", ["Unknown"])
            modality = first_result.get("modality", ["Unknown"])
            exp_type = first_result.get("type", ["Unknown"])

            # Extract confidence scores
            confidence_scores = first_result.get("confidence", {})
            avg_confidence = sum([
                confidence_scores.get("pathology", 0.0),
                confidence_scores.get("modality", 0.0),
                confidence_scores.get("type", 0.0),
            ]) / 3.0

            # Extract reasoning
            reasoning = first_result.get("reasoning", {})
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

            # Extract full result
            content = response_data["choices"][0]["message"]["content"]
            llm_output = json.loads(content)
            first_result = llm_output["results"][0]

            # Add dataset_id
            first_result["dataset_id"] = dataset_id

            return first_result

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
                    "citation_analysis": "N/A",
                    "decision_summary": "API call failed"
                }
            }
