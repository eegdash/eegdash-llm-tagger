#!/usr/bin/env python
"""
EEGDash Dataset Tagging Module

This module provides functionality to tag EEG/MEG datasets with pathology,
modality, and type labels using fixed vocabularies. It implements a rule-based
DummyTagger that can be replaced with an LLM-backed tagger in the future.
"""

from typing import Protocol, TypedDict, List, Dict, Any, Optional
import json
import argparse


# ============================================================================
# Fixed Label Vocabularies
# ============================================================================

PATHOLOGY_LABELS = [
    "Alcohol",
    "Cancer",
    "Dementia",
    "Depression",
    "Development",
    "Dyslexia",
    "Epilepsy",
    "Healthy",
    "Obese",
    "Other",
    "Parkinson's Disease",
    "Schizophrenia/Psychosis",
    "Surgery",
    "Traumatic Brain Injury",
    "Unknown",
]

MODALITY_LABELS = [  # Modality of experiment
    "Auditory",
    "Anesthesia",
    "Motor",
    "Multi sensory",
    "Tactile",
    "Other",
    "Resting State",
    "Sleep",
    "Unknown",
    "Visual",
]

TYPE_LABELS = [  # Type of experiment
    "Affect",
    "Attention",
    "Clinical/Intervention",
    "Decision making",
    "Learning",
    "Memory",
    "Motor",
    "Other",
    "Perception",
    "Resting state",
    "Sleep",
    "Unknown",
]


# ============================================================================
# Type Definitions
# ============================================================================

class ParsedMetadata(TypedDict):
    """Metadata extracted from BIDS dataset by the parser."""
    title: str
    dataset_description: str
    readme: str
    participants_overview: str
    tasks: List[str]
    events: List[str]
    paper_abstract: str


class TaggingResult(TypedDict, total=False):
    """Result of tagging operation."""
    pathology: List[str]
    modality: List[str]  # "Modality of experiment" labels
    type: List[str]      # "Type of experiment" labels
    confidence: float
    rationale: str


class Tagger(Protocol):
    """Protocol defining the tagger interface."""
    def tag(self, meta: ParsedMetadata) -> TaggingResult:
        ...


# ============================================================================
# Helper Functions
# ============================================================================

def _concat_text(meta: ParsedMetadata) -> str:
    """
    Concatenate all text fields into a single lowercase string for keyword matching.

    Args:
        meta: Parsed metadata dict

    Returns:
        Lowercase string with all text concatenated
    """
    return " ".join([
        meta.get("title", ""),
        meta.get("dataset_description", ""),
        meta.get("readme", ""),
        meta.get("participants_overview", ""),
        " ".join(meta.get("tasks", [])),
        " ".join(meta.get("events", [])),
    ]).lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the given keywords."""
    return any(keyword in text for keyword in keywords)


# ============================================================================
# DummyTagger Implementation
# ============================================================================

class DummyTagger:
    """
    Simple heuristic-based tagger used as a placeholder.
    Later we will replace this with a real LLM-backed implementation.
    """

    def tag(self, meta: ParsedMetadata) -> TaggingResult:
        """
        Tag a dataset using rule-based heuristics.

        Args:
            meta: Parsed metadata from BIDS dataset

        Returns:
            TaggingResult with pathology, modality, type labels and rationale
        """
        text = _concat_text(meta)
        tasks_text = " ".join(meta.get("tasks", [])).lower()
        events_text = " ".join(meta.get("events", [])).lower()

        # Tag each dimension
        pathology, path_rationale = self._tag_pathology(text)
        modality, mod_rationale = self._tag_modality(text, tasks_text, events_text)
        exp_type, type_rationale = self._tag_type(text, tasks_text, events_text)

        # Build rationale
        rationale = f"Pathology={', '.join(pathology)} ({path_rationale}); " \
                   f"Modality={', '.join(modality)} ({mod_rationale}); " \
                   f"Type={', '.join(exp_type)} ({type_rationale})"

        return TaggingResult(
            pathology=pathology,
            modality=modality,
            type=exp_type,
            confidence=0.0,
            rationale=rationale
        )

    def _tag_pathology(self, text: str) -> tuple[List[str], str]:
        """
        Determine pathology labels using keyword matching.

        Returns:
            Tuple of (labels, rationale)
        """
        # Check for specific pathologies
        if _contains_any(text, ["epilepsy", "epileptic"]):
            return ["Epilepsy"], "found 'epilepsy' in text"

        if _contains_any(text, ["schizophrenia", "psychosis", "psychotic"]):
            return ["Schizophrenia/Psychosis"], "found 'schizophrenia/psychosis' in text"

        if _contains_any(text, ["parkinson", "parkinson's"]):
            return ["Parkinson's Disease"], "found 'parkinson' in text"

        if _contains_any(text, ["dementia", "alzheimer"]):
            return ["Dementia"], "found 'dementia/alzheimer' in text"

        if _contains_any(text, ["depression", "depressive", "mdd"]):
            return ["Depression"], "found 'depression' in text"

        if _contains_any(text, ["tbi", "traumatic brain injury"]):
            return ["Traumatic Brain Injury"], "found 'traumatic brain injury' in text"

        if _contains_any(text, ["autism", "asd", "autistic", "developmental disorder"]):
            return ["Development"], "found 'autism/developmental' in text"

        if _contains_any(text, ["dyslexia", "dyslexic"]):
            return ["Dyslexia"], "found 'dyslexia' in text"

        if _contains_any(text, ["obese", "obesity"]):
            return ["Obese"], "found 'obesity' in text"

        if _contains_any(text, ["cancer", "tumor", "tumour"]):
            return ["Cancer"], "found 'cancer/tumor' in text"

        if _contains_any(text, ["surgery", "post-operative", "postoperative"]):
            return ["Surgery"], "found 'surgery' in text"

        if _contains_any(text, ["alcohol", "alcoholic", "alcoholism"]):
            return ["Alcohol"], "found 'alcohol' in text"

        # Check for healthy/control subjects
        if _contains_any(text, ["healthy", "control", "normal"]) and \
           not _contains_any(text, ["patient", "disorder", "disease", "clinical"]):
            return ["Healthy"], "found 'healthy/control' without pathology indicators"

        # Default to Unknown
        return ["Unknown"], "no clear pathology indicators"

    def _tag_modality(self, text: str, tasks: str, events: str) -> tuple[List[str], str]:
        """
        Determine modality of experiment labels.

        Returns:
            Tuple of (labels, rationale)
        """
        modalities = []
        reasons = []

        # Check for specific modalities
        if _contains_any(text, ["sleep", "sleeping", "nrem", "rem sleep"]):
            modalities.append("Sleep")
            reasons.append("sleep-related")

        if _contains_any(text, ["resting", "rest", "eyes closed", "eyes-closed",
                                "eyes open", "eyes-open", "resting state"]):
            modalities.append("Resting State")
            reasons.append("resting state")

        if _contains_any(text + tasks + events,
                        ["auditory", "tone", "sound", "audio", "acoustic", "hearing"]):
            modalities.append("Auditory")
            reasons.append("auditory stimuli")

        if _contains_any(text + tasks + events,
                        ["visual", "image", "grating", "checkerboard", "face",
                         "picture", "video", "viewing"]):
            modalities.append("Visual")
            reasons.append("visual stimuli")

        if _contains_any(text + tasks + events,
                        ["tactile", "somatosensory", "vibration", "touch", "haptic"]):
            modalities.append("Tactile")
            reasons.append("tactile stimuli")

        if _contains_any(text + tasks + events,
                        ["motor", "movement", "walking", "gait", "grasp", "reach",
                         "finger", "hand", "foot", "locomotion", "treadmill"]):
            modalities.append("Motor")
            reasons.append("motor activity")

        if _contains_any(text, ["anesthesia", "anesthetic", "sedation"]):
            modalities.append("Anesthesia")
            reasons.append("anesthesia")

        # Apply multi-sensory logic: if more than 1 sensory modality, use "Multi sensory"
        # (excluding Sleep and Resting State from this count)
        sensory_modalities = [m for m in modalities
                             if m not in ["Sleep", "Resting State", "Anesthesia"]]

        if len(sensory_modalities) > 1:
            # Keep Sleep/Resting State/Anesthesia if present, replace others with Multi sensory
            special_modalities_set = {"Sleep", "Resting State", "Anesthesia"}
            new_modalities = [m for m in modalities if m in special_modalities_set]
            new_modalities.append("Multi sensory")

            # Update reasons accordingly
            new_reasons = [r for m, r in zip(modalities, reasons) if m in special_modalities_set]
            new_reasons.append("multiple sensory modalities")

            modalities = new_modalities
            reasons = new_reasons

        # If nothing matched but it's task-based
        if not modalities:
            if tasks or _contains_any(text, ["task", "paradigm", "experiment"]):
                modalities.append("Other")
                reasons.append("task-based but unclear modality")
            else:
                modalities.append("Unknown")
                reasons.append("no clear modality indicators")

        rationale = ", ".join(reasons) if reasons else "no indicators"
        return modalities, rationale

    def _tag_type(self, text: str, tasks: str, events: str) -> tuple[List[str], str]:
        """
        Determine type of experiment labels.

        Returns:
            Tuple of (labels, rationale)
        """
        types = []
        reasons = []

        # Check for specific experiment types
        if _contains_any(text, ["resting", "rest", "eyes closed", "eyes-closed"]):
            types.append("Resting state")
            reasons.append("resting state")

        if _contains_any(text, ["sleep", "sleeping", "nrem", "rem"]):
            types.append("Sleep")
            reasons.append("sleep study")

        if _contains_any(text + tasks + events,
                        ["motor imagery", "movement", "walking", "gait", "reach",
                         "grasp", "finger tapping", "hand", "locomotion", "treadmill"]):
            types.append("Motor")
            reasons.append("motor task")

        if _contains_any(text + tasks + events,
                        ["decision", "choice", "oddball", "go/no-go", "go-nogo",
                         "gambling", "probability", "stroop"]):
            types.append("Decision making")
            reasons.append("decision-making task")

        if _contains_any(text + tasks + events,
                        ["memory", "n-back", "nback", "delayed match", "recall",
                         "recognition", "working memory"]):
            types.append("Memory")
            reasons.append("memory task")

        if _contains_any(text + tasks + events,
                        ["attention", "attend", "vigilance", "cue", "target detection"]):
            types.append("Attention")
            reasons.append("attention task")

        if _contains_any(text + tasks + events,
                        ["emotion", "affect", "valence", "fear", "happy", "angry",
                         "sad", "facial expression"]):
            types.append("Affect")
            reasons.append("affective/emotional task")

        if _contains_any(text + tasks + events,
                        ["learning", "feedback", "reinforcement", "training", "acquisition"]):
            types.append("Learning")
            reasons.append("learning task")

        if _contains_any(text,
                        ["clinical", "intervention", "treatment", "therapy",
                         "drug", "medication", "therapeutic"]):
            types.append("Clinical/Intervention")
            reasons.append("clinical intervention")

        if _contains_any(text + tasks + events,
                        ["perception", "discrimination", "detection", "recognition",
                         "perceptual"]):
            types.append("Perception")
            reasons.append("perceptual task")

        # Limit to top 2 types (prefer earlier matches which are more specific)
        if len(types) > 2:
            types = types[:2]
            reasons = reasons[:2]

        # Default to Unknown if nothing matched
        if not types:
            types.append("Unknown")
            reasons.append("no clear type indicators")

        rationale = ", ".join(reasons) if reasons else "no indicators"
        return types, rationale


# ============================================================================
# Public API
# ============================================================================

def tag_from_summary(meta: Dict[str, Any], tagger: Optional[Tagger] = None) -> TaggingResult:
    """
    Coerce a raw dict (from the parser) into ParsedMetadata,
    then run the tagger (default = DummyTagger) and return a TaggingResult.

    Args:
        meta: Raw dict from parser output (may include extra keys like dataset_id)
        tagger: Optional custom tagger instance (defaults to DummyTagger)

    Returns:
        TaggingResult with pathology, modality, type labels
    """
    # Coerce to ParsedMetadata format, providing defaults for missing keys
    parsed_meta = ParsedMetadata(
        title=meta.get("title", ""),
        dataset_description=meta.get("dataset_description", ""),
        readme=meta.get("readme", ""),
        participants_overview=meta.get("participants_overview", ""),
        tasks=meta.get("tasks", []) if isinstance(meta.get("tasks"), list) else [],
        events=meta.get("events", []) if isinstance(meta.get("events"), list) else []
    )

    # Use provided tagger or default to DummyTagger
    if tagger is None:
        tagger = DummyTagger()

    return tagger.tag(parsed_meta)


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """CLI entry point for testing the tagger."""
    parser = argparse.ArgumentParser(
        description="Tag EEG/MEG dataset metadata with pathology, modality, and type labels"
    )
    parser.add_argument(
        "--meta-json",
        required=True,
        help="Path to JSON file containing parser output"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print human-readable summary in addition to JSON"
    )

    args = parser.parse_args()

    # Load metadata JSON
    try:
        with open(args.meta_json, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return 1

    # Run tagger
    result = tag_from_summary(meta)

    # Print JSON output
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Print pretty summary if requested
    if args.pretty:
        print("\n" + "=" * 60)
        print("Human-Readable Summary")
        print("=" * 60)
        print(f"Pathology: {', '.join(result['pathology'])}")
        print(f"Modality:  {', '.join(result['modality'])}")
        print(f"Type:      {', '.join(result['type'])}")
        print(f"Confidence: {result.get('confidence', 0.0)}")
        if 'rationale' in result:
            print(f"\nRationale: {result['rationale']}")

    return 0


if __name__ == "__main__":
    exit(main())
