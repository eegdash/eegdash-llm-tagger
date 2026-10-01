"""EEGDash tag taxonomy v2: the single source of truth for allowed labels.

Three axes, each answering one question so no concept lives on two axes:

- ``pathology``: which clinical condition were participants recruited for?
  ``Healthy`` for normative cohorts. Patient + control designs get the
  condition. Developmental stage is not a pathology (use the condition, or
  ``Healthy``); neither is a procedure such as surgery (use the condition
  that motivated it, e.g. ``Epilepsy`` for pre-surgical iEEG).
- ``modality``: through which sensory channel were stimuli delivered?
  ``No stimulus`` when there is none (rest, sleep, self-paced movement or
  imagery without cues, anaesthesia).
- ``type``: what paradigm / construct is studied? ``Clinical`` when the
  condition itself is the research focus (biomarkers, diagnosis cohorts),
  ``Intervention`` for drug, stimulation or therapy studies.

Labels never contain ``/`` or ``,`` because the docs split tag cells on
those characters.
"""

from __future__ import annotations

from typing import Iterable

PATHOLOGY_LABELS = [
    "Healthy",
    "ADHD",
    "ALS",
    "Alcohol use disorder",
    "Autism",
    "Cancer",
    "Chronic pain",
    "Dementia",
    "Depression",
    "Disorders of consciousness",
    "Dyslexia",
    "Epilepsy",
    "Obesity",
    "Parkinson's",
    "Psychiatric (transdiagnostic)",
    "Schizophrenia spectrum",
    "Spinal cord injury",
    "Stroke",
    "TBI",
    "Other clinical",
    "Unknown",
]

MODALITY_LABELS = [
    "Visual",
    "Auditory",
    "Tactile",
    "Multisensory",
    "No stimulus",
    "Other",
    "Unknown",
]

TYPE_LABELS = [
    "Perception",
    "Attention",
    "Memory",
    "Learning",
    "Decision-making",
    "Affect",
    "Language",
    "Motor",
    "Resting-state",
    "Sleep",
    "Consciousness",
    "Clinical",
    "Intervention",
    "Other",
    "Unknown",
]

LABELS = {"pathology": PATHOLOGY_LABELS, "modality": MODALITY_LABELS, "type": TYPE_LABELS}
MAX_LABELS = 2

# Spellings the model (or v1 data) produces for a v2 label.
_ALIASES = {
    "pathology": {
        "healthy controls": "Healthy",
        "control": "Healthy",
        "parkinson's disease": "Parkinson's",
        "parkinson": "Parkinson's",
        "schizophrenia/psychosis": "Schizophrenia spectrum",
        "schizophrenia": "Schizophrenia spectrum",
        "psychosis": "Schizophrenia spectrum",
        "traumatic brain injury": "TBI",
        "alcohol": "Alcohol use disorder",
        "obese": "Obesity",
        "asd": "Autism",
        "autism spectrum disorder": "Autism",
        "amyotrophic lateral sclerosis": "ALS",
        "mci": "Dementia",
        "alzheimer's": "Dementia",
        "other": "Other clinical",
    },
    "modality": {
        "multi sensory": "Multisensory",
        "resting state": "No stimulus",
        "sleep": "No stimulus",
        "anesthesia": "No stimulus",
        "none": "No stimulus",
    },
    "type": {
        "decision making": "Decision-making",
        "resting state": "Resting-state",
        "rest": "Resting-state",
        "clinical/intervention": "Clinical",
        "anesthesia": "Consciousness",
    },
}


def normalize_labels(axis: str, values: Iterable[str] | str | None) -> list[str]:
    """Map raw labels onto the v2 vocabulary for ``axis``.

    Unknown strings are dropped; at most :data:`MAX_LABELS` are kept, in
    order. Returns ``["Unknown"]`` when nothing valid remains, and drops
    ``Unknown`` next to a real label.
    """
    if isinstance(values, str):
        values = [values]
    canonical = {label.lower(): label for label in LABELS[axis]}
    aliases = _ALIASES.get(axis, {})
    out: list[str] = []
    for raw in values or []:
        key = " ".join(str(raw).split()).lower()
        label = canonical.get(key) or aliases.get(key)
        if label and label not in out:
            out.append(label)
    real = [label for label in out if label != "Unknown"]
    return real[:MAX_LABELS] or ["Unknown"]
