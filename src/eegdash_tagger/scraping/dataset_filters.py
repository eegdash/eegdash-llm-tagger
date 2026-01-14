"""Dataset filtering functions for checking tagging status."""

from typing import Dict


def check_tagging_status(row: Dict, complete: bool = False) -> bool:
    """
    Check if a dataset has complete or incomplete tagging.

    This unified function consolidates the logic from needs_tagging() and
    has_complete_tagging() to reduce code duplication.

    Args:
        row: Dataset dict with pathology, modality, and type fields
             (values can be strings or lists of strings)
        complete: If True, check if all tags are filled. If False, check if
                  any tag is missing (inverse logic).

    Returns:
        - If complete=True: True if all three tags are filled and not 'unknown'
        - If complete=False: True if any tag is missing or 'unknown'
    """
    for key in ['pathology', 'modality', 'type']:
        raw_value = row.get(key, '')

        # Handle both list and string formats
        if isinstance(raw_value, list):
            # For lists: check if non-empty and first element is not 'unknown'
            value = raw_value[0].strip().lower() if raw_value else ''
        else:
            value = raw_value.strip().lower() if raw_value else ''

        is_filled = value and value != 'unknown'

        if complete and not is_filled:
            # Looking for complete tagging, but found an empty/unknown field
            return False
        if not complete and not is_filled:
            # Looking for incomplete tagging, and found an empty/unknown field
            return True

    # If we get here:
    # - For complete=True: all fields were filled (return True)
    # - For complete=False: all fields were filled (return False)
    return complete


def needs_tagging(row: Dict) -> bool:
    """
    Return True if this dataset is missing any of the three key tags:
    pathology, modality, or type.

    Treats empty strings or case-insensitive 'unknown' as missing.

    Args:
        row: Dataset dict with pathology, modality, and type fields

    Returns:
        True if any tag is missing or unknown
    """
    return check_tagging_status(row, complete=False)


def has_complete_tagging(row: Dict) -> bool:
    """
    Return True if this dataset has ALL three key tags filled:
    pathology, modality, and type.

    This is the opposite of needs_tagging().

    Args:
        row: Dataset dict with pathology, modality, and type fields

    Returns:
        True if all tags are present and not 'unknown'
    """
    return check_tagging_status(row, complete=True)
