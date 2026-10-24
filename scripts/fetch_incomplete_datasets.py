#!/usr/bin/env python3
"""Compatibility wrapper; prefer the installed eegdash command."""
import sys
from eegdash_tagger.cli.fetch_incomplete_datasets import main

if __name__ == "__main__":
    sys.exit(main())
