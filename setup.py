"""Setup script for eegdash-tagger package."""

from setuptools import setup, find_packages

setup(
    name="eegdash-tagger",
    version="0.1.0",
    description="Automatic tagging of EEG datasets using LLM-based predictions",
    author="Kuntal Kokate",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0",
        "requests",
        "beautifulsoup4",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "eegdash-fetch-incomplete=scripts.fetch_incomplete_datasets:main",
            "eegdash-fetch-complete=scripts.fetch_complete_datasets:main",
            "eegdash-update-csv=scripts.update_csv:main",
        ],
    },
)
