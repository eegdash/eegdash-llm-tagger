# EEGDash LLM Tagger

Automatic tagging of EEG datasets using LLM-based predictions. This tool scrapes the EEGDash website, extracts metadata from BIDS-formatted datasets, and uses machine learning to predict pathology, modality, and experiment type tags.

## Features

- **Dataset Scraping**: Automatically discovers datasets from EEGDash
- **Metadata Extraction**: Parses BIDS-formatted datasets from GitHub/OpenNeuro
- **LLM-based Tagging**: Uses language models to predict missing tags
- **CSV Updates**: Automatically updates dataset catalogs with predictions

## Installation

### Requirements

- Python 3.11 or higher
- Git
- GitHub Personal Access Token (for API access)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/eegdash-llm-tagger.git
cd eegdash-llm-tagger
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. Set up your GitHub token (for accessing OpenNeuroDatasets):
```bash
export GITHUB_TOKEN="your_github_token_here"
```

## Usage

### Fetch Incomplete Datasets

Find and process datasets with missing tags:

```bash
python scripts/fetch_incomplete_datasets.py \
    --output-json data/processed/incomplete_metadata.json \
    --verbose
```

### Fetch Complete Datasets

Process datasets that already have complete tags (for training data):

```bash
python scripts/fetch_complete_datasets.py \
    --output-json data/processed/complete_metadata.json \
    --limit 10 \
    --verbose
```

### Update CSV with LLM Predictions

Apply LLM predictions to update a dataset CSV:

```bash
python scripts/update_csv.py \
    --llm-json data/processed/llm_output.json \
    --csv dataset_summary.csv \
    --confidence-threshold 0.5 \
    --verbose
```

## Project Structure

```
eegdash-llm-tagger/
├── src/eegdash_tagger/      # Main package
│   ├── metadata/            # BIDS metadata parsing
│   ├── scraping/            # Web scraping and data collection
│   ├── tagging/             # LLM-based tagging
│   └── utils/               # CSV updates and helpers
├── scripts/                 # CLI entry points
├── tests/                   # Test files
├── tools/                   # Utility scripts
├── data/                    # Data files (gitignored)
│   ├── processed/           # Generated metadata
│   └── test/                # Test datasets
├── environment.yml          # Conda environment
└── setup.py                 # Package configuration
```

## Data Directory

The `data/` directory contains generated metadata files and is excluded from git:

- **data/processed/**: Production metadata files
  - `complete_metadata.json` - Datasets with full tags
  - `incomplete_metadata.json` - Datasets needing tags
  - `few_shot_examples.json` - LLM training examples
  - `llm_output.json` - LLM prediction results

- **data/test/**: Test datasets for development

## Development

### Running Tests

```bash
# Test metadata extraction
python tests/test_metadata.py

# Test GitHub API integration
export GITHUB_TOKEN="your_token"
python tests/test_github_api.py
```

### Utility Tools

```bash
# Show metadata for a dataset
python tools/show_metadata.py /path/to/dataset

# Create test/training datasets
python tools/create_test_set.py
```

## License

[Add your license here]

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- EEGDash project for the dataset catalog
- OpenNeuro for hosting EEG datasets
- BIDS format for standardized neuroimaging data
