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

3. Set up environment variables:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual API keys
# NEVER commit the .env file to git - it's already in .gitignore
```

Edit `.env` and add your actual API keys:
- `OPENROUTER_API_KEY`: Get from https://openrouter.ai/
- `GITHUB_TOKEN`: Generate at https://github.com/settings/tokens

4. Load environment variables (required for each terminal session):
```bash
# Option 1: Source the .env file (bash/zsh)
export $(cat .env | xargs)

# Option 2: Use python-dotenv (automatic loading)
pip install python-dotenv
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

## LLM-Based Tagging with OpenRouter.ai

### Setup

1. Get an API key from https://openrouter.ai/
2. Add it to your `.env` file (see Installation section above)
3. Load environment variables:
   ```bash
   export $(cat .env | xargs)
   ```

### Usage

#### Test with Single Dataset

Test the API integration with one dataset first:

```bash
python scripts/test_llm_tagger.py
```

This will:
- Load the first dataset from `data/processed/incomplete_metadata.json`
- Call the OpenRouter API using GPT-4 Turbo
- Display tagging results with confidence scores and reasoning
- Save output to `data/processed/test_llm_output.json`

#### Process All Incomplete Datasets

Tag all incomplete datasets (or a limited subset for testing):

```bash
# Process first 5 datasets (recommended for testing)
python scripts/tag_with_llm.py \
    --input data/processed/incomplete_metadata.json \
    --output data/processed/llm_output.json \
    --model openai/gpt-4-turbo \
    --limit 5 \
    --verbose

# Process all datasets (may incur significant API costs)
python scripts/tag_with_llm.py \
    --input data/processed/incomplete_metadata.json \
    --output data/processed/llm_output.json \
    --model openai/gpt-4-turbo \
    --verbose
```

#### Update CSV with LLM Results

After generating predictions, update your CSV:

```bash
python scripts/update_csv.py \
    --llm-json data/processed/llm_output.json \
    --csv dataset_summary.csv \
    --confidence-threshold 0.8 \
    --verbose
```

### Supported Models

- **`openai/gpt-4-turbo`** - GPT-4 Turbo (recommended, ~$0.13/dataset)
- **`openai/gpt-4`** - GPT-4 (more expensive, ~$0.33/dataset)
- **`anthropic/claude-3-opus`** - Claude 3 Opus (~$0.23/dataset)
- **`anthropic/claude-3-sonnet`** - Claude 3 Sonnet (faster, cheaper)

### Cost Estimation

Based on ~9,500 tokens per dataset (8K input + 1.5K output):

- **GPT-4 Turbo**: ~$0.13 per dataset (~$38 for all 295)
- **GPT-4**: ~$0.33 per dataset (~$97 for all 295)
- **Claude 3 Opus**: ~$0.23 per dataset (~$68 for all 295)

**Recommendation**: Start with `--limit 5` to verify results before processing all datasets.

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
