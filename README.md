# Building-Open-Datasets-for-Spanish-Medical-Tasks

## Description

This repository contains training and evaluation scripts for two medical NLP tasks:

1. **Multi-label text classification of Spanish titles/abstracts**. Assigns multiple medical categories to each article based on its title and abstract. The model analyzes the processed text to identify all applicable MeSH terms.

2. **Named Entity Recognition (NER) on AnatEM (Spanish/English)**. Detects and highlights anatomical entities within medical texts, enabling structured extraction of key terms for downstream analysis. 

Classification datasets and preprocessing follow the resources from [Dataset-Creation](https://github.com/SantiagoM99/Dataset-Creation)

## Installation
Please follow these steps to run the code:

```bash
# Clone the repository
git clone <repository-url>
cd Building-Open-Datasets-for-Spanish-Medical-Tasks

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

```

## Quick Start / Usage Example

```bash
# Example: Run multi-label classification
python train_multilabel.py --config configs/multilabel_config.yaml

# Example: Run NER task
python train_ner.py --config configs/ner_config.yaml
```

## Repository Structure

```
├── data/                # Datasets and processed files
├── results/             # Output results and logs
├── configs/             # YAML configuration files
├── scripts/             # Utility scripts
├── train_multilabel.py  # Main script for classification
├── train_ner.py         # Main script for NER
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, please discuss them first via issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Credits / Acknowledgments

- Dataset creation resources: [Dataset-Creation](https://github.com/SantiagoM99/Dataset-Creation)
- AnatEM dataset: [AnatEM DB](https://www.nactem.ac.uk/anatomytagger/)
- Thanks to all contributors and the open-source community.

## Contact

For questions or suggestions, please contact: l.gomez1@uniandes.edu.co