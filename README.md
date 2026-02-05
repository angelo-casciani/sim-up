# agents-test

## Setup

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/angelo/.local/bin:$PATH"
```

2. Install dependencies:
```bash
poetry install
```

3. Run the script:
```bash
poetry run python tests.py
```

## Notes
- Make sure `.env` file contains the required AI Core credentials
- Python 3.11+ is required