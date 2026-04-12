# Contributing to VibeScholar

VibeScholar is maintained by a single developer. Contributions are welcome but
the project's scope and direction are set by the maintainer.

## Reporting issues

Open a GitHub issue. Include:

- What you expected vs. what happened
- Steps to reproduce (if applicable)
- Your Python version and OS

## Running tests

```bash
pip install -e ".[test]"
python -m pytest tests/ -v --ignore=tests/test_index_and_retrieval.py
```

The full suite (including integration tests that build a FAISS index from test
PDFs) takes ~2 minutes:

```bash
python -m pytest tests/ -v
```

## Pull requests

- Open an issue first to discuss the change
- Keep PRs focused on a single change
- Ensure all tests pass before submitting
