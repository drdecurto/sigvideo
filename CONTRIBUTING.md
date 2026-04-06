# Contributing to sigvideo

Thank you for your interest. Below are the essentials.

## Development setup

```bash
git clone https://github.com/drdecurto/sigvideo
cd sigvideo
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/
```

The suite covers metrics, summarizer, video extraction, the primary pipeline,
VLM NLP stage, and public API (33 tests, 2 skipped when NLTK data or torch
are absent).

## Code style

- Standard Python 3.8+ syntax
- Type hints on all public functions
- Docstrings follow the existing style in `pipeline.py`

## Submitting changes

1. Fork the repository and create a feature branch.
2. Ensure `pytest tests/` passes with no new failures.
3. Open a pull request with a clear description of what changed and why.

## Reporting issues

Please open a GitHub issue and include:
- `sigvideo` version (`python -c "import sigvideo; print(sigvideo.__version__)"`)
- Python version and OS
- Minimal code or CLI command that reproduces the issue

## Citation

If you use `sigvideo` in research, please cite the paper:

```bibtex
@article{sigvideodecurto2023,
  title   = {Summarization of Videos with the Signature Transform},
  author  = {de Curt{\`o}, J. and de Zarz{\`a}, I. and Roig, G. and Calafate, C.T.},
  journal = {Electronics},
  volume  = {12}, number = {7}, pages = {1735}, year = {2023},
  doi     = {10.3390/electronics12071735}
}
```
