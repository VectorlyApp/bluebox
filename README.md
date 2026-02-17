<p align="center">
  <a href="https://www.vectorly.app/"><img src="https://img.shields.io/badge/Website-Vectorly.app-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white" /></a>
  <a href="https://console.vectorly.app"><img src="https://img.shields.io/badge/Console-console.vectorly.app-8b5cf6?style=for-the-badge&logo=googlechrome&logoColor=white" /></a>
  <a href="https://vectorly.app/discord-invite"><img src="https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white" /></a>
  <a href="https://www.youtube.com/@VectorlyAI"><img src="https://img.shields.io/badge/YouTube-@VectorlyAI-ff0000?style=for-the-badge&logo=youtube&logoColor=white" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-10b981?style=for-the-badge&logo=apache&logoColor=white" /></a>
</p>
# bluebox 🟦

Index the world's undocumented APIs.

**Why "Blue Box"?** Named after the [phone phreaking devices](https://en.wikipedia.org/wiki/Blue_box) that let tech enthusiasts in the 1960s and 70s explore telephone networks.

**You are in the right place if you ...**

* need to scrape data behind UI interactions
* are dealing with closed APIs
* want to reverse engineer websites

## Tutorial

https://github.com/user-attachments/assets/934728e1-1384-4b44-a7b0-d93480d329de

## Prerequisites

- Python 3.12+
- Vectorly API key (required, used by `bluebox` agent for web data extraction)
  - Sign up at [console.vectorly.app](https://console.vectorly.app)
  - macOS/Linux: `export VECTORLY_API_KEY="your-key"`
  - Windows (PowerShell): `setx VECTORLY_API_KEY "your-key"`
  - Or add it to your `.env` file: `VECTORLY_API_KEY=your-key`
- LLM provider API key (required, used by `bluebox` agent for orchestration)
  - Configure one of the following:
  - OpenAI (default):
    - macOS/Linux: `export OPENAI_API_KEY="your-key"`
    - Windows (PowerShell): `setx OPENAI_API_KEY "your-key"`
    - `.env`: `OPENAI_API_KEY=your-key`
  - Anthropic:
    - macOS/Linux: `export ANTHROPIC_API_KEY="your-key"`
    - Windows (PowerShell): `setx ANTHROPIC_API_KEY "your-key"`
    - `.env`: `ANTHROPIC_API_KEY=your-key`
- [uv](https://github.com/astral-sh/uv) (optional, for dependency management)
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`

## Installation


```bash
# Clone the repository
git clone https://github.com/VectorlyApp/bluebox.git
cd bluebox

# Create and activate virtual environment
python3 -m venv bluebox-env
source bluebox-env/bin/activate  # On Windows: bluebox-env\Scripts\activate

# Install in editable mode
pip install -e .

# Or using uv (faster)
uv venv bluebox-env
source bluebox-env/bin/activate
uv pip install -e .
```

## Bluebox agent

The `bluebox` agent is a conversational AI agent that automates web data extraction. It searches the Vectorly web routine index for relevant web APIs, executes matched endpoints in parallel, and falls back to a live AI browser agent when no suitable pre-built routine is available.

### Quickstart
```bash
bluebox-agent
```

**What it does:**

- Interprets natural language requests and maps them to relevant routines
- Executes multiple routines concurrently for faster results
- Falls back to an AI browser agent for tasks without predefined routines
- Post-processes outputs using Python (CSV, JSON, etc.)
- Saves generated files to a local workspace

Ask it anything: *"Run a price analysis on Rolex Sea Dweller 16600"* — the agent automatically selects the right routine, runs it, and delivers structured results.


## Create your own routines

To learn about the core technology powering BlueBox, see [routine_discovery.md](routine_discovery.md).

## Contributing 🤝

We welcome contributions! Here's how to get started:

1. **Report bugs or request features** — Open an [issue](https://github.com/VectorlyApp/bluebox/issues)
2. **Submit code** — Fork the repo and open a [pull request](https://github.com/VectorlyApp/bluebox/pulls)
3. **Test your code** — Add unit tests and make sure all tests pass:

```bash
python -m pytest tests/ -v
```

Please follow existing code style and include tests for new features.
