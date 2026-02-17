<p align="center">
  <a href="https://www.vectorly.app/"><img src="https://img.shields.io/badge/Website-Vectorly.app-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white" /></a>
  <a href="https://console.vectorly.app"><img src="https://img.shields.io/badge/Console-console.vectorly.app-8b5cf6?style=for-the-badge&logo=googlechrome&logoColor=white" /></a>
  <a href="https://www.youtube.com/@VectorlyAI"><img src="https://img.shields.io/badge/YouTube-@VectorlyAI-ff0000?style=for-the-badge&logo=youtube&logoColor=white" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-10b981?style=for-the-badge&logo=apache&logoColor=white" /></a>
</p>

# bluebox 🟦

Index the world's undocumented APIs

**Why "Blue Box"?** Named after the [phone phreaking devices](https://en.wikipedia.org/wiki/Blue_box) that let tech enthusiasts in the 1960s and 70s explore telephone networks.

**You are in the right place if you ...**

* need to scrape data behind UI interactions
* are dealing with closed APIs
* are tired of complicated, endless API integrations
* want to reverse engineer websites

## Tutorial

https://github.com/user-attachments/assets/934728e1-1384-4b44-a7b0-d93480d329de

## Prerequisites

- Python 3.12+
- [uv (Python package manager)](https://github.com/astral-sh/uv) (optional, for development)
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`
- Vectorly API key
  - used by `bluexbox` agent for web data extraction
  - sign up at [console.vectorly.app](https://console.vectorly.app)
  - macOS/Linux: `export VECTORLY_API_KEY="your-key"`
  - Or Windows (PowerShell): `setx VECTORLY_API_KEY "your-key"`
  - Or add it to your `.env` file: `VECTORLY_API_KEY=your-key`
- OpenAI API key or Anthropic API key
  - set `export OPENAI_API_KEY="your-key"`  or `export ANTHROPIC_API_KEY="your-key"` (or add to `.env`)

## Installation

### From PyPI (Recommended)

**Note:** We recommend using a virtual environment to avoid dependency conflicts.

```bash
# Create and activate a virtual environment
# Option 1: Using uv (recommended - handles Python version automatically)
uv venv bluebox-env
source bluebox-env/bin/activate  # On Windows: bluebox-env\Scripts\activate
uv pip install bluebox-lib

# Option 2: Using python3 (if Python 3.12+ is your default)
python3 -m venv bluebox-env
source bluebox-env/bin/activate  # On Windows: bluebox-env\Scripts\activate
pip install bluebox-lib

# Option 3: Using pyenv (if you need a specific Python version)
pyenv install 3.12.3  # if not already installed
pyenv local 3.12.3
python -m venv bluebox-env
source bluebox-env/bin/activate  # On Windows: bluebox-env\Scripts\activate
pip install bluebox-lib

# Troubleshooting: If pip is not found, recreate the venv or use:
python -m ensurepip --upgrade  # Install pip in the venv
pip install bluebox-lib
```

### From Source (Development)

For development or if you want the latest code:

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

## BlueBox Agent

The BlueBox Agent is a conversational AI agent that automates web tasks. It searches for matching [Vectorly routines](https://vectorly.app/docs/routines/overview), executes them in parallel, and falls back to a live AI browser agent for anything without a pre-built routine.

```bash
bluebox-agent
```

**What it does:**

- Searches for relevant routines based on your natural language request
- Executes multiple routines in parallel for speed
- Falls back to an AI browser agent for free-form tasks when no routine exists
- Post-processes results with Python (CSV, JSON, etc.)
- Saves output files to a local workspace

Ask it anything: *"Find one-way trains from Boston to NYC on March 22"* — the agent finds the right routine, runs it, and gives you structured results.

## Create Your Own Routines

To learn about the core technology powering BlueBox, see [routine_discovery.md](routine_discovery.md).

## Contributing 🤝

We welcome contributions! Here's how to get started:

1. **Report bugs or request features** — Open an [issue](https://github.com/VectorlyApp/bluebox/issues)
2. **Submit code** — Fork the repo and open a [pull request](https://github.com/VectorlyApp/bluebox/pulls)
3. **Test your code** — Add unit tests and make sure all tests pass:

```bash
pytest tests/ -v
```

Please follow existing code style and include tests for new features.
