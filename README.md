# Process Improvement Validation Agent

This repository contains an agentic pipeline for discovering process simulation models and extracting execution statistics from event logs, generating improvement hypotheses using an LLM-based agent, and validating them through process simulation.

## Overview

This project combines process mining (SIMOD/pm4py), generative AI (SAP Gen AI Hub SDK), and process simulation (Prosimos) to automatically identify and validate process improvements.

**Features:**
- **Process Simulation Model Discovery**: Automatically discovers baseline simulation models from event logs using SIMOD.
- **Process Execution Statistics Computation**: Automatically computes execution statistics from the input event log using PM4Py (cycle time, throughput, waiting time, resource utilization).
- **Improvement Hypotheses Generation**: Generates improvement ideas using the LLM agent based on actual process metrics.
- **Simulation Validation**: Tests the proposed improvements via the Prosimos simulator.
- **Reports Generation**: Generates improvement reports with ROI analysis.

## Architecture

```
src/
├── improvement_validation_pipeline.py  # Main orchestrator
├── main.py                             # CLI entry point
├── simulator/                          # Process discovery & simulation
│   ├── simod_discoverer.py            # SIMOD integration + log analysis
│   ├── prosimos_runner.py             # Prosimos simulation engine
│   └── metrics_extractor.py           # Metrics extraction from simulation
├── llm/                               # LLM-based hypothesis generation
│   └── hypothesis_generator.py        # LLM integration
├── validation/                        # Improvement validation
│   └── improvement_validator.py       # Baseline vs simulated comparison
├── reporting/                         # Report generation
│   └── report_generator.py            # Report formatting
└── metrics/                           # Event log analysis
    └── metrics_extractor.py           # pm4py-based log metric extraction
```

## Installation & Setup

### Prerequisites

- **Python**: 3.11 or higher
- **Docker**: For SIMOD process discovery
- **Poetry**: For dependency management
- **SAP Gen AI Hub Account**: For LLM access (with API credentials)

### Step 1: Clone & Navigate

```bash
git clone <repository-url>
cd agents-test
```

### Step 2: Set Up Poetry Environment

Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Create virtual environment with Poetry:
```bash
poetry install
```

This will:
- Create a virtual environment
- Install all dependencies from `pyproject.toml`
- Set up SAP SDK and PyPI package sources

Activate the virtual environment:
```bash
poetry shell
```

Or use `poetry run` to run commands directly:
```bash
poetry run python src/main.py --log sepsis.xes
```

### Step 3: Docker Setup

Install Docker if not already installed: https://docs.docker.com/get-docker/

Pull the SIMOD image:
```bash
docker pull nokal/simod:latest
```

Verify Docker works:
```bash
docker --version
docker run --rm hello-world
```

**Note**: If using `sudo docker`, ensure passwordless sudo for Docker:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Step 4: SAP Gen AI Hub Credentials

Create a `.env` file in the project root:
```bash
# .env
gen_ai_hub_api_key="your-api-key-here"
gen_ai_hub_api_version="2024-05-15"
gen_ai_hub_client_id="your-client-id"
gen_ai_hub_client_secret="your-client-secret"
gen_ai_hub_resource_group="default"
gen_ai_hub_auth_url="https://auth.example.com"
gen_ai_hub_api_url="https://api.example.com"
```

**Get credentials:**
1. Log in to SAP Gen AI Hub console
2. Create a new API client
3. Copy credentials to `.env`

⚠️ **Never commit `.env` to version control!** It's listed in `.gitignore`.

### Step 5: Event Logs (Optional)

Place event logs in the `logs/` directory:
```
logs/
├── sepsis.xes          # Example: Sepsis treatment process
├── road_traffic.xes    # Example: Traffic violation process
├── bpic_2012.xes       # Example: BPI Challenge 2012
└── (your logs here)
```

Supported formats:
- **XES** (.xes) - Standard format, auto-converted to CSV for SIMOD
- **CSV** (.csv) - Detected automatically, no conversion needed

## Quick Start

### Run with Default Settings

```bash
poetry run python src/main.py --log sepsis.xes
```

This will:
1. Discover baseline model from `logs/sepsis.xes`
2. Run 3 improvement iterations (each iteration generates a new hypothesis)
3. Simulate and validate each hypothesis
4. Generate reports in `reports/`

### Run with Custom Parameters

```bash
poetry run python src/main.py \
  --log sepsis.xes \
  --temperature 0.0 \
  --max_tokens 4096 \
  --max_iterations 5 \
  --model gpt-4.1 \
  --docker_image nokal/simod:latest
```

### CLI Options

```
--log LOG                    Event log filename (default: road_traffic.xes)
--model MODEL                LLM model (default: gpt-4.1)
--temperature TEMP           LLM temperature [0.0-1.0] (default: 0.0)
--max_tokens NUM             Max LLM response tokens (default: 4096)
--max_iterations NUM         Improvement iterations (default: 3, set 1 for single-shot)
--docker_image IMAGE         SIMOD Docker image (default: nokal/simod:latest)
--no_sudo                    Don't use sudo for Docker
```

### Examples

**Quick test with deterministic output:**
```bash
poetry run python src/main.py --log sepsis.xes --temperature 0.0 --max_iterations 2
```

**Extended analysis:**
```bash
poetry run python src/main.py --log bpic_2012.xes --max_iterations 10 --max_tokens 8192
```

**Without sudo (if configured):**
```bash
poetry run python src/main.py --log road_traffic.xes --no_sudo
```

## Outputs

### Reports Directory (`reports/`)

```
reports/
├── improvement_report_gpt-4.1_sepsis_0.0_4096_20260212_121344.txt
└── interaction_logs/
  └── interaction_log_sepsis_20260212_121344.txt
```

**Improvement Report** contains:
- Executive summary
- Baseline metrics (cycle time, throughput, waiting time, resource utilization)
- Validated improvements with ROI analysis
- Implementation roadmap

**Interaction Log** contains:
- Detailed LLM prompts and responses
- SIMOD discovery output (model keys, metrics)
- Simulation results for each hypothesis
- Validation comparisons

## How It Works

### 1. Baseline Discovery

```
Event Log (XES/CSV)
    ↓ (SIMOD via Docker)
Process Model + Prosimos Parameters
    ↓ (pm4py if Prosimos fails)
Real Metrics (cycle time, throughput, resources)
```

- **SIMOD Discovery** (30 min timeout): Discovers BPMN model + Prosimos simulation parameters
- **Fallback**: If SIMOD doesn't output Prosimos files, extracts metrics directly from event log using pm4py
- **Real Metrics**: Calculates actual baseline KPIs from event log timestamps

### 2. Hypothesis Generation

LLM analyzes baseline metrics and suggests improvements. This step is repeated
for each iteration so you can compare multiple alternatives:
- Resource allocation changes
- Activity reordering
- Waiting time reduction
- Cost optimization

**Parameters considered (from SIMOD + Prosimos output and log metrics):**
- **Process structure**: BPMN model and gateway branching probabilities
- **Arrival behavior**: arrival time distribution and calendar
- **Resources**: resource profiles (counts), resource calendars, and utilization
- **Activity performance**: task-resource duration distributions
- **Baseline KPIs**: cycle time, throughput, waiting time, total cost

### 3. Simulation & Validation

- Applies hypothesized changes to simulation model
- Runs Prosimos with modified parameters (5 min timeout)
- Compares baseline vs. improved metrics
- Validates statistical significance

### 4. Reporting

Generates executive report with:
- Baseline analysis
- Validated improvements
- ROI calculations
- Implementation roadmap

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Process Discovery | SIMOD (Docker) | Discovers models from logs |
| Simulation | Prosimos | Validates improvements |
| LLM | SAP Gen AI Hub (GPT-4.1) | Generates hypotheses |
| Event Log Analysis | pm4py | Extracts metrics from logs |
| Log Format | pm4py XES/Pandas | Handles XES and CSV logs |
| Configuration | YAML | SIMOD parameters |
| Reporting | Python strings | Executive reports |

## Configuration Details

### SIMOD Discovery
- **Timeout**: 30 minutes
- **Docker Image**: `nokal/simod:latest`
- **Output**: BPMN + Prosimos JSON (if available)
- **Fallback**: Real metric extraction from event log

### LLM (GPT-4.1)
- **Temperature**: 0.0 (deterministic)
- **Max Tokens**: 4096 (configurable)
- **Provider**: SAP Gen AI Hub SDK

### Prosimos Simulation
- **Timeout**: 5 minutes per simulation
- **Parameters**: Discovered from SIMOD or constructed from baseline metrics

## Reproducibility

All runs are fully reproducible:

✅ **Deterministic LLM Output**
- Temperature set to 0.0 by default
- Use `--temperature 0.0` for consistent results

✅ **Detailed Interaction Logs**
- Every LLM prompt and response captured
- Simulation parameters and results logged
- All metrics extractable

✅ **Version Control**
- Pin Python version (3.11+)
- Lock dependencies via Poetry
- Docker image: `nokal/simod:latest`

**Reproduce a run:**
```bash
# Extract parameters from interaction log
cat reports/interaction_logs/interaction_log_sepsis_20260212_121344.txt

# Re-run with same parameters
poetry run python src/main.py \
  --log sepsis.xes \
  --temperature 0.0 \
  --max_tokens 4096 \
  --max_iterations 3 \
  --model gpt-4.1
```

## Project Structure

### Core Modules

**`improvement_validation_pipeline.py`** (Main Orchestrator)
- Composes all modules into workflow
- Manages SIMOD discovery → LLM hypotheses → Simulation → Validation
- Handles interaction logging and report generation
- ~260 lines (refactored from 1,200+ line monolith)

**`simulator/simod_discoverer.py`**
- SIMOD Docker integration
- XES to CSV conversion (skips if already CSV)
- Real baseline metric extraction from event logs
- Fallback to mock only if extraction fails

**`simulator/prosimos_runner.py`**
- Parameterized simulation execution
- Timeout handling
- Output parsing

**`llm/hypothesis_generator.py`**
- GPT-4.1 integration via SAP Gen AI Hub SDK
- Baseline-aware improvement suggestion
- Structured JSON output parsing

**`validation/improvement_validator.py`**
- Statistical comparison (baseline vs. simulated)
- Significance testing
- Effect size calculation

**`reporting/report_generator.py`**
- Markdown-formatted executive reports
- KPI formatting and analysis
- ROI projections

**`metrics/metrics_extractor.py`**
- pm4py event log analysis
- Case statistics (cycle time, throughput)
- Activity and resource extraction

## Troubleshooting

### Docker Issues

**Error**: "Docker not found"
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Or verify it's in PATH
which docker
```

**Error**: "permission denied while trying to connect to Docker daemon"
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Restart Docker daemon
sudo systemctl restart docker
```

**Error**: "Image not found: nokal/simod:latest"
```bash
# Pull the image
docker pull nokal/simod:latest

# Verify
docker images | grep simod
```

### SAP Gen AI Hub Issues

**Error**: "TypeError: AICoreV2Client.__init__() missing argument"
- Verify `.env` file exists and has all required fields
- Check `load_dotenv()` is called before SDK imports

**Error**: "401 Unauthorized"
- Verify SAP credentials in `.env`
- Check API key hasn't expired

### SIMOD Discovery Issues

**Error**: "SIMOD discovery timeout"
- Timeout is 30 minutes (configurable in code)
- Large event logs may exceed timeout
- System falls back to real metric extraction from log

**Error**: "No Prosimos parameters found"
- Expected behavior - means SIMOD didn't produce Prosimos output
- System automatically extracts real metrics from event log instead
- Simulation phase may be skipped (fallback to baseline metrics)

### Memory Issues

If running out of memory:
1. Reduce `--max_iterations` (fewer hypothesis generations)
2. Use smaller event logs for testing
3. Increase system swap (if available)

## Performance Notes

- **First run**: Pulls SIMOD Docker image (~2-3 GB)
- **SIMOD discovery**: 10-30 minutes depending on log size
- **LLM generation**: ~10-30 seconds per hypothesis
- **Simulation**: ~1-5 minutes per hypothesis
- **Total runtime**: 2-4 hours for 3 iterations on medium logs

## Development

### Virtual Environment

Activate Poetry shell:
```bash
poetry shell
```

Or run commands with `poetry run`:
```bash
poetry run python script.py
```

### Adding Dependencies

Add to project:
```bash
poetry add package-name
```

Add as dev dependency:
```bash
poetry add --group dev package-name
```

### Code Quality

Check syntax:
```bash
python -m py_compile src/**/*.py
```

### Testing

Run main pipeline:
```bash
poetry run python src/main.py --log sepsis.xes --max_iterations 1
```

## Contributing

For reproducible contributions:
1. Test with temperature 0.0
2. Document findings in interaction logs
3. Update README with any new features
4. Use Poetry for dependency management

## License

See [LICENSE](LICENSE) file

## Authors

- Angelo Casciani (SAP)

## Support

For issues:
1. Check interaction logs in `reports/interaction_logs/`
2. Verify Docker is running: `docker ps`
3. Verify `.env` has credentials
4. Check event log format (XES or CSV)
5. See Troubleshooting section above

## Citation

If you use this project, please cite:

```bibtex
@software{agent_process_improvement,
  title={Process Improvement Validation Agent},
  author={Casciani, Angelo},
  year={2026},
  organization={SAP}
}
```

---

**Last Updated**: February 12, 2026  
**Python Version**: 3.11+  
**Dependencies**: See `pyproject.toml`