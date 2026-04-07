# Process Improvement Agent

An agentic iterative Large Language Model (LLM)-based pipeline for process-improvement generation and validation, combining SIMOD and PM4Py baseline simulation models extraction, and a two-fold simulation for the improvement hypothesis validation, supporting Discrete Event Simulation (DES) simulation via Prosimos and agent-based simulation (ABS) via AgentSimulator.

## Project Structure

```
└── improvement-agent/
    ├── README.md
    ├── LICENSE
    ├── pyproject.toml                            # Poetry dependencies
    ├── setup.sh                                  # Automated setup script 
    ├── logs/                                     # Event logs for testing
    ├── src/
    │   ├── gui.py                                # Gradio web interface
    │   ├── improvement_validation_pipeline.py    # Main orchestration pipeline
    │   ├── main.py                               # CLI entry point
    │   ├── AgentSimulator/                       # AgentSimulator source code
    │   ├── calibration/                          
    │   │   └── simod_calibrator.py               # SIMOD model calibration
    │   ├── evaluation/
    │   │   ├── __init__.py
    │   │   ├── qualitative/                      # Qualitative evaluation setup scripts
    │   │   │   └── ...
    │   │   └── quantitative/                     # Quantitative evaluation setup scripts
    │   │       └── ...
    │   ├── llm/
    │   │   ├── __init__.py
    │   │   └── hypothesis_generator.py                  # Improvement hypotheses generation via LLM
    │   ├── reporting/
    │   │   ├── __init__.py
    │   │   └── report_generator.py                      # Improvement report generation
    │   ├── simulator/
    │   │   ├── __init__.py
    │   │   ├── simod_discoverer.py                      # SIMOD simulation model discovery
    │   │   ├── prosimos_runner.py                       # DES simulation via Prosimos
    │   │   ├── agent_simulator_runner.py                # ABS simulation via AgentSimulator
    │   │   └── metrics_extractor.py                     # KPI extraction utilities
    │   ├── utils/                                       # Shared utilities
    │   │   ├── __init__.py
    │   │   └── column_utils.py
    │   └── validation/
    │       ├── __init__.py
    │       └── improvement_validator.py                 # Improvement validation logic
```

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- Docker
- SAP Gen AI Hub credentials in `.env`

### Install & Run

Clone the repository locally:
```bash
https://github.com/angelo-casciani/improvement-agent
```

Run the automated setup script:
```bash
./setup.sh
poetry run python src/main.py --log p2p.csv --max_iterations 3
```

<details>
<summary>CLI Options</summary>

| Option | Description |
|--------|-------------|
| `--log` | Input event log (CSV/XES) |
| `--model` | LLM model identifier |
| `--temperature` | LLM temperature |
| `--max_tokens` | Max tokens per LLM call |
| `--max_iterations` | Number of improvement iterations |
| `--no_sudo` | Run Docker without sudo (needed for Windows) |
| `--des_runs` | DES simulation runs (default: 10) |
| `--abs_runs` | ABS simulation runs (default: 10) |

</details>

### Outputs

| Output | Location |
|--------|----------|
| Reports | `reports/improvement_report_*.txt` |
| Interaction logs | `reports/interaction_logs/interaction_log_*.txt` |

## Web GUI

A Gradio-based web interface is available for interactive use:

```bash
poetry run python src/gui.py
```

This launches a local web server (default: http://localhost:7860) with:

- **Event log selection**: Dropdown to select event logs from the `logs/` directory
- **LLM selection**: Choose from supported LLMs (GPT-4.1, Claude, Mistral, etc.)
- **Temperature**: Slider to control LLM creativity
- **Max iterations**: Number of improvement cycles to attempt
- **Report display**: Final improvement report shown on completion

The GUI runs the same pipeline as the CLI but with a user-friendly interface.

## Architecture

```text
Event Log (XES/CSV)
  -> PM4Py baseline KPIs extraction
  -> SIMOD simulation model discovery
  -> Iteration loop:
       LLM hypothesis generation
       Parallel simulation (DES + ABS)
       Averaged validation (KPI score > 0.01)
       Utilization veto (reject if overall resource utilization > 90%)
       Scenario update (if the improvement hypothesis is validated)
  -> Improvement report generation
```

## Simulators

The pipeline runs **both simulators in parallel** for every LLM-generated hypothesis, averaging their results for its validation. In this way, DES captures queue-based capacity effects and ABS captures behavioral patterns, and averaging provides a more robust validation that neither simulator alone can achieve.

| Simulator | What it Models | Reliable KPIs | Limitations |
|-----------|----------------|---------------|-------------|
| **DES (Prosimos)** | BPMN, resource pools, calendars, queuing | cycle_time, waiting_time, throughput | No agent routing/handover |
| **ABS (AgentSimulator)** | Agent behaviors, handover sequences | cycle_time, resource utilization | No queue model (waiting_time = 0) |

## Validation

Each simulator computes a score, then scores are averaged:

```
score = delta_throughput - delta_cycle_time
averaged_score = (DES_score + ABS_score) / 2
```

**Pass condition:** `averaged_score > 0.01` (1% improvement threshold)

Improvements are **rejected** if average resource utilization exceeds 90%, regardless of score to prevent overloading resources.

## Hypothesis Priority

The pipeline focuses on actionable interventions on resources:

1. *shift responsibility* between resources with the same role (preferred when workload imbalance exists);
2. *resource allocation*, hiring new resources with a specific profile; and
3. *calendar updates* for specific resources.

## License

See [LICENSE](LICENSE).
