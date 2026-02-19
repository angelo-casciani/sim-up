# Process Improvement Validation Agent

Agentic pipeline for process-improvement discovery and validation using:
- PM4Py baseline extraction from event logs
- SIMOD/Prosimos model discovery and DES simulation
- optional AgentSimulator behavioral simulation
- iterative LLM hypothesis generation
- automated validation and report generation

## Documentation

- [Architecture](ARCHITECTURE_DIAGRAMS.md)
- [Validation Logic](VALIDATION_LOGIC.md)

## What is Working End-to-End

- Full multi-iteration pipeline execution completes successfully.
- Simulator-relative baseline validation is active.
- Dataset-specific calibrated model selection is active.
- Report generation is robust to `impact`/`improvement` payload variants.

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- Docker (for SIMOD)
- SAP Gen AI Hub credentials in `.env`

### Install

```bash
poetry install
```

### Run

```bash
poetry run python src/main.py --log P2P/P2P_train_preprocessed.xes --max_iterations 3
```

### Common Options

```bash
--log LOG
--model MODEL
--temperature TEMP
--max_tokens NUM
--max_iterations NUM
--docker_image IMAGE
--no_sudo
```

## Outputs

- Reports: `reports/improvement_report_*.txt`
- Interaction logs: `reports/interaction_logs/interaction_log_*.txt`

## Notes

- Calibration artifacts are loaded automatically when available.
- If a dataset-specific calibrated model exists, it is preferred over generic calibrated model files.
- Validation compares improvements against the simulated baseline from the same simulator context.

# Architecture

## High-Level Flow

```text
Event Log (XES/CSV)
  -> PM4Py baseline extraction
  -> SIMOD discovery / model assets
  -> calibrated model selection
  -> iteration loop:
       LLM hypothesis
       baseline simulation
       improved simulation
       validation
       reporting
  -> final report artifacts
```

## Runtime Components

- `src/improvement_validation_pipeline.py`
  - orchestrates full workflow
  - handles iteration lifecycle and artifact generation
- `src/simulator/simod_discoverer.py`
  - discovers process/simulation assets
- `src/simulator/prosimos_runner.py`
  - runs DES simulations
- `src/simulator/agent_simulator_runner.py`
  - runs optional ABS simulations
- `src/validation/improvement_validator.py`
  - compares simulator baseline vs simulator improvement
- `src/calibration/simod_calibrator.py`
  - applies calibration factors, including wrapped `prosimos_params` structures
- `src/reporting/report_generator.py`
  - generates final textual report with schema-tolerant improvement payload handling

## Model Selection Order

For a given dataset `<dataset>`:

1. `simod_model_<dataset>_calibrated.json`
2. `simod_model_calibrated.json`
3. discovered/raw model output

This ensures dataset-specific calibration is preferred when present.

## Validation Strategy

The pipeline uses **simulator-relative validation**:

- simulate baseline under current model assumptions
- simulate each hypothesis under the same assumptions
- validate directional improvement between those two simulated states

This avoids rejecting useful hypotheses solely because simulator absolute values differ from raw event-log reality.

## Reporting Contracts

Improvement payload compatibility is tolerant to either structure:

- `impact`
- `improvement`

Fallback logic prevents runtime report failures when one shape is absent.


# Validation Logic

This document defines the current end-to-end validation behavior used by the pipeline.

## Goal

Validate whether an LLM-generated improvement hypothesis is beneficial for a given event log by comparing:
- a simulated **baseline** (from the same simulator stack), and
- a simulated **improvement scenario** (with hypothesis-applied changes).

The comparison is intentionally **relative to simulator baseline** to avoid over-penalizing known absolute realism gaps between simulator output and raw event-log metrics.

## Execution Flow

1. Extract baseline process metrics from event log (PM4Py).
2. Discover/prepare simulation model (SIMOD/Prosimos-compatible assets).
3. Select calibrated simulation model with dataset-aware priority:
   - `simod_model_<dataset>_calibrated.json`
   - fallback: `simod_model_calibrated.json`
   - fallback: discovered/raw model
4. Simulate baseline with current simulator settings.
5. Generate one hypothesis per iteration.
6. Simulate improved scenario with hypothesis changes.
7. Validate improvement against simulated baseline.
8. Repeat until `max_iterations` is reached.

## Validation Basis

The validator compares improved metrics against simulator baseline metrics. Typical metrics include:
- cycle time
- throughput
- waiting time
- resource utilization

A hypothesis is accepted only when validation rules indicate a net positive change.

## Why Relative Baseline

Prosimos/ABS can be directionally valid while not perfectly matching absolute real-world values from logs. Using a simulator-relative baseline ensures:
- fair comparison under the same simulation assumptions,
- robust ranking of alternative hypotheses,
- reduced false negatives due to model realism mismatch.

## Data/Schema Robustness

Reporting supports both improvement payload shapes:
- `impact`
- `improvement`

If one is missing, the other is used as fallback to avoid report-generation failures.

## Current Practical Guarantees

- End-to-end pipeline runs complete successfully with multiple iterations.
- Dataset-specific calibrated model loading is active.
- Baseline simulation path is active (not bypassed).
- Report generation is tolerant to known payload schema variations.


## License

See [LICENSE](LICENSE).