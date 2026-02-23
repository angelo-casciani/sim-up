# Process Improvement Validation Agent

Agentic pipeline for process-improvement discovery and validation using:
- PM4Py baseline extraction from event logs
- SIMOD/Prosimos model discovery and DES simulation
- AgentSimulator behavioral simulation
- iterative LLM hypothesis generation
- automated validation and report generation

## Documentation

- Runtime and validation behavior are documented in this README.
- AgentSimulator internals and native commands are in `src/AgentSimulator/README.md`.

## What is Working End-to-End

- Full multi-iteration pipeline execution completes successfully.
- Simulator-relative baseline validation is active (DES and ABS each compared against own baseline).
- Dataset-specific calibrated model selection is active.
- Report generation is robust to `impact`/`improvement` payload variants.
- Validation accepts an improvement when DES **or** ABS passes guardrails (OR decision rule).

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- Docker (for SIMOD)
- SAP Gen AI Hub credentials in `.env`

### Install

```bash
./setup.sh
```

The setup script performs all required bootstrap operations:
- enforce Poetry environment on Python 3.11
- install Poetry dependencies
- ensure SIMOD Docker image (`nokal/simod:latest`) is available
- clone AgentSimulator into `src/AgentSimulator` (if missing)
- verify AgentSimulator import in the Poetry environment

If you keep AgentSimulator in a different location, set:

```bash
export AGENT_SIMULATOR_PATH=/absolute/path/to/AgentSimulator
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
--des_runs NUM
--abs_runs NUM
```

## Simulator Notes

- DES uses SIMOD-discovered BPMN + Prosimos parameters.
- ABS uses AgentSimulator on CSV logs (with hypothesis-driven temporary log edits where supported).
- DES metrics are horizon-filtered to the test split window; ABS runs full-log simulation by default.

## Outputs

- Reports: `reports/improvement_report_*.txt`
- Interaction logs: `reports/interaction_logs/interaction_log_*.txt`

## Notes

- Calibration artifacts are loaded automatically when available.
- If a dataset-specific calibrated model exists, it is preferred over generic calibrated model files.
- Validation compares improvements against the simulated baseline from the same simulator context.

## DES vs ABS: Why Results Can Differ

Prosimos (DES) and AgentSimulator (ABS) are intentionally different models:

- **Prosimos (DES)** runs on SIMOD-discovered BPMN + Prosimos JSON parameters.
- **AgentSimulator (ABS)** re-simulates behavior from the event log with agent-based logic.
- They may share log source and horizon filtering, but they do **not** use the exact same internal state-transition and timing model.

Because of this, absolute values can diverge. The pipeline evaluates impact **per simulator** and validates improvements using simulator-relative baselines:

1. Run **DES baseline** and **ABS baseline**.
2. Apply LLM changes.
3. Run **DES improved** and **ABS improved**.
4. Compute impact separately:
  - DES improved vs DES baseline
  - ABS improved vs ABS baseline
5. Validate each simulator independently with guardrails, then accept if DES **or** ABS validates.

This makes comparison fairer and easier to debug when DES/ABS disagree.

# Architecture

## High-Level Flow

```text
Event Log (XES/CSV)
  -> PM4Py baseline extraction
  -> SIMOD discovery / model assets
  -> one-time simulator baselines (DES + ABS)
  -> calibrated model selection
  -> iteration loop:
       LLM hypothesis
       improved simulation
       validation
    cumulative scenario update (if validated)
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

- build one fixed DES baseline and one fixed ABS baseline at the start
- simulate each new hypothesis against the current working scenario
- compare each simulator output against its own fixed baseline
- apply hard guardrails (throughput non-decrease, bounded regressions, cycle-time cap)
- accept iteration if **DES or ABS** validates (OR rule)

This avoids rejecting useful hypotheses solely because simulator absolute values differ from raw event-log reality.

## Reporting Contracts

Improvement payload compatibility is tolerant to either structure:

- `impact`
- `improvement`

Fallback logic prevents runtime report failures when one shape is absent.


# Validation Logic

Current end-to-end validation behavior used by the pipeline.

## Goal

Validate whether an LLM-generated improvement hypothesis is beneficial for a given event log by comparing:
- a simulated **baseline** (from the same simulator stack), and
- a simulated **improvement scenario** (with hypothesis-applied changes).

The comparison is intentionally **relative to simulator baseline** to avoid over-penalizing known absolute realism gaps between simulator output and raw event-log metrics.

In practice, the validator answers one question per iteration:
"Under the same simulator assumptions, did this proposal produce an acceptable directional improvement?"

## Execution Flow

1. Extract baseline process metrics from event log (PM4Py).
2. Discover/prepare simulation model (SIMOD/Prosimos-compatible assets).
3. Select calibrated simulation model with dataset-aware priority:
   - `simod_model_<dataset>_calibrated.json`
   - fallback: `simod_model_calibrated.json`
   - fallback: discovered/raw model
4. Simulate baseline once for both simulators with current settings.
5. Generate one hypothesis per iteration.
6. Simulate improved scenario with hypothesis changes.
7. Validate DES and ABS independently against their fixed baselines.
8. Accept iteration when DES or ABS passes validation guardrails.
9. If accepted, apply that change to the working scenario for subsequent iterations.
10. Repeat until `max_iterations` is reached.

Important distinction:
- **Baselines are fixed** during the whole run (for fair scoring).
- **Working scenario is cumulative** across accepted iterations (for realistic multi-step improvement building).

## Validation Basis

The validator compares improved metrics against simulator baseline metrics using:
- cycle time
- throughput
- waiting time
- resource utilization

Guardrails enforce conservative acceptance:
- throughput must not decrease,
- cycle-time increase is capped,
- excessive regressions fail validation.

Final decision rule is OR: if DES validates or ABS validates, the proposal is accepted.

### Metric Direction

- **Lower is better:** cycle time, waiting time, cost
- **Higher is better:** throughput

Resource utilization is interpreted contextually (e.g., avoiding overload concentration), not as a single universal min/max target.

### Per-Simulator Validation (what is scored)

For each simulator independently (DES and ABS):
1. Compare improved metrics vs that simulator's fixed baseline metrics.
2. Compute percentage deltas for key metrics.
3. Enforce guardrails (hard-fail checks).
4. Determine pass/fail for that simulator.

Then combine with:
- **Overall pass = DES pass OR ABS pass**

This avoids dropping potentially valuable proposals when one simulator captures a benefit the other does not.

### Why OR + Guardrails

OR keeps the system exploratory across two different modeling paradigms (DES structural and ABS behavioral), while guardrails prevent clearly harmful proposals from slipping through.

Example interpretation:
- DES improves throughput with bounded regressions, ABS is neutral/slightly worse -> can still pass.
- ABS improves cycle time, DES is neutral/slightly worse -> can still pass.
- A simulator violates hard guardrails (e.g., large cycle-time increase) -> that simulator fails regardless of other gains.

## Why Relative Baseline

Prosimos/ABS can be directionally valid while not perfectly matching absolute real-world values from logs. Using a simulator-relative baseline ensures:
- fair comparison under the same simulation assumptions,
- robust ranking of alternative hypotheses,
- reduced false negatives due to model realism mismatch.

This is especially important because DES and ABS are calibrated differently and may produce different absolute scales.

## Data/Schema Robustness

Reporting supports both improvement payload shapes:
- `impact`
- `improvement`

If one is missing, the other is used as fallback to avoid report-generation failures.

## Current Practical Guarantees

- End-to-end pipeline runs complete successfully with multiple iterations.
- Dataset-specific calibrated model loading is active.
- Baseline simulation path is active (not bypassed).
- Defaults are `des_runs=10` and `abs_runs=10` for more stable averages.
- Report generation is tolerant to known payload schema variations.

## Reading Validation Output

When reviewing logs or reports:
- Check **DES impact vs DES baseline** and **ABS impact vs ABS baseline** separately.
- Check both simulator-specific validation outcomes.
- Check final decision rule outcome (`OR (DES or ABS)`).
- For multi-iteration runs, remember accepted changes are cumulative in scenario state, but still evaluated against the original fixed simulator baselines.


## License

See [LICENSE](LICENSE).