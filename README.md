# Process Improvement Validation Agent

Agentic pipeline for process-improvement discovery and validation using:
- PM4Py baseline extraction from event logs
- SIMOD/Prosimos model discovery and DES simulation
- AgentSimulator behavioral simulation
- iterative LLM hypothesis generation
- automated change-type-aware validation and report generation

## Documentation

- Runtime and validation behavior are documented in this README.
- AgentSimulator internals and the local modification are documented in `src/AgentSimulator/README.md`.
- Simulator design and split policy are documented in `misc/SIMULATORS_README.md`.

## What is Working End-to-End

- Full multi-iteration pipeline execution completes successfully.
- Simulator-relative baseline validation is active (DES and ABS each compared against own baseline).
- Dataset-specific calibrated model selection is active.
- Report generation is robust to `impact`/`improvement` payload variants.
- Change-type-aware validation: different simulators act as oracle depending on hypothesis type.
- DES time horizon always constrained to the test-split window for comparable metrics.

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
- verify AgentSimulator import in the Poetry environment

### Run

```bash
poetry run python src/main.py --log P2P/train_preprocessed.csv --max_iterations 3
```

### Common Options

```bash
--log LOG
--model MODEL
--temperature TEMP
--max_tokens NUM
--max_iterations NUM
--no_sudo
--des_runs NUM
--abs_runs NUM
```

## Key Design Decisions

### Simulator Strengths

| Simulator | Models | Reliable KPIs | Not reliable |
|---|---|---|---|
| **DES (Prosimos)** | BPMN + resource pools + calendars + queuing | `cycle_time`, `waiting_time`, `throughput` | agent routing, handover patterns |
| **ABS (AgentSimulator)** | Agent behaviors + handover sequences | `cycle_time` (distribution distance), resource utilization | `waiting_time` (always 0 — no queue model) |

### Change-Type-Aware Validation Oracle

The validation oracle depends on which type of change is being evaluated:

| `parameter_type` | Primary oracle | Secondary role | Rationale |
|---|---|---|---|
| `role_redesign` | **ABS primary** | DES: throughput safety check | ABS explicitly models who does what; DES checks throughput does not regress |
| `resources` | **DES only** | ABS: informational only | Resource queuing only modelled in DES |

**Note:** `calendar` (calendar adjustment) and `activity_durations` (activity duration optimization) are disabled in the hypothesis generator:
- **Calendar:** Not yet implemented in ProsimosRunner (`_modify_calendar_hours` missing).
- **Activity durations:** Not actionable for process owners — identifies WHERE to improve but not HOW.

The pipeline focuses on actionable interventions with clear priority: **role_redesign** (preferred when workload imbalance exists) > **resource_allocation**.

For `resources`, ABS results are logged as informational context only and do not affect the accept/reject decision.

For `role_redesign`, if DES reports a throughput decrease > 10%, the proposal is rejected regardless of ABS outcome.

### DES Time Horizon

DES is always constrained to run within the test-split time window (the same window used by ABS). This ensures:
- DES throughput is a realistic case rate, not a fixed 1000-case budget
- Waiting time reflects the actual load level of the test period
- DES and ABS throughput figures are directly comparable

### Tracked KPIs

Three KPIs are used for validation and reporting:
- `cycle_time` (lower is better)
- `throughput` (higher is better)
- `waiting_time` (lower is better; note: ABS always reports 0.0 by design)

Resource utilization is tracked and logged but is not a primary validation criterion.

## Outputs

- Reports: `reports/improvement_report_*.txt`
- Interaction logs: `reports/interaction_logs/interaction_log_*.txt`

## DES vs ABS: Why Results Can Differ

Prosimos (DES) and AgentSimulator (ABS) are fundamentally different models:

- **Prosimos (DES)** runs on SIMOD-discovered BPMN + Prosimos JSON parameters. It models:
  - resource pools, calendars, and working-hour windows
  - explicit queuing and waiting time
  - throughput as a function of resource capacity
- **AgentSimulator (ABS)** re-learns behavior from the event log. It models:
  - agent-level handover and routing sequences
  - who does what activity and in what order
  - behavioural distribution distances (CTDD, REDD, AEDD) — lower = more faithful reproduction of the log
  - **no queue model**: `waiting_time` is always 0.0

Because of these structural differences, ABS cannot validate capacity or calendar changes (those affect queuing, which ABS does not simulate). ABS is the right oracle for role redesign, because it tracks agent routing sequences directly.

The pipeline uses change-type-aware oracles instead of a flat OR rule to match each hypothesis type to the simulator best equipped to evaluate it.

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
       improved simulation (DES + ABS in parallel)
       change-type-aware validation (_decide_improvement)
       cumulative scenario update (if validated)
       reporting
  -> final report artifacts
```

## Runtime Components

- `src/improvement_validation_pipeline.py`
  - orchestrates full workflow
  - handles iteration lifecycle and artifact generation
  - `_decide_improvement()`: change-type-aware oracle logic
- `src/simulator/simod_discoverer.py`
  - discovers process/simulation assets
- `src/simulator/prosimos_runner.py`
  - runs DES simulations (always time-horizon constrained)
- `src/simulator/agent_simulator_runner.py`
  - runs ABS simulations; applies case-level log modifications
- `src/validation/improvement_validator.py`
  - compares simulator baseline vs simulator improvement per-simulator
- `src/calibration/simod_calibrator.py`
  - applies calibration factors, including wrapped `prosimos_params` structures
- `src/reporting/report_generator.py`
  - generates final textual report with schema-tolerant improvement payload handling
- `src/llm/hypothesis_generator.py`
  - generates process improvement hypotheses; prompts include oracle guidance per change type

## Model Selection Order

For a given dataset `<dataset>`:

1. `simod_model_<dataset>_calibrated.json`
2. `simod_model_calibrated.json`
3. discovered/raw model output

This ensures dataset-specific calibration is preferred when present.

## Validation Strategy

The pipeline uses **change-type-aware, simulator-relative validation**:

- build one fixed DES baseline and one fixed ABS baseline at the start
- simulate each new hypothesis against the current working scenario
- compare each simulator output against its own fixed baseline
- apply hard guardrails (throughput non-decrease, bounded regressions, cycle-time cap)
- accept or reject using the oracle appropriate for the hypothesis `parameter_type`

This avoids:
- rejecting `role_redesign` hypotheses because DES cannot observe routing improvements
- accepting `resources`/`calendar` hypotheses based on ABS output (which ignores queuing)

# Validation Logic

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
3. Select calibrated simulation model with dataset-aware priority.
4. Simulate baseline once for both simulators.
5. Generate one hypothesis per iteration.
6. Simulate improved scenario with hypothesis changes (DES + ABS in parallel).
7. Validate DES and ABS independently against their fixed baselines.
8. Select oracle for combined decision based on `parameter_type`.
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
- resource utilization (informational)

Note: `waiting_time` reported by ABS is **always 0.0** because AgentSimulator has no queue model. ABS waiting_time is not used for validation decisions.

Guardrails enforce conservative acceptance:
- throughput must not decrease,
- cycle-time increase is capped at 10%,
- excessive regressions fail validation.

### Metric Direction

- **Lower is better:** cycle time, waiting time
- **Higher is better:** throughput

Resource utilization is interpreted contextually (e.g., avoiding overload concentration), not as a single universal min/max target.

### Per-Simulator Validation (what is scored)

For each simulator independently (DES and ABS):
1. Compare improved metrics vs that simulator's fixed baseline metrics.
2. Compute percentage deltas for key metrics.
3. Enforce guardrails (hard-fail checks).
4. Determine pass/fail for that simulator.

Then combine using the change-type oracle (see table above).

### Why Change-Type-Aware Oracles

A flat OR rule (old behaviour) would:
- allow ABS to "validate" resource additions it cannot model (ABS has no queue, so adding a resource cannot lower ABS waiting time)
- force ABS outcomes to veto role redesigns where DES is blind to routing improvements

The oracle table resolves this by matching the hypothesis to the simulator best equipped to evaluate it.

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
- DES time horizon always constrained to test split.
- `total_cost` is not reported (hardcoded dummy value, excluded from all KPI paths).

## Reading Validation Output

When reviewing logs or reports:
- Check `parameter_type` of the hypothesis to know which oracle was active.
- Check **DES impact vs DES baseline** and **ABS impact vs ABS baseline** separately.
- Check the `Decision oracle` line in the interaction log to see which rule applied.
- For `resources`/`calendar` hypotheses, ABS is shown as `[informational only]`.
- For `role_redesign` hypotheses, DES is shown as `[throughput safety check]`.
- For multi-iteration runs, accepted changes are cumulative in scenario state but still evaluated against the original fixed simulator baselines.

## License

See [LICENSE](LICENSE).
