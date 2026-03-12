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
- **Dual simulator execution:** Both DES and ABS simulators always run in parallel.
- **Averaged validation:** Results from both simulators are averaged for the final decision.
- **Validation formula:** `(DES_score + ABS_score) / 2 > 0.01` where score = `delta_tp - delta_ct`.
- **Utilization veto:** Improvements rejected if cumulative resource utilization > 90%.
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

Both simulators (DES and ABS) always run in parallel. The validation decision is based on the **averaged score** from both simulators:

```
averaged_score = (DES_score + ABS_score) / 2
```

Where each simulator's score is: `delta_throughput - delta_cycle_time`

| Simulator | Models | Reliable KPIs |
|---|---|---|
| **DES (Prosimos)** | BPMN + resource pools + calendars + queuing | `cycle_time`, `waiting_time`, `throughput` |
| **ABS (AgentSimulator)** | Agent behaviors + handover sequences | `cycle_time`, resource utilization |

**Note:** `activity_durations` (activity duration optimization) is disabled in the hypothesis generator as it identifies WHERE to improve but not HOW.

The pipeline focuses on actionable interventions with clear priority: **role_redesign** (preferred when workload imbalance exists) > **resource_allocation** > **calendar_adjustment**.

### Validation Formula

Both simulators compute a score using:
```
score = delta_throughput - delta_cycle_time
```

The final decision uses the **averaged score**:
```
averaged_score = (DES_score + ABS_score) / 2 > 0.01
```

Where:
- `delta_throughput`: positive = improvement (throughput increased)
- `delta_cycle_time`: negative = improvement (cycle time decreased)

**Example:** DES: +3% tp, -2% ct = 0.05 score; ABS: +1% tp, -4% ct = 0.05 score => averaged = 0.05 > 0.01 ✓

### Utilization Veto

An improvement is **automatically rejected** if the cumulative (average) resource utilization exceeds 90%, regardless of the formula score. This prevents improvements that overload resources.

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
       both simulators in parallel (DES + ABS)
       formula-based validation (averaged score > 0.01)
       utilization veto check (reject if avg util > 90%)
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
  - `validate_with_formula()`: validates using `delta_tp - delta_ct > 0.01` with utilization veto
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

The pipeline uses **dual simulator execution with averaged validation**:

- build one fixed DES baseline and one fixed ABS baseline at the start
- for each hypothesis, run **both simulators** in parallel
- compute score for each: `delta_throughput - delta_cycle_time`
- average the scores: `(DES_score + ABS_score) / 2`
- validate: averaged score > 0.01 (1% threshold)
- apply utilization veto: reject if either simulator's avg utilization > 90%

This approach:
- provides more robust validation by combining both simulator perspectives
- reduces noise from individual simulator quirks
- ensures both queue-based (DES) and behavioral (ABS) aspects are considered

# Validation Logic

## Goal

Validate whether an LLM-generated improvement hypothesis is beneficial for a given event log by comparing:
- a simulated **baseline** (from the same simulator stack), and
- a simulated **improvement scenario** (with hypothesis-applied changes).

The comparison is intentionally **relative to simulator baseline** to avoid over-penalizing known absolute realism gaps between simulator output and raw event-log metrics.

In practice, the validator answers one question per iteration:
"Under the same simulator assumptions, did this proposal produce an acceptable directional improvement?"

## Validation Formula

The core validation uses a unified formula:

```
score = delta_throughput - (delta_cycle_time / 2) > 0.02
```

This formula:
- Rewards throughput increases (positive delta_tp adds to score)
- Rewards cycle time decreases (negative delta_ct, when subtracted, adds to score)
- Requires a combined improvement of at least 2% to pass

### Utilization Veto

Even if the formula passes, improvements are **rejected** if:
- Cumulative (average) resource utilization > 90%

This prevents accepting improvements that would overload resources.

## Execution Flow

1. Extract baseline process metrics from event log (PM4Py).
2. Discover/prepare simulation model (SIMOD/Prosimos-compatible assets).
3. Select calibrated simulation model with dataset-aware priority.
4. Simulate baseline once for both simulators.
5. Generate one hypothesis per iteration.
6. **Simulate with both simulators** in parallel (DES + ABS).
7. Compute score for each: `delta_tp - delta_ct`.
8. Average scores: `(DES_score + ABS_score) / 2 > 0.01`.
8. Apply utilization veto if avg utilization > 90%.
9. If accepted, apply that change to the working scenario for subsequent iterations.
10. Repeat until `max_iterations` is reached.

Important distinction:
- **Baselines are fixed** during the whole run (for fair scoring).
- **Working scenario is cumulative** across accepted iterations (for realistic multi-step improvement building).

## Validation Basis

The validator compares improved metrics against simulator baseline metrics using:
- cycle time
- throughput
- resource utilization (for veto check)

Note: `waiting_time` reported by ABS is **always 0.0** because AgentSimulator has no queue model. ABS waiting_time is not used for validation decisions.

### Validation Thresholds

| Threshold | Value | Description |
|-----------|-------|-------------|
| Formula threshold | 0.01 (1%) | Averaged score must exceed this |
| Utilization veto | 0.90 (90%) | Reject if avg resource utilization exceeds this |

### Metric Direction

- **Lower is better:** cycle time, waiting time
- **Higher is better:** throughput

Resource utilization is used for the veto check — if average utilization exceeds 90%, the improvement is rejected regardless of other metrics.

### Per-Simulator Validation (what is scored)

Both simulators run for every hypothesis:
1. Each simulator computes: `score = delta_tp - delta_ct`
2. Scores are averaged: `(DES_score + ABS_score) / 2`
3. Averaged score must exceed 0.01 (1%)
4. Check utilization veto: reject if either simulator's avg utilization > 90%
5. Determine pass/fail based on averaged result

### Why Dual Simulator with Averaging

Running both simulators and averaging:
- **More robust:** Reduces noise from individual simulator quirks
- **Balanced view:** Combines queue-based (DES) and behavioral (ABS) perspectives
- **Fair validation:** Neither simulator dominates the decision

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
- Check the **averaged score** value (must be > 0.01 to pass).
- Check individual **DES score** and **ABS score** to understand each simulator's view.
- Check for **utilization veto** — if triggered, improvement is rejected regardless of score.
- For multi-iteration runs, accepted changes are cumulative in scenario state but still evaluated against the original fixed simulator baselines.

## License

See [LICENSE](LICENSE).
