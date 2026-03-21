# Process Improvement Validation Agent

An agentic pipeline for process-improvement discovery and validation, combining PM4Py baseline extraction, SIMOD/Prosimos DES simulation, AgentSimulator behavioral simulation, and iterative LLM hypothesis generation.

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- Docker (for SIMOD)
- SAP Gen AI Hub credentials in `.env`

### Install & Run

```bash
./setup.sh
poetry run python src/main.py --log P2P/train_preprocessed.csv --max_iterations 3
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
| `--no_sudo` | Run Docker without sudo |
| `--des_runs` | DES simulation runs (default: 10) |
| `--abs_runs` | ABS simulation runs (default: 10) |

</details>

## Architecture

```text
Event Log (XES/CSV)
  -> PM4Py baseline extraction
  -> SIMOD discovery / model assets
  -> One-time simulator baselines (DES + ABS)
  -> Iteration loop:
       LLM hypothesis generation
       Parallel simulation (DES + ABS)
       Averaged validation (score > 0.01)
       Utilization veto (reject if > 90%)
       Cumulative scenario update (if validated)
  -> Final report artifacts
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `src/improvement_validation_pipeline.py` | Orchestrates workflow and iteration lifecycle |
| `src/simulator/simod_discoverer.py` | Discovers process/simulation assets |
| `src/simulator/prosimos_runner.py` | Runs DES simulations |
| `src/simulator/agent_simulator_runner.py` | Runs ABS simulations |
| `src/validation/improvement_validator.py` | Formula-based validation with utilization veto |
| `src/llm/hypothesis_generator.py` | Generates improvement hypotheses |
| `src/reporting/report_generator.py` | Generates final reports |

## Simulators

The pipeline runs **both simulators in parallel** for every hypothesis, averaging their results for validation.

| Simulator | What it Models | Reliable KPIs | Limitations |
|-----------|----------------|---------------|-------------|
| **DES (Prosimos)** | BPMN, resource pools, calendars, queuing | cycle_time, waiting_time, throughput | No agent routing/handover |
| **ABS (AgentSimulator)** | Agent behaviors, handover sequences | cycle_time, resource utilization | No queue model (waiting_time = 0) |

**Why both?** DES captures queue-based capacity effects; ABS captures behavioral patterns. Averaging provides robust validation that neither simulator alone can achieve.

## Validation

### Formula

Each simulator computes a score, then scores are averaged:

```
score = delta_throughput - delta_cycle_time
averaged_score = (DES_score + ABS_score) / 2
```

**Pass condition:** `averaged_score > 0.01` (1% improvement threshold)

**Example:** DES (+3% tp, -2% ct) = 0.05; ABS (+1% tp, -4% ct) = 0.05 => averaged = 0.05 > 0.01 ✓

### Utilization Veto

Improvements are **rejected** if average resource utilization exceeds 90%, regardless of score. This prevents overloading resources.

### Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Formula | 0.01 (1%) | Minimum averaged score to pass |
| Utilization veto | 0.90 (90%) | Maximum allowed resource utilization |

### Metric Directions

- **Lower is better:** cycle_time, waiting_time
- **Higher is better:** throughput

### Why Relative Baselines?

Simulators may not perfectly match absolute real-world values. Comparing improvements against simulator-specific baselines ensures fair evaluation under consistent assumptions.

## Hypothesis Priority

The pipeline focuses on actionable interventions:

1. **shift_responsibility** (preferred when workload imbalance exists)
2. **resource_allocation**
3. **calendar_adjustment**

Note: `activity_durations` is disabled as it identifies WHERE to improve but not HOW.

### Role Compatibility

Shift responsibility proposals (shifting responsibilities between roles) are validated for compatibility. A shift is **rejected** if:

- The source or destination is an external actor (e.g., Applicant, Customer)
- The roles have no overlapping tasks in the `role_task_mapping`
- The roles are different types with incompatible task assignments

Valid shifts require roles to share common tasks or be instances of the same role type (e.g., Loan Officer-001 → Loan Officer-002).

## Outputs

| Output | Location |
|--------|----------|
| Reports | `reports/improvement_report_*.txt` |
| Interaction logs | `reports/interaction_logs/interaction_log_*.txt` |

## License

See [LICENSE](LICENSE).
