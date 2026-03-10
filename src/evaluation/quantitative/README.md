# Quantitative Evaluation: LLM vs Grid Search

This module provides a controlled experimental framework to compare **LLM-guided optimization** against **systematic grid search** for process improvement.

## Motivation

The goal is to demonstrate that an LLM can act as a **"semantic heuristic"** that uses contextual clues (queue times, resource utilization) to find optimal configurations faster than exhaustive search.

## Experimental Setup

### The Race

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXPERIMENTAL DESIGN                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Discover M_opt (optimal model) with SIMOD              │
│          └── Simulate to get baseline KPIs                      │
│                                                                 │
│  Step 2: Inject Fault → M_fault                                 │
│          └── Reduce resource role from N to 1                   │
│          └── Creates severe bottleneck (e.g., 99% utilization)  │
│                                                                 │
│  Step 3: The Race                                               │
│          ┌─────────────────┐    ┌─────────────────┐             │
│          │  Grid Search    │    │   LLM Agent     │             │
│          │  (Contender A)  │    │  (Contender B)  │             │
│          ├─────────────────┤    ├─────────────────┤             │
│          │ Try count=1     │    │ See 99% util    │             │
│          │ Try count=2     │    │ → Propose +3    │             │
│          │ Try count=3     │    │ See improvement │             │
│          │ ...             │    │ → Propose +2    │             │
│          │ Try count=N     │    │ SOLUTION!       │             │
│          │ SOLUTION!       │    │                 │             │
│          └─────────────────┘    └─────────────────┘             │
│                                                                 │
│  Metric: Number of simulation calls to find the fix             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight

- **Grid Search**: Must traverse the entire search space (1 to N)
- **LLM Agent**: Uses semantic signals to "jump" toward the solution
  - High utilization (99%) → Need more resources
  - High waiting time → Bottleneck identified
  - Previous attempts → Learn and adjust

## Usage

### Basic Run

```bash
# Run on a single log
poetry run python src/evaluation/quantitative/run_evaluation.py \
    --model anthropic--claude-4.5-opus \
    --logs LoanApp.csv

# Run on multiple logs
poetry run python src/evaluation/quantitative/run_evaluation.py \
    --model gpt-4.1 \
    --logs LoanApp.csv ACR.csv Production.csv

# Run on all available logs (excluding o2c)
poetry run python src/evaluation/quantitative/run_evaluation.py \
    --model gpt-4.1
```

### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gpt-4.1` | LLM model for guided search |
| `--temperature` | `0.0` | LLM sampling temperature |
| `--grid-min` | `1` | Minimum resource count for grid search |
| `--grid-max` | `10` | Maximum resource count for grid search |
| `--fault-count` | `1` | Resource count to inject as fault |
| `--tolerance` | `2.0` | Tolerance % for KPI solution matching |
| `--max-llm-iters` | `10` | Maximum LLM search iterations |
| `--logs` | all | Specific log files to evaluate |
| `--no-sudo` | auto | Disable sudo for Docker (auto-detected on Windows) |
| `--visualize-only` | - | Path to existing summary.json for visualization only |

### Visualization Only

```bash
# Re-generate plots from existing results
poetry run python src/evaluation/quantitative/run_evaluation.py \
    --visualize-only reports/evaluation/quantitative_eval_20260310_094749/evaluation_summary.json
```

## Output Structure

```
reports/evaluation/quantitative_eval_YYYYMMDD_HHMMSS/
├── evaluation_summary.json      # Full results for all logs
├── result_LoanApp.json          # Individual log result
├── result_ACR.json
├── ...
├── LoanApp/                     # Working directory per log
│   ├── baseline/                # SIMOD discovery output
│   └── ...
└── figures/
    ├── LoanApp_convergence.png  # Per-log convergence chart
    ├── LoanApp_convergence.pdf
    ├── aggregate_comparison.png # Bar chart: iterations comparison
    ├── aggregate_comparison.pdf
    ├── convergence_overlay.png  # All experiments overlaid
    └── convergence_overlay.pdf
```

## Results Interpretation

### Example Output

```
Aggregate Results:
  Avg Grid iterations: 6.00
  Avg LLM iterations:  2.00
  Avg Speedup ratio:   3.00x
  Grid success rate:   100.0%
  LLM success rate:    100.0%

Per-log results:
  LoanApp.csv: Grid=6, LLM=2, Speedup=3.00x
```

### Convergence Chart

The convergence chart shows:
- **X-axis**: Number of simulation calls (iterations)
- **Y-axis**: Cycle time (seconds)
- **Red line**: Grid search trajectory (slow, stepped descent)
- **Green line**: LLM agent trajectory (fast jump to solution)
- **Blue dashed**: Optimal target
- **Gray dotted**: Faulty baseline

### What Makes Good Results for a Paper

1. **Meaningful fault**: Degradation should be significant (>100% cycle time increase)
2. **Clear speedup**: LLM should find solution in fewer iterations than grid search
3. **LLM rationale**: The LLM should correctly identify the bottleneck in its reasoning

## Module Structure

```
src/evaluation/quantitative/
├── __init__.py
├── README.md                    # This file
├── run_evaluation.py            # CLI entry point
├── evaluation_runner.py         # Main orchestrator
├── fault_injector.py            # Fault injection logic
├── grid_search.py               # Systematic grid search
├── llm_search.py                # LLM-guided search (uses HypothesisGenerator)
└── results_visualizer.py        # Publication-quality charts
```

## How It Works

### 1. Fault Injection (`fault_injector.py`)

Groups resources by role (e.g., "Clerk-000001" to "Clerk-000008" → "Clerk" role with 8 resources) and removes profiles to create a bottleneck.

```python
# Example: Reduce Clerks from 8 to 1
target_resource = "Clerk"
original_count = 8
fault_count = 1
# Result: Only Clerk-000001 remains, creating 99% utilization
```

### 2. Grid Search (`grid_search.py`)

Systematically tries resource counts from 1 until KPIs match optimal (within tolerance):

```python
for count in range(1, max_count + 1):
    metrics = simulate(model, count)
    if within_tolerance(metrics.cycle_time, optimal.cycle_time):
        return count  # Found solution, stop searching
```

### 3. LLM Search (`llm_search.py`)

Uses the existing `HypothesisGenerator` from the main pipeline for LLM calls.
Constrained to only propose `resource_allocation` changes:

```python
# LLM sees metrics with 99% Clerk utilization
# LLM proposes: {"target": "Clerk", "change": "increase_count", "value": 4}
# Simulate, check improvement, iterate
```

### 4. Visualization (`results_visualizer.py`)

Generates publication-quality charts:
- Individual convergence plots per log
- Aggregate bar chart comparison
- Overlaid convergence patterns

## Paper Argument

> "While systematic search guarantees optimality eventually, it is computationally
> intractable for complex processes. The LLM acts as a **Semantic Heuristic**,
> pruning the search space by leveraging domain knowledge encoded in its training."

The experimental results demonstrate that the LLM achieves equivalent solutions with significantly fewer simulation calls by:
1. Recognizing bottleneck patterns (high utilization, high waiting time)
2. Making informed jumps rather than incremental exploration
3. Learning from previous attempts within the same search
