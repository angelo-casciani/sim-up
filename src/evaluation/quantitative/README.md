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

### Convergence Criteria

Both search methods converge when the simulated KPI reaches the **optimal baseline level** (or better):
- Values **better than or equal to** optimal are accepted immediately
- Values **worse than** optimal must be within `tolerance_pct%` to be accepted

This ensures the search finds configurations that achieve equivalent (or superior) performance to the baseline.

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
| `--tolerance` | `5.0` | Tolerance % for KPI solution matching (worse only) |
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
    ├── LoanApp_convergence.png  # Convergence chart (log scale)
    ├── LoanApp_tradeoff.png     # Resource-Performance trade-off (Pareto view)
    ├── LoanApp_utilization.png  # Resource utilization analysis
    ├── aggregate_comparison.png # Bar chart: iterations comparison
    ├── convergence_overlay_cycle_time.png
    ├── convergence_overlay_throughput.png
    ├── convergence_overlay_waiting_time.png
    └── *.pdf                    # PDF versions for publication
```

## Publication-Quality Visualizations

The visualizer generates several chart types optimized for academic publications:

### 1. Convergence Plot (`*_convergence.png`)

Shows the race between Grid Search and LLM to find the solution:

- **Logarithmic Y-axis**: Clearly separates massive fault spike from nuanced improvements
- **Resource count annotations**: Each point shows how many resources were tried
- **LLM star marker**: Prominent gold star with horizontal target line
- **Publication fonts**: Larger axis labels (14pt) and titles (16pt)

### 2. Resource Trade-off Plot (`*_tradeoff.png`)

Visualizes the Pareto trade-off between resource allocation and performance:

- **X-axis**: Number of resources (e.g., Clerks)
- **Y-axis**: Cycle Time (log scale)
- **Shows**: Grid path (minimum viable), LLM jump (intelligent), Original state (baseline)
- **Use case**: Operations Research visualization for ESWA/IS journals

### 3. Resource Utilization Plot (`*_utilization.png`)

Shows how resource utilization changes as resources are added:

- **X-axis**: Number of resources
- **Y-axis**: Average utilization percentage
- **Color zones**:
  - **Green (70-85%)**: Optimal utilization zone
  - **Red (>85%)**: Over-utilized / bottleneck risk
  - **Yellow (<70%)**: Under-utilized / resource waste
- **Use case**: Demonstrates the trade-off between capacity and efficiency

### 4. Aggregate Comparison (`aggregate_comparison.png`)

Bar chart comparing iterations to solution across all logs:

- **Error bars**: Standard deviation when multiple experiments exist
- **Speedup annotations**: Shows speedup ratio between bars
- **Value labels**: Exact iteration counts on each bar

### 5. Convergence Overlay (`convergence_overlay_*.png`)

Overlaid curves for all experiments on a single metric:

- **Log scale**: For time-based metrics (cycle_time, waiting_time)
- **Resource annotations**: Numbers on points indicate resource count tried
- **Reference lines**: Optimal and faulty baselines

### Visualization Features

| Feature | Implementation | Purpose |
|---------|----------------|---------|
| Log Y-axis | `plt.yscale('log')` | Handle massive fault spikes |
| Resource annotations | Text labels on points | Show decision context |
| Star markers | Gold star for LLM | Visual prominence |
| Horizontal dashed line | From LLM solution | Target baseline reference |
| Colorblind palette | Vermillion, Bluish-green | Accessibility |
| Thick lines | `linewidth=2.5` | Print readability |
| Large fonts | 14-16pt | Journal figure scaling |
| Utilization zones | Colored axhspan | Show optimal/over/under-utilized |

## Results Interpretation

### Example Output

```
Aggregate Results:
  Avg Grid iterations: 3.00
  Avg LLM iterations:  1.00
  Avg Speedup ratio:   3.00x
  Grid success rate:   100.0%
  LLM success rate:    100.0%

Per-log results:
  LoanApp.csv: Grid=3, LLM=1, Speedup=3.00x
```

### What Makes Good Results for a Paper

1. **Meaningful fault**: Degradation should be significant (>100% cycle time increase)
2. **Clear speedup**: LLM should find solution in fewer iterations than grid search
3. **LLM rationale**: The LLM should correctly identify the bottleneck in its reasoning
4. **Trade-off visualization**: Shows efficient resource allocation vs over-provisioning

## Module Structure

```
src/evaluation/quantitative/
├── __init__.py
├── README.md                    # This file
├── run_evaluation.py            # CLI entry point
├── evaluation_runner.py         # Main orchestrator (5 DES runs averaged)
├── fault_injector.py            # Fault injection (pattern + activity-based grouping)
├── grid_search.py               # Systematic grid search
├── llm_search.py                # LLM-guided search (uses HypothesisGenerator)
└── results_visualizer.py        # Publication-quality charts
```

## How It Works

### 1. Resource Grouping (`fault_injector.py`)

Resources are grouped into roles for fault injection using two strategies:

**Pattern-based grouping** (preferred):
```python
# "Clerk-000001" to "Clerk-000008" → "Clerk" role with 8 resources
target_resource = "Clerk"
original_count = 8
```

**Activity-based grouping** (fallback):
```python
# Resources performing identical activities are grouped together
# e.g., Users 1-5 all do tasks {A, B, C} → "Group_A+B+C" role
```

**Role Selection Criteria**:
- PRIMARY: Prefer roles with the **highest resource count** (more dramatic fault effect)
- SECONDARY: Higher utilization as tie-breaker
- Minimum: Prefers roles with 3+ resources; falls back to 2+ if none found

### 2. Fault Injection

Reduces resources to create a bottleneck:
```python
fault_count = 1
# Result: Only Clerk-000001 remains, creating 99% utilization
```

### 3. Grid Search (`grid_search.py`)

Systematically tries resource counts from 1 until KPIs match (or beat) optimal:

```python
for count in range(1, max_count + 1):
    metrics = simulate(model, count)
    if metrics.cycle_time <= optimal.cycle_time:  # Better or equal
        return count
    if within_tolerance(metrics.cycle_time, optimal.cycle_time):
        return count  # Within tolerance
```

### 4. LLM Search (`llm_search.py`)

Uses the existing `HypothesisGenerator` for LLM calls, constrained to `resource_allocation` changes:

```python
# LLM sees metrics with 99% Clerk utilization
# LLM proposes: {"target": "Clerk", "change": "increase_count", "value": 4}
# Simulate, check improvement, iterate
```

### 5. Simulation Averaging

Each simulation call runs **5 DES replicas** and averages results to reduce variance:

```python
des_runs = 5  # Default, configurable
for run in range(des_runs):
    metrics = prosimos.simulate(model, modifications)
    results.append(metrics)
return average(results)
```

### 6. Fair Benchmarking with Simulation Cache

To ensure fair comparison between Grid Search and LLM Search, the evaluation uses a **simulation cache**:

**How It Works**:
```python
# Cache keyed by resource count
simulation_cache: Dict[int, Dict] = {}

def simulate_with_role_count(role_name: str, count: int) -> Dict:
    # Check cache first - ensures fair benchmarking
    if count in simulation_cache:
        print(f"[CACHE HIT] Using cached result for count={count}")
        return simulation_cache[count]  # Identical metrics guaranteed

    # Run simulation and cache the result
    result = run_simulation(model, count)
    simulation_cache[count] = result
    return result
```

**Why This Matters**:
- DES simulators have inherent randomness (arrival times, service times)
- Without caching, Grid testing count=3 might get cycle_time=75,000s
- While LLM testing count=3 might get cycle_time=78,000s (different random draws)
- This variance could unfairly favor one method over the other
- The cache ensures: **identical inputs → identical outputs**

**Example**:
1. Grid Search tests count=1 → runs simulation → caches result
2. Grid Search tests count=2 → runs simulation → caches result
3. Grid Search tests count=3 → runs simulation → caches result (**solution found**)
4. LLM Search tests count=4 → runs simulation → caches result (**solution found in 1 iteration**)

If LLM had also proposed count=3, it would get the **exact same cached result** as Grid Search.

## Paper Argument

> "While systematic search guarantees optimality eventually, it is computationally
> intractable for complex processes. The LLM acts as a **Semantic Heuristic**,
> pruning the search space by leveraging domain knowledge encoded in its training."

The experimental results demonstrate that the LLM achieves equivalent solutions with significantly fewer simulation calls by:
1. Recognizing bottleneck patterns (high utilization, high waiting time)
2. Making informed jumps rather than incremental exploration
3. Learning from previous attempts within the same search

## Supported Event Logs

| Log | Resources | Grouping Method | Status |
|-----|-----------|-----------------|--------|
| LoanApp.csv | 19 | Pattern-based (Clerk, Loan Officer, etc.) | Supported |
| cvs_pharmacy.csv | 8 | Pattern-based (Technician, Pharmacist) | Supported |
| Confidential_1000.csv | 26 | Pattern-based (ABC, DEF, etc.) | Supported |
| Confidential_2000.csv | 26 | Pattern-based | Supported |
| p2p.csv | 27 | Activity-based (6 groups) | Supported |
| ACR.csv | 511 | Activity-based (20 groups) | Supported |
| BPIC_2012W.csv | 62 | Activity-based (5 groups) | Supported |
| BPIC_2017W.csv | 148 | Activity-based (15 groups) | Supported |
| Production.csv | 41 | Activity-based (7 groups) | Supported |
