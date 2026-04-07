# Quantitative Evaluation

This evaluation assesses the agentic process improvement framework through two main evaluation types:

1. **LLM Benchmark** - Compares multiple LLM models on the improvement pipeline
2. **Improvement Type Evaluation** - Compares LLM-guided search vs grid search across improvement types

## Improvement Types

The framework supports three improvement types:
- **Resource Allocation** - Adding resources to bottleneck roles
- **Calendar Adjustment** - Extending working hours for constrained resources
- **Shift Responsibility** - Redistributing workload between roles

## LLM Benchmark Evaluation

Benchmarks multiple LLM models on the `ImprovementValidationAgent` pipeline measuring:
- **Speed**: Time and iterations to find a validated improvement
- **Impact**: Quality of improvement (averaged validation score)

### Models Tested

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4.1, gpt-5, gpt-5-mini, gpt-5-nano, o3 |
| Anthropic | claude-4.5-opus, claude-4.5-sonnet, claude-4.5-haiku |
| Mistral | mistral-small-instruct, mistral-large-instruct |

### Usage

```bash
# Run benchmark with defaults
python src/evaluation/quantitative/run_llm_benchmark.py --log LoanApp.csv

# Run with specific models
python src/evaluation/quantitative/run_llm_benchmark.py \
    --models gpt-5 anthropic--claude-4.5-opus o3 --log LoanApp.csv

# Run with multiple repetitions
python src/evaluation/quantitative/run_llm_benchmark.py --log LoanApp.csv --repetitions 3

# Resume an interrupted benchmark
python src/evaluation/quantitative/run_llm_benchmark.py --resume --log LoanApp.csv

# Check checkpoint status
python src/evaluation/quantitative/run_llm_benchmark.py --status --log LoanApp.csv

# Clear checkpoint and start fresh
python src/evaluation/quantitative/run_llm_benchmark.py --clear-checkpoint --log LoanApp.csv
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--log`, `-l` | `LoanApp.csv` | Event log file in logs/ directory |
| `--models`, `-m` | All 10 models | Specific models to benchmark |
| `--repetitions`, `-r` | `1` | Number of runs per model |
| `--max-iterations` | `5` | Maximum hypothesis iterations per run |
| `--output-dir`, `-o` | `output` | Output directory for results |
| `--des-runs` | `5` | Number of DES (Prosimos) runs |
| `--abs-runs` | `5` | Number of ABS (AgentSimulator) runs |
| `--resume` | - | Resume from checkpoint |
| `--status` | - | Show checkpoint status |
| `--clear-checkpoint` | - | Clear existing checkpoint |
| `--dry-run` | - | Print configuration without running |

### Output

```
output/llm_benchmark_YYYYMMDD_HHMMSS/
├── benchmark_results.json      # Full results with per-iteration data
├── benchmark_results.csv       # Spreadsheet export
├── benchmark_report.md         # Summary with rankings
└── figures/                    # Visualization charts
```

## Event Logs

| Log | Domain |
|-----|--------|
| LoanApp.csv | Financial |
| BPIC_2012W.csv | Banking |
| BPIC_2017W2.csv | Banking |
| processed_o2c.csv | Order-to-Cash |

---

## Improvement Type Evaluation (LLM vs Grid Search)

Compares LLM-guided search against systematic grid search using fault injection methodology.

Selected LLM: claude-4.5-opus.

### Methodology

1. **Discover** baseline model from event log using SIMOD
2. **Inject fault** to create degraded model
3. **Run LLM search** to recover from fault
4. **Run grid search** on same fault
5. **Compare** iterations needed to converge

### Convergence Criterion

An improvement is considered **converged** when the primary KPI improves by at least the tolerance threshold.

| Improvement Type | Primary KPI | Tolerance | Rationale |
|------------------|-------------|-----------|-----------|
| Resource Allocation | waiting_time | 30% | Higher tolerance due to simulation variance from resource changes |
| Calendar Adjustment | waiting_time | 5% | Moderate variance from hour extensions |
| Shift Responsibility | cycle_time | 5% | Symmetric tolerance |

### Experiments by Improvement Type

#### Resource Allocation (E1.1 - E1.5)

| ID | Log | Target Role | Fault |
|----|-----|-------------|-------|
| E1.1 | LoanApp.csv | Clerk | 8→1 |
| E1.2 | LoanApp.csv | Loan Officer | 4→1 |
| E1.3 | LoanApp.csv | AML Investigator | 2→1 |
| E1.4 | LoanApp.csv | Appraiser | 2→1 |
| E1.5 | LoanApp.csv | Senior Officer | 2→1 |

#### Calendar Adjustment (E2.1 - E2.5)

| ID | Log | Target Resource | Fault |
|----|-----|-----------------|-------|
| E2.1 | LoanApp.csv | Clerk | 8h→2h |
| E2.2 | LoanApp.csv | Loan Officer | 8h→2h |
| E2.3 | p2p.csv | Karel de Groot | 8h→2h |
| E2.4 | helpdesk.csv | Value 2 | 8h→2h |
| E2.5 | env_permit.csv | Resource01 | 8h→2h |

#### Shift Responsibility (E3.1 - E3.5)

| ID | Log | Overloaded Resource | Shift To |
|----|-----|---------------------|----------|
| E3.1 | p2p.csv | Kim Passa | Immanuel Karagianni |
| E3.2 | p2p.csv | Elvira Lores | Christian Francois |
| E3.3 | p2p.csv | Karel de Groot | Magdalena Predutta |
| E3.4 | p2p.csv | Francois de Perrier | Karel de Groot |
| E3.5 | p2p.csv | Pedro Alvares | Karalda Nimwada |

### Usage

```bash
# Run all improvement types
python src/evaluation/quantitative/run_quantitative_evaluation.py --skip-grid-search --repetitions 1

# Run specific improvement type
python src/evaluation/quantitative/run_quantitative_evaluation.py \
    --logs-dir logs --type resource_allocation

python src/evaluation/quantitative/run_quantitative_evaluation.py \
    --logs-dir logs --type calendar_adjustment

python src/evaluation/quantitative/run_quantitative_evaluation.py \
    --logs-dir logs --type shift_responsibility

# Run specific experiments
python src/evaluation/quantitative/run_quantitative_evaluation.py \
    --logs-dir logs --experiments E1.1 E1.2 E2.1

# Generate final plots
python src/evaluation/quantitative/generate_figures.py
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--logs-dir`, `-l` | `../logs` | Directory containing event logs |
| `--output-dir`, `-o` | `../output` | Output directory for results |
| `--model`, `-m` | `anthropic--claude-4.5-opus` | LLM model for guided search |
| `--repetitions`, `-r` | `5` | Repetitions per experiment |
| `--tolerance` | `5.0` | KPI tolerance % for convergence |
| `--des-runs` | `5` | DES simulation runs |
| `--abs-runs` | `5` | ABS simulation runs |
| `--type`, `-t` | `all` | Improvement type to evaluate |
| `--experiments`, `-e` | - | Specific experiment IDs |
| `--skip-grid-search` | - | Use theoretical iterations instead |
| `--no-dual-validation` | - | Disable dual-simulator validation |
| `--dry-run` | - | Print configuration without running |

### Output

```
output/impr_type_eval_YYYYMMDD_HHMMSS/
├── evaluation_report.json      # Full results
├── evaluation_report.md        # Human-readable report
├── E1.1/, E1.2/, ...           # Per-experiment directories
└── figures/                    # Visualization plots
```

---

## Metrics

| Metric | Definition |
|--------|------------|
| **Time to improvement** | Wall-clock seconds until validated improvement |
| **Iterations to improvement** | Hypothesis iterations until success |
| **Averaged validation score** | `(S_DES + S_ABS) / 2` where `S = delta_throughput - delta_cycle_time` |
| **Speedup ratio** | Grid iterations / LLM iterations |
| **Success rate** | Percentage of runs finding validated improvement |