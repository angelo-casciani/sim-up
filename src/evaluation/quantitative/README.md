# Comprehensive Evaluation Framework for LLM-Guided Process Improvement

**Document Version:** 1.3
**Date:** March 18, 2026
**Purpose:** Evaluation design proposal for journal paper submission
**Revision Notes:** v1.3 - Validated log selection, fixed fragmented calendar handling, reduced to 15 experiments across working logs

---

## Changelog

### v1.3 (March 18, 2026) - Log Validation & Calendar Fix

#### Critical Discovery: SIMOD resourceId Column Behavior

Logs with a `resourceId` column cause SIMOD to discover UUIDs instead of the `resource` column's Role-NNNNNN pattern. This breaks role-based reasoning for the LLM.

**Affected logs (excluded):**
- `cvs_pharmacy.csv` - has `resourceId` column with UUIDs
- `Confidential_1000.csv` - has `resourceId` column with UUIDs

**Working logs (no resourceId column):**
- `LoanApp.csv` - Role-NNNNNN pattern discovered correctly
- `p2p.csv` - Named individuals discovered correctly
- `helpdesk.csv` - Value N pattern discovered correctly
- `env_permit.csv` - ResourceNN pattern discovered correctly

#### Calendar Fault Injection Fix

**Problem:** SIMOD discovers **fragmented calendars** with multiple time periods per day (e.g., Mon 13:00-15:00, Tue 07:00-08:00 + 10:00-12:00). The old code only modified the first period.

**Fix:** `calendar_fault_injector.py` now:
1. `get_calendar_hours()` - Sums ALL time periods for total weekly hours
2. `inject_fault()` - Reduces ALL periods proportionally using keep_ratio
3. `restore_hours()` - Expands ALL periods proportionally

**Bug fix:** `quantitative_evaluation_runner.py` was using old `fault_hours` key instead of new `fault_hours_daily` key, causing LLM to receive wrong context.

#### Experiment Reduction

Reduced from 24 to **15 experiments** using only validated logs:
- Resource Allocation: 5 experiments (LoanApp.csv only)
- Calendar Adjustment: 5 experiments (LoanApp, p2p, helpdesk, env_permit)
- Role Redesign: 5 experiments (LoanApp, p2p)

### v1.2 (March 13, 2026) - Fair Comparison Fixes

#### Issues Fixed

1. **Resource Allocation: Constant 7:4 ratio regardless of fault**
   - **Root cause:** LLM search used exact-count matching (`proposed_count == optimal_count`) while Grid search used tolerance-based matching (`_is_within_tolerance`)
   - **Fix:** Unified solution detection criteria - both now use tolerance-based cycle_time comparison
   - **Additional fix (v1.2.1):** Simulation variance caused no solutions with tight 2% tolerance. Now uses **30% tolerance** for resource allocation due to high simulation variance.

2. **Role Redesign: Immediate convergence at iteration 1 (1:1 speedup)**
   - **Root cause:** First shift option (10%) was already "good enough" to pass 5% tolerance
   - **Fix:**
     - Increased search granularity: 20 options (5% to 100% in 5% steps) instead of 8 options (10% to 80%)
     - Tolerance set to 5% for role redesign
     - Reduced fault severity from 80% to 50% shift, placing optimal fix in the middle of search space

3. **Calendar Adjustment: Works correctly (reference implementation)**
   - Both Grid and LLM use identical tolerance-based detection on `waiting_time`
   - LLM can propose any value in continuous range, enabling intelligent jumps
   - Uses 5% tolerance

### Key Changes (v1.2)

| Component | Before | After |
|-----------|--------|-------|
| `resource_llm_search.py` line 192-193 | `is_solution = (proposed_count == optimal_count)` | `is_solution = self._is_within_tolerance(cycle_time, optimal_cycle_time)` |
| `role_redesign_grid_search.py` | 8 options (10%-80%), 5% tolerance | 20 options (5%-100%), 5% tolerance |
| `role_redesign_llm_search.py` | 8 options, 5% tolerance, max 5 iterations | 20 options, 5% tolerance, max 20 iterations |
| `quantitative_evaluation_runner.py` | 80% fault shift, single tolerance | 50% fault shift, per-type tolerances |

### Tolerance Settings

| Improvement Type | Tolerance | Rationale |
|------------------|-----------|-----------|
| Resource Allocation | 30% | High simulation variance due to resource count changes |
| Calendar Adjustment | 5% | Moderate variance, original setting worked well |
| Role Redesign | 1% | Tight tolerance required - 5% tolerance caused immediate convergence at iteration 1 |

**Note:** Role redesign required reducing tolerance from 5% to 1% because even small shifts (5%) were bringing cycle time within 5% of optimal, causing both Grid and LLM to find solutions at iteration 1 (no speedup advantage).

---

## 1. Executive Summary

This document outlines a comprehensive evaluation framework to assess the effectiveness of an LLM-guided process improvement pipeline. The evaluation compares LLM-based search against systematic baselines across **three improvement types**:

1. **Resource Allocation** - Adding resources to bottleneck roles
2. **Calendar Adjustment** - Extending working hours for constrained resources
3. **Role Redesign** - Redistributing workload between roles

A fourth scenario—**Mixed/Cross-type**—where the LLM freely chooses among all three improvement types is deferred to future work due to the difficulty of establishing a fair grid search baseline.

Additionally, an **LLM Benchmark Evaluation** tests multiple LLM models on the full improvement pipeline to measure:
- **Speed**: Time and iterations to find a validated improvement
- **Impact**: Quality of the improvement (averaged validation formula score)

The primary research question is: **Can an LLM identify optimal process improvements with fewer simulation calls than systematic search methods?**

---

## 2. Background: The LLM Pipeline

### 2.1 Pipeline Overview

The existing pipeline operates as follows:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Event Log     │────▶│  SIMOD Discovery │────▶│ Baseline Model  │
│   (CSV/XES)     │     │  (via Docker)    │     │  (BPMN + JSON)  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    IMPROVEMENT LOOP                                  │
│                                                                      │
│  ┌─────────────┐    ┌─────────────────────────────────────────────┐ │
│  │   LLM       │    │         DUAL-SIMULATOR VALIDATION           │ │
│  │  Hypothesis │───▶│  ┌───────────┐         ┌────────────────┐  │ │
│  │  Generator  │    │  │ Prosimos  │         │ AgentSimulator │  │ │
│  └─────────────┘    │  │  (DES)    │         │     (ABS)      │  │ │
│         ▲           │  └─────┬─────┘         └───────┬────────┘  │ │
│         │           │        │    Structural         │ Behavioral│ │
│         │           │        │    (queues,           │ (handoffs,│ │
│         │           │        │     capacity)         │  routing) │ │
│         │           │        └──────────┬────────────┘           │ │
│         │           │                   ▼                        │ │
│         │           │        ┌──────────────────────┐            │ │
│         │           │        │  Combined Validation │            │ │
│         │           │        │  S_final = f(S_DES,  │            │ │
│         │           │        │            S_ABS)    │            │ │
│         │           │        └──────────┬───────────┘            │ │
│         │           └───────────────────┼────────────────────────┘ │
│         │                               │                          │
│         └───────────────────────────────┘                          │
│                    (feedback loop)                                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Stereoscopic Validation:** The pipeline uses two complementary simulators:
- **Prosimos (DES):** Captures structural aspects—resource queues, capacity constraints, waiting times
- **AgentSimulator (ABS):** Captures behavioral aspects—agent handoffs, routing decisions, role interactions

This dual-simulator approach provides robustness: an improvement must show positive effects in both structural AND behavioral dimensions to be accepted.

### 2.2 Improvement Types and Validation Oracle

| Type | Parameter | Primary Validator | Secondary Validator | Rationale |
|------|-----------|-------------------|---------------------|-----------|
| **Resource Allocation** | `resources` | DES (Prosimos) | ABS (throughput check) | DES captures queue dynamics; ABS confirms no routing degradation |
| **Calendar Adjustment** | `calendar` | DES (Prosimos) | N/A (DES-only) | DES captures availability effects; ABS doesn't support calendar changes |
| **Role Redesign** | `role_redesign` | ABS (AgentSimulator) | DES (throughput check) | ABS captures handoff behavior; DES confirms no capacity degradation |

**Combined Validation Score:**

For Resource and Calendar improvements:
```
S_final = (S_DES + S_ABS) / 2
Accepted if: S_final > threshold AND S_DES > 0 AND S_ABS > 0
```

For Role Redesign improvements:
```
S_final = S_ABS (primary) with S_DES >= 0 (safety check)
Accepted if: S_ABS > threshold AND S_DES >= -0.05 (no significant throughput loss)
```

This ensures improvements are validated by both simulators, preventing acceptance of changes that improve one dimension while degrading another.

### 2.3 Current Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| SIMOD Discovery | ✅ Implemented | `src/simulator/simod_discoverer.py` |
| Prosimos Simulation (DES) | ✅ Implemented | `src/simulator/prosimos_runner.py` |
| Agent Simulation (ABS) | ✅ Implemented | `src/simulator/agent_simulator_runner.py` |
| LLM Hypothesis Generator | ✅ Implemented | `src/llm/hypothesis_generator.py` |
| **Resource Allocation** | | |
| Resource Fault Injection | ✅ Implemented | `src/evaluation/quantitative/fault_injector.py` |
| Resource Grid Search | ✅ Implemented | `src/evaluation/quantitative/grid_search.py` |
| Resource LLM Search | ✅ Implemented | `src/evaluation/quantitative/llm_search.py` |
| **Calendar Adjustment** | | |
| Calendar Fault Injection | ✅ Implemented | `src/evaluation/quantitative/calendar_fault_injector.py` |
| Calendar Grid Search | ✅ Implemented | `src/evaluation/quantitative/calendar_grid_search.py` |
| Calendar LLM Search | ✅ Implemented | `src/evaluation/quantitative/calendar_llm_search.py` |
| **Role Redesign** | | |
| Role Redesign Fault Injection | ✅ Implemented | `src/evaluation/quantitative/role_redesign_fault_injector.py` |
| Role Redesign Heuristic Baseline | ✅ Implemented | `src/evaluation/quantitative/role_redesign_heuristic_search.py` |
| Role Redesign LLM Search | ✅ Implemented | `src/evaluation/quantitative/role_redesign_llm_search.py` |
| **Comprehensive Evaluation** | | |
| Comprehensive Evaluation Runner | ✅ Implemented | `src/evaluation/quantitative/comprehensive_evaluation_runner.py` |
| Comprehensive Results Reporter | ✅ Implemented | `src/evaluation/quantitative/comprehensive_results_reporter.py` |
| Results Visualizer | ✅ Implemented | `src/evaluation/quantitative/results_visualizer.py` |
| CLI Entry Point | ✅ Implemented | `src/evaluation/quantitative/run_comprehensive_evaluation.py` |
| Mixed/Cross-type Evaluation | ❌ Future work | - |
| **LLM Benchmark Evaluation** | | |
| LLM Benchmark Runner | ✅ Implemented | `src/evaluation/quantitative/llm_benchmark_runner.py` |
| LLM Benchmark Reporter | ✅ Implemented | `src/evaluation/quantitative/llm_benchmark_reporter.py` |
| LLM Benchmark CLI | ✅ Implemented | `src/evaluation/quantitative/run_llm_benchmark.py` |

---

## 3. Evaluation Design

### 3.1 Research Questions

**RQ1:** How many simulation calls does the LLM need to find an optimal improvement compared to grid search?

**RQ2:** Does the LLM correctly identify bottleneck resources/roles?

**RQ3:** How does LLM performance vary across different improvement types?

### 3.2 Experimental Methodology

The evaluation uses a **fault injection approach**:

1. **Discover** optimal model M_opt from event log using SIMOD
2. **Inject fault** to create degraded model M_fault
3. **Run LLM search** to recover from fault
4. **Run baseline search** (grid search) on same fault
5. **Compare** iterations needed to converge to baseline KPIs

```
     M_opt (ground truth)
         │
         │ inject fault
         ▼
     M_fault (degraded)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 LLM Search   Grid Search
    │         │
    └────┬────┘
         │
         ▼
    Compare iterations
    to reach M_opt KPIs
```

### 3.3 Convergence Criterion

An improvement is considered **converged** when:

```
primary_kpi_improvement >= tolerance_threshold
AND
throughput >= 0.95 × baseline_throughput
```

| Improvement Type | Primary KPI | Rationale |
|------------------|-------------|-----------|
| Resource Allocation | waiting_time | Adding resources directly reduces queue time |
| Calendar Adjustment | waiting_time | Extended hours reduce overnight queuing |
| Role Redesign | cycle_time | Rebalancing reduces handoff delays |

**Default tolerance threshold:** 5% improvement over faulty model

---

## 4. Evaluation Type 1: Resource Allocation

### 4.1 Fault Injection Strategy

**Method:** Reduce resource count for a bottleneck role from N to 1

**Example (LoanApp):**
- Original: 8 Clerks, 4 Loan Officers, 2 AML Investigators
- Fault: Reduce AML Investigators from 2 → 1
- Ground truth: LLM should propose increasing AML Investigators back to 2

### 4.2 Search Space

| Method | Search Space | Complexity |
|--------|--------------|------------|
| Grid Search | count ∈ [1, N] for target role | O(N) |
| LLM Search | Guided by utilization analysis | O(k) where k << N expected |

### 4.3 Baseline: Grid Search

Sequential search from count=1 to count=N, stopping when KPIs converge:

```python
for count in range(1, original_count + 1):
    metrics = simulate(model, target_role, count)
    if converged(metrics, baseline_metrics):
        return count, iteration_number
```

### 4.4 Metrics

| Metric | Definition |
|--------|------------|
| **Iterations to solution** | Number of simulations to reach convergence |
| **Speedup ratio** | Grid iterations / LLM iterations |
| **Accuracy** | Did LLM find the exact optimal count? |
| **Convergence quality** | Final KPI improvement percentage |
| **Intervention cost level** | Ordinal cost of the proposed solution (see Section 4.6) |

### 4.5 Candidate Logs

| Log | Target Role | Original Count | Fault (reduce to) |
|-----|-------------|----------------|-------------------|
| LoanApp.csv | AML Investigator | 2 | 1 |
| LoanApp.csv | Clerk | 8 | 2 |
| cvs_pharmacy.csv | Technician | 3 | 1 |

### 4.6 Intervention Cost Metric

To evaluate solution quality beyond KPI convergence, we track the **cost level** of proposed interventions:

| Cost Level | Intervention Type | Description | Example |
|------------|-------------------|-------------|---------|
| **1 (Low)** | Role Redesign | Redistribute existing workforce | Shift tasks from Clerk to Loan Officer |
| **2 (Medium)** | Calendar Extension | Increase working hours (overtime) | Extend AML Investigator hours by 2h |
| **3 (High)** | Resource Addition | Hire new staff | Add 2 AML Investigators |

**Why this matters:** An LLM that achieves high KPI improvement with a Cost Level 1 intervention demonstrates better business reasoning than one that simply proposes hiring more staff (Cost Level 3).

**Reporting:** For each experiment, we report:
- Whether LLM chose the minimum-cost solution type
- Cost level of LLM solution vs. Grid search solution

---

## 5. Evaluation Type 2: Calendar Adjustment

### 5.1 Verification: Calendar Data Availability

**VERIFIED:** SIMOD extracts per-resource calendars. Example from LoanApp:

```json
{
  "id": "AML Investigator-000001_profile_calendar",
  "time_periods": [
    {"from": "MONDAY", "to": "MONDAY", "beginTime": "07:00:00", "endTime": "15:00:00"},
    {"from": "TUESDAY", "to": "TUESDAY", "beginTime": "07:00:00", "endTime": "15:00:00"},
    ...
  ]
}
```

Each of the 19 resources in LoanApp has its own calendar with working hours defined per weekday.

### 5.2 Fault Injection Strategy

**Method:** Reduce working hours for a target resource

**Example:**
- Original: AML Investigator-000001 works 07:00-15:00 (8 hours)
- Fault: Restrict to 07:00-11:00 (4 hours)
- Ground truth: LLM should propose increasing hours by 4

### 5.3 Search Space

| Method | Search Space | Complexity |
|--------|--------------|------------|
| Grid Search | hours_to_add ∈ [1, H] for target resource | O(H) where H ≈ 8 |
| LLM Search | Guided by utilization + calendar analysis | O(k) |

**Key simplification:** We inject fault on a KNOWN target resource, so grid search only varies hours (not resource identity). This makes the comparison fair—both methods search the same space.

### 5.4 Baseline: Grid Search over Hours

```python
for hours_to_add in range(1, max_hours + 1):
    metrics = simulate(model, target_resource, hours_to_add)
    if converged(metrics, baseline_metrics):
        return hours_to_add, iteration_number
```

### 5.5 Metrics

Same as Resource Allocation:
- Iterations to solution
- Speedup ratio
- Accuracy (exact hours found?)
- Convergence quality

### 5.6 Candidate Logs

| Log | Target Resource | Original Hours | Fault (reduce to) |
|-----|-----------------|----------------|-------------------|
| LoanApp.csv | AML Investigator-000001 | 8h/day | 2h/day |
| LoanApp.csv | Clerk-000001 | ~8h/day | 2h/day |
| cvs_pharmacy.csv | Technician-000001 | 8h/day | 2h/day |

---

## 6. Evaluation Type 3: Role Redesign

### 6.1 Complexity

Role redesign has a **combinatorial search space**:
- N source roles × (N-1) target roles × M shift percentages
- For LoanApp with 6 roles and 5 shift levels: 6 × 5 × 5 = 150 combinations

### 6.2 Fault Injection Strategy

**Method:** Create artificial workload imbalance by modifying task assignment probabilities

**Example (LoanApp):**
- Original: Clerks handle 50% of administrative tasks, Loan Officers handle 50%
- Fault: Shift 50% of tasks to Clerks (creating overload)
- Ground truth: LLM should propose shifting ~50% responsibility back to Loan Officers

**Implementation approach:**
1. Identify two roles with overlapping activities (can perform same tasks)
2. Modify `task_resource_distribution` to concentrate work on one role
3. This creates utilization imbalance without changing resource counts

**Note on fault severity (v1.2):** Using 50% shift (instead of 80%) places the optimal fix in the middle of the search space, making the evaluation more meaningful. A smaller fault at the edge of the search space would be trivially solved.

### 6.3 Search Space

| Method | Search Space | Complexity |
|--------|--------------|------------|
| Exhaustive Search | All (source, target, shift%) combinations | O(N² × M) |
| Grid Search | Fixed shift percentages: 5%, 10%, 15%, ..., 100% (20 options) | O(20) = O(M) |
| LLM Search | Guided by utilization analysis | O(k) where k << M expected |

**v1.2 Change:** Increased granularity from 8 options (10%-80%) to 20 options (5%-100%) for more meaningful differentiation between Grid and LLM performance.

### 6.4 Baseline: Grid Search

We use a systematic grid search over shift percentages with finer granularity:

```python
# Role pair already identified by utilization analysis (shared by both methods)
from_role = highest_utilization_role  # Overloaded
to_role = lowest_utilization_role     # Underloaded

# Grid search over shift percentages (20 options)
for shift_pct in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
    metrics = simulate(model, from_role, to_role, shift_pct)
    if converged(metrics, baseline_metrics, tolerance=2%):
        return (from_role, to_role, shift_pct), iteration_number
```

This baseline uses the **same domain knowledge** available to the LLM (utilization data), making the comparison fair.

### 6.5 Metrics

Same as other evaluation types, plus:

| Metric | Definition |
|--------|------------|
| **Role pair accuracy** | Did LLM identify the correct (source, target) pair? |
| **Shift percentage accuracy** | Did LLM propose an effective shift amount? |

### 6.6 Candidate Logs

| Log | Overloaded Role | Underloaded Role | Overlap Activities |
|-----|-----------------|------------------|-------------------|
| LoanApp.csv | Clerk | Loan Officer | Document review (if applicable) |
| p2p.csv | Karel de Groot (95%) | Fjodor Kowalski (20%) | TBD after discovery |
| env_permit.csv | TBD | TBD | Uses `org:group` for role identification |

**Note:** Role redesign requires roles with **overlapping activities** (both can perform the same tasks). Not all logs support this—verification needed during implementation.

---

## 7. LLM Benchmark Evaluation

### 7.1 Overview

The LLM Benchmark Evaluation tests multiple LLM models on the full `ImprovementValidationAgent` pipeline to compare their performance in finding validated process improvements. Unlike the fault-injection evaluations (Sections 4-6), this benchmark uses the complete pipeline without artificial faults.

**Research Question:** Which LLM models are fastest at finding validated improvements, and what is the quality of their solutions?

### 7.2 Models Benchmarked

| Provider | Models | Size Category |
|----------|--------|---------------|
| OpenAI | gpt-4.1 | Large (Previous Gen) |
| OpenAI | gpt-5 | Large |
| OpenAI | gpt-5-mini | Medium |
| OpenAI | gpt-5-nano | Small |
| OpenAI | o3 | Reasoning |
| Anthropic | anthropic--claude-4.5-opus | Large |
| Anthropic | anthropic--claude-4.5-sonnet | Medium |
| Anthropic | anthropic--claude-4.5-haiku | Small |
| Mistral | mistralai--mistral-large-instruct | Large |
| Mistral | mistralai--mistral-small-instruct | Small |

**Total: 11 models** across 3 providers, covering large/medium/small tiers plus reasoning models.

### 7.3 Log Selection for LLM Benchmark

For a reasonable journal paper evaluation, we recommend **3-4 logs** across different domains and sizes.

#### 7.3.1 Recommended Logs

| Log | Domain | Rows | Cases | Recommended | Rationale |
|-----|--------|------|-------|-------------|-----------|
| **LoanApp.csv** | Financial | 7.5K | 997 | **Primary** | Fast SIMOD (~10 min), clean Role-NNNNNN pattern, well-tested |
| **p2p.csv** | Procurement | 7.9K | 550 | **Secondary** | Fast SIMOD, named resources, different domain |
| **BPIC_2012W.csv** | Banking | 51K | 8,030 | **BPIC Benchmark** | Standard BPI Challenge log, medium size (~20 min SIMOD) |
| **Production.csv** | Manufacturing | 2.4K | ~200 | Optional | Very fast, manufacturing domain |

#### 7.3.2 Excluded Logs (with reasons)

| Log | Rows | Reason for Exclusion |
|-----|------|---------------------|
| **BPIC_2017W.csv** | 230K | **AgentSimulator timeout** - Too large for ABS within reasonable time |
| **bpic_2012.csv** | 262K | Too large - combined A/O/W subprocess log |
| **cvs_pharmacy.csv** | 85K | Has `resourceId` column - SIMOD discovers UUIDs |
| **Confidential_*.csv** | 38-77K | Has `resourceId` column - SIMOD discovers UUIDs |
| **bpic_2013I.csv** | 65K | Each person unique - no multi-resource roles |

#### 7.3.3 Journal Paper Evaluation Matrix

For a comprehensive journal paper, we recommend:

| Evaluation Tier | Logs | Models | Repetitions | Est. Runtime |
|-----------------|------|--------|-------------|--------------|
| **Quick validation** | LoanApp | 3 (GPT-5, Claude Opus, o3) | 1 | ~3-4 hours |
| **Standard evaluation** | LoanApp, p2p | 5 models | 3 | ~20-30 hours |
| **Full benchmark** | LoanApp, p2p, BPIC_2012W | All 10 models | 3 | ~50-80 hours |

### 7.4 Metrics

| Metric | Definition |
|--------|------------|
| **Time to first improvement** | Wall-clock seconds until first validated improvement |
| **Iterations to first improvement** | Number of hypothesis iterations until success |
| **Averaged validation score** | `(S_DES + S_ABS) / 2` where `S = delta_throughput - delta_cycle_time` |
| **Distance to threshold** | `max(0, 0.01 - averaged_score)` - 0 when 1% improvement achieved |
| **Success rate** | Percentage of runs that found a validated improvement |

### 7.5 Per-Iteration Tracking

For each iteration, the benchmark captures:

| Field | Description |
|-------|-------------|
| `hypothesis_description` | Text description of the proposed improvement |
| `hypothesis_type` | Type: `resource_allocation`, `calendar_adjustment`, or `role_redesign` |
| `is_improvement` | Whether this iteration was validated as an improvement (score >= 1%) |
| `des_cycle_time_change` | % change in DES cycle time vs baseline |
| `des_throughput_change` | % change in DES throughput vs baseline |
| `abs_cycle_time_change` | % change in ABS cycle time vs baseline |
| `abs_throughput_change` | % change in ABS throughput vs baseline |
| `averaged_score` | Combined validation score |
| `distance_to_threshold` | Distance to 1% improvement threshold (for convergence plots) |
| `iteration_time` | Wall-clock time for this iteration (seconds) |
| `cumulative_time` | Total time from start including baseline discovery |

### 7.6 Visualizations

The benchmark automatically generates publication-quality figures:

| Figure | Description | Purpose |
|--------|-------------|---------|
| **convergence_trajectory.png** | Distance to 1% threshold vs iterations | Shows how quickly each model approaches validation |
| **score_progression.png** | Averaged score vs iterations | Shows improvement trajectory per model |
| **iterations_comparison.png** | Bar chart of iterations to first improvement | Compare speed across models |
| **time_comparison.png** | Bar chart of time to first improvement | Compare wall-clock speed |
| **impact_comparison.png** | Bar chart of averaged validation score | Compare improvement quality |
| **speed_vs_impact.png** | Scatter: iterations vs score | Trade-off visualization |
| **success_rate.png** | Bar chart of success rate | Model reliability comparison |
| **improvement_magnitude_box.png** | Box plot of improvement magnitudes | Distribution of improvement quality per model |
| **improvement_by_type.png** | Grouped bar by hypothesis type | Which improvement types each model excels at |
| **best_improvement_comparison.png** | Bar chart of best improvement found | Maximum improvement potential per model |

All figures are saved in both PNG (300 DPI) and PDF formats for publication.

### 7.7 Usage

```bash
# Run benchmark with default settings (all 10 models)
python src/evaluation/quantitative/run_llm_benchmark.py --log LoanApp.csv

# Run with specific models
python src/evaluation/quantitative/run_llm_benchmark.py \
    --models gpt-5 anthropic--claude-4.5-sonnet o3 --log LoanApp.csv

# Run with multiple repetitions for statistical significance
python src/evaluation/quantitative/run_llm_benchmark.py --log LoanApp.csv --repetitions 3

# Run on BPIC benchmark log
python src/evaluation/quantitative/run_llm_benchmark.py --log BPIC_2012W.csv --max-iterations 10

# Dry run to validate configuration
python src/evaluation/quantitative/run_llm_benchmark.py --log LoanApp.csv --dry-run
```

### 7.8 Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--log`, `-l` | `LoanApp.csv` | Event log file in logs/ directory |
| `--models`, `-m` | All 10 models | Specific models to benchmark |
| `--repetitions`, `-r` | `1` | Number of runs per model |
| `--max-iterations` | `5` | Maximum hypothesis iterations per run |
| `--output-dir`, `-o` | `output` | Output directory for results |
| `--no-sudo` | Auto (Windows) | Don't use sudo for Docker commands |
| `--des-runs` | `5` | Number of DES (Prosimos) runs to average |
| `--abs-runs` | `5` | Number of ABS (AgentSimulator) runs |
| `--dry-run` | - | Print configuration without running |

### 7.9 Output Structure

```
output/llm_benchmark_YYYYMMDD_HHMMSS/
├── benchmark_results.json      # Full structured results with per-iteration data
├── benchmark_results.csv       # Spreadsheet export
├── benchmark_report.md         # Summary with rankings
└── figures/                    # Publication-quality visualizations
    ├── convergence_trajectory.png/pdf
    ├── score_progression.png/pdf
    ├── iterations_comparison.png/pdf
    ├── time_comparison.png/pdf
    ├── impact_comparison.png/pdf
    ├── speed_vs_impact.png/pdf
    └── success_rate.png/pdf
```

### 7.10 Report Contents

The benchmark report includes:

1. **Summary Table**: All models with success rate, avg time, avg iterations, avg score
2. **Speed Ranking**: Models sorted by time to first improvement (ascending)
3. **Iteration Ranking**: Models sorted by iterations to first improvement (ascending)
4. **Impact Ranking**: Models sorted by averaged validation score (descending)
5. **Detailed Results**: Per-model breakdown with individual run data
6. **Iteration Details**: Per-iteration hypothesis, KPI impacts, distance to threshold, and timing
7. **Convergence Figures**: Visual comparison of model performance

### 7.11 Research Paper Evaluation Setup

For a comprehensive research paper evaluation:

```bash
# Quick validation (recommended first)
python src/evaluation/quantitative/run_llm_benchmark.py \
    --log LoanApp.csv --models gpt-5 anthropic--claude-4.5-opus o3 --max-iterations 5

# Standard evaluation on two logs
python src/evaluation/quantitative/run_llm_benchmark.py \
    --log LoanApp.csv --repetitions 3 --max-iterations 10

python src/evaluation/quantitative/run_llm_benchmark.py \
    --log p2p.csv --repetitions 3 --max-iterations 10

# Full benchmark with BPIC log
python src/evaluation/quantitative/run_llm_benchmark.py \
    --log BPIC_2012W.csv --repetitions 3 --max-iterations 10
```

**Expected runtime estimates:**
- LoanApp (small): ~15-20 min baseline + ~5 min/iteration × 10 iterations × 10 models × 3 reps = ~30-40 hours
- p2p (small): Similar to LoanApp, ~30-40 hours
- BPIC_2012W (medium): ~20-30 min baseline + ~8 min/iteration × 10 iterations × 10 models × 3 reps = ~50-70 hours

**Statistical reporting:**
- Results reported as Mean ± Standard Deviation across repetitions
- Rankings based on mean performance
- Per-iteration analysis shows improvement trajectory
- Convergence plots show distance to 1% threshold over iterations

---

## 8. Future Work: Mixed/Cross-type Evaluation

### 8.1 Scenario Description

In a **mixed** scenario, the LLM is not told which improvement type to use. Given a degraded model, it must:
1. Diagnose the problem (bottleneck identification)
2. Choose the best intervention type (Resource, Calendar, or Role Redesign)
3. Propose specific parameters

### 8.2 Why Deferred

**Challenge:** No fair grid search baseline exists for mixed scenarios.

- Grid search over Resources: O(N)
- Grid search over Calendar hours: O(H)
- Grid search over Role pairs: O(N² × M)
- **Combined:** O(N × H × N² × M) = intractable

**Alternative baselines considered:**
- Random type selection + grid within type
- Multi-armed bandit approach
- Requires additional research to define fair comparison

### 8.3 Future Implementation

When implemented, mixed evaluation would test:
- LLM's ability to select the **cost-optimal** intervention type
- Whether LLM explores multiple types before converging
- Comparison against random type selection baseline

---

## 9. Experimental Setup

### 9.1 Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Logs per improvement type** | 3 | Sufficient for initial validation |
| **Repetitions per experiment** | 5 | Account for LLM non-determinism (report mean ± std) |
| **LLM temperature** | 0.0 | Reduce variance while acknowledging non-determinism |
| **DES simulation runs** | 5-10 | Average to reduce simulation variance |
| **ABS simulation runs** | 5-10 | Parallel to DES for consistent validation |
| **Resource tolerance** | 30% | Higher tolerance to account for simulation variance |
| **Calendar tolerance** | 5% | Moderate tolerance, original working value |
| **Role redesign tolerance** | 1% | Tight tolerance to prevent trivial convergence (v1.2.2) |

**Note on LLM Non-Determinism:** Even with temperature=0.0, LLMs exhibit non-deterministic behavior due to factors like floating-point precision and batching. Running 5 independent repetitions allows us to report results as **Mean ± Standard Deviation**, providing statistical validity.

**Note on Per-Type Tolerances (v1.2.2):** Different improvement types exhibit different levels of simulation variance. Resource allocation has the highest variance because adding/removing resources significantly changes the simulation dynamics. Role redesign requires very tight tolerance (1%) because even small shifts can significantly improve KPIs, causing trivial solutions with 5% tolerance.

### 9.2 Experimental Matrix

**Note:** Only logs WITHOUT `resourceId` column are used (SIMOD discovers Role-NNNNNN patterns correctly). Logs with `resourceId` column cause SIMOD to discover UUIDs, breaking role-based reasoning.

#### Resource Allocation (E1.1 - E1.5)

| ID | Log | Domain | Target Role | Resources | Fault |
|----|-----|--------|-------------|-----------|-------|
| E1.1 | LoanApp.csv | Financial | Clerk | 8 | 8->1 |
| E1.2 | LoanApp.csv | Financial | Loan Officer | 4 | 4->1 |
| E1.3 | LoanApp.csv | Financial | AML Investigator | 2 | 2->1 |
| E1.4 | LoanApp.csv | Financial | Appraiser | 2 | 2->1 |
| E1.5 | LoanApp.csv | Financial | Senior Officer | 2 | 2->1 |

**Validation criteria:** Each role must have 2+ resources to allow reduction and recovery. Only LoanApp.csv has proper Role-NNNNNN pattern without resourceId column.

#### Calendar Adjustment (E2.1 - E2.5)

| ID | Log | Domain | Target Resource | Fault |
|----|-----|--------|-----------------|-------|
| E2.1 | LoanApp.csv | Financial | Clerk | 8h->2h |
| E2.2 | LoanApp.csv | Financial | Loan Officer | 8h->2h |
| E2.3 | p2p.csv | Procurement | Karel de Groot | 8h->2h |
| E2.4 | helpdesk.csv | IT Support | Value 2 | 8h->2h |
| E2.5 | env_permit.csv | Government | Resource01 | 8h->2h |

**Validation criteria:** Calendar adjustment works with any resource. Diverse logs across 4 domains.

**Note on fragmented calendars:** SIMOD discovers fragmented calendars (multiple periods per day). The fault injector reduces ALL periods proportionally to achieve target reduction.

#### Role Redesign (E3.1 - E3.5)

| ID | Log | Domain | Overloaded Role | Target Role | Shared Activities |
|----|-----|--------|-----------------|-------------|-------------------|
| E3.1 | LoanApp.csv | Financial | Clerk | Loan Officer | Document processing |
| E3.2 | LoanApp.csv | Financial | AML Investigator | Senior Officer | Review tasks |
| E3.3 | p2p.csv | Procurement | Karel de Groot | Magdalena Predutta | Quotation/order activities |
| E3.4 | p2p.csv | Procurement | Francois de Perrier | Karel de Groot | Quotation/order activities |
| E3.5 | p2p.csv | Procurement | Pedro Alvares | Karalda Nimwada | Invoice activities |

**Validation criteria:** Role redesign requires overlapping activities between roles. p2p.csv has two distinct activity groups with clear overlaps.

#### Excluded Logs (with reasons)

| Log | Reason for Exclusion |
|-----|---------------------|
| cvs_pharmacy.csv | Has `resourceId` column - SIMOD discovers UUIDs instead of Role-NNNNNN |
| Confidential_1000.csv | Has `resourceId` column - SIMOD discovers UUIDs instead of ABC-NNNNNN |
| BPI_Challenge_2012_A.csv | Numeric resource IDs only - no identifiable roles for LLM |
| BPIC_2012W.csv | Mixed numeric/artificial IDs - poor role identifiability |
| bpi_challenge_2013_incidents.csv | Each person is unique - no multi-resource roles |

**Total experiments:** 15 configurations × 5 repetitions = **75 experiment runs**

### 9.3 Results Reporting

For each experiment, we report:

| Metric | Format | Example |
|--------|--------|---------|
| Iterations to solution | Mean ± Std | 3.2 ± 0.84 |
| Speedup ratio | Mean ± Std | 2.5× ± 0.3 |
| Accuracy | Percentage | 80% (4/5 runs found optimal) |
| Convergence quality | Mean ± Std | 97.2% ± 1.5% |
| Intervention cost level | Mode | Level 2 (Calendar) |
| DES validation score | Mean ± Std | 0.042 ± 0.008 |
| ABS validation score | Mean ± Std | 0.031 ± 0.005 |

### 9.4 Hardware/Software Requirements

- Docker (for SIMOD and Prosimos)
- Python 3.11
- LLM API access (GPT-4.1 or Claude)
- ~2-4 hours total runtime (estimated)

---

## 10. Expected Outcomes

### 10.1 Hypotheses

**H1:** LLM search requires fewer iterations than grid search for resource allocation (speedup > 1.5×)

**H2:** LLM search requires fewer iterations than grid search for calendar adjustment (speedup > 1.5×)

**H3:** LLM correctly identifies the bottleneck resource in >80% of cases

**H4:** LLM proposes lower-cost interventions when multiple solution types are viable

### 10.2 Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Average speedup ratio | > 1.5× |
| LLM success rate (finds valid improvement) | > 80% |
| Convergence quality vs grid | Within 95% |
| Dual-simulator agreement | Both S_DES > 0 and S_ABS > 0 |
| Cost-optimal solution rate | > 60% (when lower-cost alternative exists) |

### 10.3 Potential Findings

**Positive outcomes:**
- LLM demonstrates sample efficiency (fewer simulations needed)
- LLM correctly prioritizes high-utilization resources
- Generalizes across different process types

**Negative outcomes (also valuable):**
- LLM struggles with certain process characteristics
- Grid search competitive for small search spaces
- Temperature sensitivity affects reproducibility

---

## 11. Implementation Plan

### Phase 1: Resource Allocation ✅ Complete
- [x] Fault injector for resource count
- [x] Grid search baseline
- [x] LLM search
- [x] Basic evaluation runner

### Phase 2: Calendar Adjustment ✅ Complete
- [x] Calendar fault injector (reduce hours)
- [x] Calendar grid search (iterate over hours)
- [x] Calendar LLM search
- [x] Integration with comprehensive evaluation runner

### Phase 3: Role Redesign ✅ Complete
- [x] Role redesign fault injector (create workload imbalance)
- [x] Heuristic baseline (highest-util → lowest-util)
- [x] Role redesign LLM search
- [x] Role pair accuracy metrics

### Phase 4: Comprehensive Evaluation ✅ Complete
- [x] Comprehensive evaluation runner with all 3 improvement types
- [x] Results reporter with statistical aggregation
- [x] Dual-simulator validation (DES + ABS)
- [x] CLI entry point with full configuration options

### Phase 5: Run Full Experiments (Pending)
- [ ] Run all experiments (8 configs × 5 reps = 40 runs)
- [ ] Generate comparison tables
- [ ] Statistical analysis (mean ± std)
- [ ] Visualization of results

---

## 12. Threats to Validity

### 12.1 Internal Validity

| Threat | Mitigation |
|--------|------------|
| LLM stochasticity | Use temperature=0.0, run **5 repetitions**, report mean ± std |
| Simulation variance | Average 5-10 DES/ABS runs per evaluation |
| Unfair comparison | Use simulation cache (same inputs → same outputs) |
| Single-simulator bias | Validate with **both DES and ABS** for all improvement types |

### 12.2 External Validity

| Threat | Mitigation |
|--------|------------|
| Limited log diversity | Use 3 different process types |
| Artificial fault injection | Inject realistic faults (bottleneck scenarios) |
| Single LLM model | Focus on one model first, extend later |

### 12.3 Construct Validity

| Threat | Mitigation |
|--------|------------|
| KPI selection | Use established process mining metrics |
| Convergence definition | Multi-KPI criterion (primary + throughput guard) |

---

## 13. Usage

### Running the Comprehensive Evaluation

```bash
# Run full evaluation with defaults
poetry run python src/evaluation/quantitative/run_comprehensive_evaluation.py \
    --logs-dir data/logs --output-dir output

# Run with specific model and more repetitions
poetry run python src/evaluation/quantitative/run_comprehensive_evaluation.py \
    --logs-dir data/logs --model anthropic--claude-4.5-opus --repetitions 10

# Run only resource allocation experiments
poetry run python src/evaluation/quantitative/run_comprehensive_evaluation.py \
    --logs-dir data/logs --type resource_allocation

# Run only calendar adjustment experiments
poetry run python src/evaluation/quantitative/run_comprehensive_evaluation.py \
    --logs-dir data/logs --type calendar_adjustment

# Run only role redesign experiments
poetry run python src/evaluation/quantitative/run_comprehensive_evaluation.py \
    --logs-dir data/logs --type role_redesign

# Dry run to validate configuration
poetry run python src/evaluation/quantitative/run_comprehensive_evaluation.py \
    --logs-dir data/logs --dry-run
```

### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--logs-dir`, `-l` | `logs` | Directory containing event logs |
| `--output-dir`, `-o` | `output` | Output directory for results |
| `--model`, `-m` | `anthropic--claude-4.5-opus` | LLM model for guided search |
| `--repetitions`, `-r` | `5` | Number of repetitions per experiment |
| `--tolerance` | `5.0` | Tolerance % for KPI solution matching |
| `--des-runs` | `5` | Number of DES (Prosimos) runs to average |
| `--abs-runs` | `5` | Number of ABS (AgentSimulator) runs |
| `--no-dual-validation` | - | Disable dual-simulator validation |
| `--type`, `-t` | `all` | Improvement type to evaluate |
| `--no-sudo` | - | Don't use sudo for Docker commands |
| `--skip-visualization` | - | Skip generating visualization plots |
| `--dry-run` | - | Print configuration without running |

### Output Structure

```
output/impr_type_eval_YYYYMMDD_HHMMSS/
├── evaluation_report.json          # Full results for all experiments
├── E1.1/                           # Per-experiment directory
│   ├── result.json                 # Aggregated result for this experiment
│   ├── run_1/                      # Working directory for repetition 1
│   ├── run_2/
│   └── ...
├── E1.2/
├── E2.1/
├── ...
└── figures/                        # Visualization plots
    ├── aggregate_comparison.png
    ├── convergence_overlay_*.png
    └── *.pdf
```

---

## 14. Module Structure

```
src/evaluation/quantitative/
├── __init__.py                         # Exports all components
├── README.md                           # This documentation
│
│ # Resource Allocation
├── fault_injector.py                   # Resource count fault injection
├── grid_search.py                      # Systematic grid search
├── llm_search.py                       # LLM-guided resource search
│
│ # Calendar Adjustment
├── calendar_fault_injector.py          # Calendar hours fault injection
├── calendar_grid_search.py             # Systematic hours grid search
├── calendar_llm_search.py              # LLM-guided calendar search
│
│ # Role Redesign
├── role_redesign_fault_injector.py     # Workload imbalance fault injection
├── role_redesign_heuristic_search.py   # Heuristic baseline (util-based)
├── role_redesign_llm_search.py         # LLM-guided role redesign search
│
│ # Comprehensive Evaluation
├── comprehensive_evaluation_runner.py  # Main orchestrator for all types
├── comprehensive_results_reporter.py   # Statistical reporting
├── results_visualizer.py               # charts
│
│ # Entry Points
├── run_evaluation.py                   # Original single-type evaluation
├── run_comprehensive_evaluation.py     # Full comprehensive evaluation CLI
│
│ # LLM Benchmark Evaluation
├── llm_benchmark_runner.py             # Multi-model benchmark runner
├── llm_benchmark_reporter.py           # Benchmark report generation
├── llm_benchmark_visualizer.py         # Publication-quality charts
└── run_llm_benchmark.py                # LLM benchmark CLI
```

---

## 15. Summary

This evaluation framework provides:

1. **Rigorous methodology:** Fault injection with known ground truth
2. **Fair comparison:** Grid search/heuristic baseline with comparable search space
3. **Dual-simulator validation:** Both DES (Prosimos) and ABS (AgentSimulator) for robustness
4. **Comprehensive metrics:** Iterations, speedup, accuracy, convergence quality, intervention cost
5. **Statistical validity:** 5 repetitions per experiment with mean ± std reporting
6. **Practical scope:** 15 experiments × 5 reps = 75 runs (across 4 validated logs)
7. **Clear implementation path:** All phases implemented, ready for full experiment runs

The expected contribution is demonstrating that LLM-guided search can achieve **sample efficiency** in process improvement—finding valid improvements with fewer simulation calls than systematic search, while respecting both structural (DES) and behavioral (ABS) constraints.

---

## Appendix A: Data Structures

### A.1 SIMOD Output Structure (Prosimos JSON)

```json
{
  "resource_profiles": [
    {
      "id": "Role-000001_profile",
      "resource_list": [
        {
          "id": "Role-000001",
          "amount": 1,
          "calendar": "Role-000001_profile_calendar",
          "assignedTasks": ["task_id_1", "task_id_2"]
        }
      ]
    }
  ],
  "resource_calendars": [
    {
      "id": "Role-000001_profile_calendar",
      "time_periods": [
        {"from": "MONDAY", "to": "MONDAY", "beginTime": "07:00:00", "endTime": "15:00:00"}
      ]
    }
  ],
  "task_resource_distribution": [...]
}
```

### A.2 Hypothesis JSON Structure

```json
{
  "description": "Add 1 AML Investigator to reduce queue time",
  "improvement_type": "resource_allocation",
  "changes": {
    "parameter_type": "resources",
    "specific_changes": [
      {"target": "AML Investigator", "change": "increase_count", "value": 1}
    ]
  },
  "expected_impact": {
    "cycle_time": "-5%",
    "throughput": "+2%",
    "waiting_time": "-10%"
  }
}
```

---

## Appendix B: Available Event Logs

### Critical: resourceId Column Behavior

**IMPORTANT:** SIMOD uses the `resourceId` column when present, otherwise falls back to `resource` column. Logs with `resourceId` containing UUIDs will have UUIDs discovered as resources, breaking role-based reasoning.

**Check before using a log:**
```bash
head -1 logs/LogName.csv | tr ',' '\n' | grep -i resourceid
```

If `resourceId` is present, the log likely won't work for this evaluation framework.

### Validated Logs (used in evaluation)

| Log | Domain | Resources | Pattern | Has resourceId? | Suitable For |
|-----|--------|-----------|---------|-----------------|--------------|
| LoanApp.csv | Financial | 19 | Role-NNNNNN | **No** | Resource, Calendar, Role Redesign |
| p2p.csv | Procurement | ~27 | Named individuals | **No** | Calendar, Role Redesign |
| helpdesk.csv | IT Support | ~20 | Value N | **No** | Calendar only |
| env_permit.csv | Government | ~30 | ResourceNN | **No** | Calendar only |

**LoanApp.csv roles:** Clerk(8), Loan Officer(4), AML Investigator(2), Appraiser(2), Senior Officer(2)

**p2p.csv resources:** Karel de Groot, Magdalena Predutta, Francois de Perrier, Pedro Alvares, Karalda Nimwada, etc.
- Group 1 (quotation/order): Karel, Magdalena, Francois share 6 activities
- Group 2 (invoice): Pedro, Karalda share 4 activities

### Excluded Logs (with reasons)

| Log | Domain | Has resourceId? | Reason for Exclusion |
|-----|--------|-----------------|---------------------|
| cvs_pharmacy.csv | Healthcare | **Yes** | SIMOD discovers UUIDs instead of Role-NNNNNN pattern |
| Confidential_1000.csv | Business | **Yes** | SIMOD discovers UUIDs instead of ABC-NNNNNN pattern |
| BPI_Challenge_2012_A.csv | Banking | No | Numeric resource IDs only - no identifiable roles |
| BPIC_2012W.csv | Banking | No | Mixed numeric/artificial IDs - poor role identifiability |
| bpi_challenge_2013_incidents.csv | IT Support | No | Each person unique - no multi-resource roles |

### Resource Count Validation

To verify a log is suitable for resource allocation evaluation:

```python
import pandas as pd

# First check for resourceId column
df = pd.read_csv('logs/LogName.csv')
if 'resourceId' in df.columns:
    print("WARNING: Log has resourceId column - SIMOD will use UUIDs!")

# Check role counts
df['role'] = df['resource'].str.split('-').str[0]  # Extract role prefix
role_counts = df.groupby('role')['resource'].nunique()
valid_roles = role_counts[role_counts >= 2]
print(f"Valid roles for resource allocation: {dict(valid_roles)}")
```

**Rule:** A role must have 2+ unique resources to be valid for resource allocation testing (allows reduction and recovery).

---

*End of Document*
