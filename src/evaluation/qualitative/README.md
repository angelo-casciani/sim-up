# Qualitative Evaluation Framework for LLM-Guided Process Improvement

**Document Version:** 1.0
**Date:** March 19, 2026
**Purpose:** Qualitative evaluation design for journal paper submission

---

## 1. Overview

This evaluation complements the quantitative benchmarks by assessing whether the LLM-generated process improvement recommendations are **actionable** and **meaningful** to domain experts. While quantitative evaluation measures detection accuracy and statistical performance, qualitative evaluation measures real-world utility.

### Research Questions

1. **RQ1:** Are LLM-generated process improvement recommendations perceived as actionable by domain experts?
2. **RQ2:** How well do LLM-as-judge evaluations correlate with human expert assessments?
3. **RQ3:** What types of recommendations are most/least useful in practice?

---

## 2. Dataset: Order-to-Cash (O2C) Event Logs

### 2.1 Data Source

Real-world SAP Order-to-Cash process logs from production systems.

### 2.2 Log Processing

Raw O2C logs are transformed to the standard format compatible with Prosimos and AgentSimulator using `log_processor.py`.

**Input format (raw):**
- `c_caseid` - Case identifier
- `c_eventname` - Activity name
- `c_time` - Event timestamp
- `EventCreatedByUserId` - Resource identifier

**Output format (processed):**
- `case_id` - Integer case identifier (0-indexed)
- `resource` - Cleaned resource name
- `activity_name` - Activity name
- `start_timestamp` - Event timestamp (ISO format with UTC timezone)
- `end_timestamp` - Same as start (events are instantaneous)
- `agent` - Integer agent ID

**Processing assumptions:**
- Events are treated as **instantaneous** (start_timestamp = end_timestamp)
- Memory address patterns in resource IDs are cleaned to "SystemAgent"
- Case IDs are mapped to sequential integers for compatibility

### 2.3 Available Logs

| Log ID | Cases | Events | Resources | Process Variant |
|--------|-------|--------|-----------|-----------------|
| 742427502 | 12 | 43 | 1 | Small pilot |
| 743866877 | 1,651 | 15,829 | 2 | Standard O2C |
| 744153953 | 582 | 1,861 | 1 | Invoice-focused |
| 744181388 | 47,416 | 93,923 | 2 | High-volume |
| 744204018 | 592 | 2,048 | 1 | Standard O2C |
| 744256331 | 7,380 | 7,380 | 1 | Single-event cases |
| 744865784 | 1,747 | 2,885 | 1 | Standard O2C |
| 745179240 | 29,653 | 30,827 | 1 | High-volume |
| 745347473 | 16,099 | 54,347 | 2 | Complex variant |

---

## 3. Evaluation Methodology

### 3.1 Study Design

**Mixed-methods approach:**
1. Run the improvement validation pipeline on each O2C log
2. Collect LLM-generated recommendations
3. Human experts evaluate recommendations using structured rubric
4. LLM-as-judge evaluates same recommendations with identical rubric
5. Compute agreement metrics between human and LLM evaluations
6. Qualitative analysis of disagreement cases

### 3.2 Evaluation Rubric

Each recommendation is scored on 5 dimensions (1-5 Likert scale):

| Dimension | Description | Score Anchors |
|-----------|-------------|---------------|
| **Correctness** | Is the insight factually accurate given the event log data? | 1=Incorrect, 5=Fully accurate |
| **Relevance** | Does it address a real bottleneck or inefficiency in the process? | 1=Irrelevant, 5=Highly relevant |
| **Actionability** | Can this recommendation be implemented in practice? | 1=Impossible, 5=Immediately actionable |
| **Specificity** | Is the recommendation concrete or vague? | 1=Very vague, 5=Highly specific |
| **Impact** | Would implementing this meaningfully improve KPIs? | 1=No impact, 5=Transformative |

### 3.3 Human Expert Selection

**Criteria:**
- 3-5 evaluators per log (minimum for inter-rater reliability)
- Domain expertise: O2C process knowledge OR BPM/process mining expertise
- Mix of industry practitioners and academic researchers

**Protocol:**
1. Brief training session on evaluation rubric
2. Independent evaluation (no discussion between evaluators)
3. Optional free-text comments for each recommendation

### 3.4 LLM-as-Judge Configuration

**Model selection:**
- Use a **different model family** than the one generating recommendations
- Example: If GPT-4 generates, use Claude for judging (mitigates self-enhancement bias)

**Prompt template:**
```
You are evaluating a process improvement recommendation for an Order-to-Cash business process.

Context:
- Process log summary: [statistics]
- Discovered bottlenecks: [from simulation]

Recommendation to evaluate:
[recommendation text]

Score this recommendation on each dimension (1-5):
1. Correctness - Is the insight factually accurate?
2. Relevance - Does it address a real bottleneck?
3. Actionability - Can this be implemented?
4. Specificity - Is it concrete or vague?
5. Impact - Would it meaningfully improve KPIs?

Provide brief justification for each score.
```

---

## 4. Statistical Analysis

### 4.1 Inter-Rater Reliability (Humans)

**Metric:** Krippendorff's alpha (ordinal data)
- α > 0.80: Good reliability
- α > 0.67: Acceptable reliability
- α < 0.67: Poor reliability (revisit rubric)

### 4.2 Human-LLM Agreement

**Primary metrics:**
- **Spearman's rho (ρ)** - Rank correlation (appropriate for ordinal scales)
- **Cohen's weighted Kappa** - Agreement accounting for chance
- **Mean Absolute Error (MAE)** - Average score difference

**Analysis by dimension:**
- Report metrics separately for each of the 5 evaluation dimensions
- Identify which dimensions show strongest/weakest agreement

### 4.3 Qualitative Analysis

**Disagreement case analysis:**
- Sample cases where human-LLM scores differ by ≥2 points
- Thematic coding of disagreement patterns
- Identify systematic LLM biases (e.g., verbosity bias, position bias)

---

## 5. Expected Outputs

### 5.1 Quantitative Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Krippendorff's α (humans) | TBD | Inter-rater reliability |
| Spearman's ρ (human-LLM) | TBD | Correlation |
| Cohen's κ (human-LLM) | TBD | Agreement |
| MAE (human-LLM) | TBD | Score difference |

### 5.2 Qualitative Findings

- Themes in high-quality recommendations
- Common failure modes
- Domain-specific insights for O2C process

---

## 6. Scripts and Usage

### 6.1 Log Processor

```bash
# Process all O2C logs
python -m src.evaluation.qualitative.log_processor

# Process single log
python -m src.evaluation.qualitative.log_processor --single-log 744181388
```

### 6.2 Run Qualitative Evaluation

```bash
# Run pipeline on all O2C logs and generate evaluation package
python -m src.evaluation.qualitative.run_qualitative_evaluation

# Run on single log
python -m src.evaluation.qualitative.run_qualitative_evaluation --single-log 744181388

# Custom model and iterations
python -m src.evaluation.qualitative.run_qualitative_evaluation --model gpt-4.1 --max-iterations 3

# Regenerate forms from existing data (skip pipeline)
python -m src.evaluation.qualitative.run_qualitative_evaluation --skip-pipeline
```

### 6.3 Output Files

After running the evaluation, you'll find in `reports/qualitative/`:

| File | Purpose |
|------|---------|
| `reports/*.txt` | Full improvement reports per log |
| `evaluation_data.json` | Structured data for analysis |
| `google_forms_content.txt` | Ready-to-paste content for Google Forms |
| `llm_judge_prompts.json` | Prompts for LLM-as-judge evaluation |

---

## 7. Google Forms Structure

### 7.1 Form Overview

The Google Form should be structured as follows:

```
FORM TITLE: Evaluation of AI-Generated Process Improvement Recommendations

SECTIONS:
├── Section 1: Introduction & Instructions
├── Section 2: Evaluator Information
├── Section 3-N: One section per recommendation
│   ├── Process Context (read-only description)
│   ├── Recommendation Text (read-only description)
│   └── 5 Likert Questions + 1 Optional Comment
└── Section N+1: Closing & Thank You
```

### 7.2 Form Sections in Detail

#### Section 1: Introduction
**Type:** Description only (no questions)

**Content:**
```
Thank you for participating in this evaluation of LLM-generated process
improvement recommendations for Order-to-Cash business processes.

You will be shown recommendations generated by an AI system. For each one,
you'll see:
• Process context (statistics about the event log)
• The recommendation (what change is proposed and why)
• Simulation results (predicted impact)

Please rate each recommendation on 5 dimensions using a 1-5 scale.

Estimated time: 15-20 minutes
```

#### Section 2: Evaluator Information
| Question | Type | Options |
|----------|------|---------|
| Your expertise area | Multiple choice | BPM Researcher / Industry Practitioner / Process Mining Expert / Other |
| Years of experience | Multiple choice | 0-2 / 3-5 / 6-10 / 10+ |
| Familiarity with O2C processes | Linear scale | 1 (None) to 5 (Expert) |

#### Section 3-N: Recommendation Evaluation (repeat for each)

**Part A: Context (Description field)**
```
=== PROCESS CONTEXT ===
LOG: [log_name]
• Total Cases: X
• Total Events: Y
• Unique Activities: Z
• Top Activities: [list]
```

**Part B: Recommendation (Description field)**
```
=== RECOMMENDATION ===
DESCRIPTION: [description]
TYPE: [type]
RATIONALE: [rationale]
PROPOSED CHANGE: [target] → [change] (value: [value])
SIMULATION RESULTS: [DES/ABS summaries]
IMPACT: [impact summary]
```

**Part C: Evaluation Questions**

| Question | Type | Scale |
|----------|------|-------|
| Q1. CORRECTNESS: Is this insight factually accurate given the process data? | Linear scale | 1 (Incorrect) → 5 (Fully accurate) |
| Q2. RELEVANCE: Does it address a real bottleneck or inefficiency? | Linear scale | 1 (Irrelevant) → 5 (Highly relevant) |
| Q3. ACTIONABILITY: Can this recommendation be implemented in practice? | Linear scale | 1 (Impossible) → 5 (Immediately actionable) |
| Q4. SPECIFICITY: Is the recommendation concrete or vague? | Linear scale | 1 (Very vague) → 5 (Highly specific) |
| Q5. POTENTIAL IMPACT: Would implementing this meaningfully improve KPIs? | Linear scale | 1 (No impact) → 5 (Transformative) |
| Q6. COMMENTS (Optional) | Long answer | Free text |

### 7.3 Creating the Form

1. Go to [Google Forms](https://forms.google.com)
2. Create new form with title "Process Improvement Recommendation Evaluation"
3. Run `python -m src.evaluation.qualitative.run_qualitative_evaluation`
4. Open `reports/qualitative/google_forms_content.txt`
5. Copy each section into the form following the structure above
6. Set "Require response" for Q1-Q5
7. Enable "Collect email addresses" for reliability analysis
8. Share form with evaluators

### 7.4 Data Export

After collecting responses:
1. Go to Responses → Download responses (.csv)
2. Save as `reports/qualitative/human_evaluations.csv`
3. Use the analysis script (TBD) to compute agreement metrics

---

## 8. LLM-as-Judge Setup

### 8.1 Prompt Structure

The `llm_judge_prompts.json` file contains prompts structured for LLM evaluation:

```json
{
  "log_id": "744181388",
  "recommendation_id": 1,
  "system_prompt": "You are an expert BPM consultant...",
  "user_prompt": "Evaluate the following recommendation...",
  "recommendation_data": {...}
}
```

### 8.2 Running LLM Evaluation

```python
import json
from gen_ai_hub.proxy.native.openai import chat

# Load prompts
with open('reports/qualitative/llm_judge_prompts.json') as f:
    prompts = json.load(f)

# Use a DIFFERENT model than the one that generated recommendations
# e.g., if GPT-4 generated, use Claude for judging
JUDGE_MODEL = 'anthropic--claude-4-sonnet'  # Different family!

results = []
for prompt in prompts:
    response = chat.completions.create(
        model_name=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": prompt['system_prompt']},
            {"role": "user", "content": prompt['user_prompt']}
        ],
        temperature=0.0
    )
    # Parse JSON response
    scores = json.loads(response.choices[0].message.content)
    results.append({
        'log_id': prompt['log_id'],
        'recommendation_id': prompt['recommendation_id'],
        **scores
    })
```

### 8.3 Important: Avoiding Self-Enhancement Bias

**Critical:** Use a different model family for judging than for generating:

| Generation Model | Judge Model |
|-----------------|-------------|
| GPT-4.1 | Claude 4 Sonnet |
| Claude 4 Opus | GPT-4.1 |
| Mistral Large | GPT-4.1 or Claude |

---

## 9. Relationship to Quantitative Evaluation

| Aspect | Quantitative | Qualitative |
|--------|--------------|-------------|
| **Data** | Synthetic fault injection | Real-world O2C logs |
| **Ground truth** | Known injected faults | Expert judgment |
| **Metrics** | Precision, recall, RMSE | Agreement, correlation |
| **Strength** | Controlled, reproducible | Ecological validity |
| **Weakness** | Artificial scenarios | Subjective, expensive |

**Complementary claims:**
- Quantitative: "The system can detect process faults accurately"
- Qualitative: "The system produces recommendations experts find useful"

---

## 8. Limitations

1. **Sample size**: 9 O2C logs from single domain may limit generalizability
2. **Expert availability**: Recruiting qualified evaluators is challenging
3. **Subjectivity**: Even with rubric, expert opinions may vary
4. **LLM-as-judge limitations**: Known biases in LLM evaluation

---

## 9. Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Log processing | Complete | Processed CSV files |
| Pipeline runs | 1-2 days | Recommendations per log |
| Expert recruitment | 1 week | 3-5 qualified evaluators |
| Human evaluation | 2 weeks | Scored rubrics |
| LLM evaluation | 1 day | Automated scores |
| Analysis | 1 week | Statistical results + themes |

---

## 10. File Structure

```
src/evaluation/qualitative/
├── README.md                       # This document
├── log_processor.py                # O2C log transformation script
├── run_qualitative_evaluation.py   # Main evaluation runner
└── __init__.py

reports/qualitative/                # Generated after running evaluation
├── reports/
│   ├── report_742427502.txt
│   ├── report_743866877.txt
│   └── ...
├── evaluation_data.json            # Structured data for analysis
├── google_forms_content.txt        # Ready-to-paste form content
└── llm_judge_prompts.json          # Prompts for LLM evaluation

logs/o2c/
├── 742427502/
│   ├── raw_742427502.csv
│   └── processed_742427502.csv
├── 743866877/
│   ├── 743866877.csv
│   └── processed_743866877.csv
└── ...
```
