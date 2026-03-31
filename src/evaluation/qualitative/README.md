# Qualitative Evaluation Framework for LLM-Guided Process Improvement

**Document Version:** 2.1
**Date:** March 30, 2026
**Purpose:** Human expert evaluation for journal paper submission

---

## 1. Overview

This evaluation complements the quantitative benchmarks by assessing whether the LLM-generated process improvement recommendations are **actionable** and **meaningful** to domain experts. While quantitative evaluation measures detection accuracy and statistical performance, qualitative evaluation measures real-world utility through expert feedback.

### Research Questions

1. **RQ1:** Are LLM-generated process improvement recommendations perceived as actionable by domain experts?
2. **RQ2:** Do expert-identified improvement opportunities align with AI-generated recommendations?
3. **RQ3:** Would domain experts trust and use an AI-assisted process improvement tool?

---

## 2. Dataset: Order-to-Cash (O2C) Event Logs

### 2.1 Data Source

Real-world SAP Order-to-Cash process logs from production systems, anonymized for evaluation.

### 2.2 Log Processing

Raw O2C logs are transformed to the standard format compatible with Prosimos and AgentSimulator using `log_processor.py`.

**Input format (raw):**
- `c_caseid` - Case identifier
- `c_eventname` - Activity name
- `c_time` - Event timestamp
- `EventCreatedByUserId` - Resource identifier

**Output format (processed):**
- `case_id` - Integer case identifier (0-indexed)
- `resource` - Cleaned resource name (role-based: SalesRep_1, BillingClerk_2, etc.)
- `activity_name` - Activity name
- `start_timestamp` - Event timestamp (ISO format with UTC timezone)
- `end_timestamp` - Same as start (events are instantaneous)
- `agent` - Integer agent ID

### 2.3 Selected Logs for Evaluation

Two logs were selected to provide variety in process complexity and industry:

| Log ID | Industry | Events | Characteristics |
|--------|----------|--------|-----------------|
| **745347473** | Specialty Chemicals / Chemical Manufacturing | 9,553 | Billing and warehouse operations |
| **744204018** | General Manufacturing / Distribution | 1,756 | Sales-focused O2C with order management |

**Log 745347473 (Process A)** - Specialty Chemicals Manufacturing:
- Order-to-Cash process focused on billing and warehouse operations
- 9 resources across 3 roles: BillingClerk (3), WarehouseStaff (3), SalesRep (3)
- ~4 months of data (May - August 2025)
- 2,890 cases (sampled from ~16k), 9,553 events, 12 activities

**Log 744204018 (Process B)** - Sales-Focused O2C:
- Sales order management and billing focused process
- 6 resources across 2 roles: SalesRep (3), BillingClerk (3)
- ~4 months of data (April - August 2025, ~114 days)
- 592 cases, 1,756 events, 16 activities

---

## 3. Evaluation Methodology

### 3.1 Study Design

**Human expert evaluation with independent baseline assessment:**

1. Show process context (log statistics) to expert
2. Ask expert what improvements they would suggest (captures independent baseline)
3. Explain the AI methodology briefly
4. Present AI-generated recommendation
5. Expert evaluates recommendation quality and alignment with their own thinking
6. Collect overall feedback on the approach

This design allows comparison between expert intuition and AI recommendations.

### 3.2 Participant Criteria

**Target:** 5-8 evaluators

**Required expertise (at least one):**
- Process Analyst with O2C experience
- SAP Functional Consultant
- BPM/Process Mining Practitioner
- Business Process Researcher

**Recruitment:** Internal SAP experts in Process Mining, ETL Process Mining, ranging from Senior Manager to Technical Expert.

### 3.3 Evaluation Dimensions

Each recommendation is scored on 3 core dimensions (1-5 Likert scale):

| Dimension | Question | Scale |
|-----------|----------|-------|
| **Problem Validity** | Does this recommendation address a real problem in the process? | 1 = Not at all → 5 = Definitely yes |
| **Practicality** | Is this recommendation practical to implement in a real organization? | 1 = Not practical → 5 = Very practical |
| **Expected Impact** | Would implementing this meaningfully improve the process? | 1 = No improvement → 5 = Significant improvement |

**Additional alignment question:**
- "Would you have suggested the same or a similar improvement?" (Yes, exactly / Yes, similar / Partially / No, different)

---

## 4. Google Form Structure

### 4.1 Form Overview

```
FORM TITLE: Evaluation of AI-Generated Process Improvement Recommendations

SECTIONS:
├── Section 1: Introduction
├── Section 2: Your Background (demographics)
├── Section 3: Process A - Context + Expert's Own Suggestion
├── Section 4: Process A - AI Recommendation Evaluation
├── Section 5: Process B - Context + Expert's Own Suggestion
├── Section 6: Process B - AI Recommendation Evaluation
├── Section 7: Overall Assessment
└── Section 8: Thank You
```

**Estimated completion time:** 12-15 minutes

### 4.2 Section Details

#### Section 1: Introduction
Description explaining:
- Study purpose (AI-assisted process improvement)
- What they will do (review 2 processes, evaluate AI recommendations)
- Time estimate (~15 minutes)
- Data usage (academic research, anonymized)

#### Section 2: Your Background

| Question | Type | Options |
|----------|------|---------|
| Current role | Multiple choice | Process Analyst / Business Consultant / SAP Functional Expert / BPM Researcher / Process Mining Specialist / Other |
| Years of experience with business processes | Multiple choice | 0-2 / 3-5 / 6-10 / 10+ years |
| Familiarity with O2C processes | Linear scale 1-5 | 1 = Not familiar → 5 = Expert |
| Familiarity with process mining tools | Linear scale 1-5 | 1 = Not familiar → 5 = Expert |

#### Section 3: Process A - Context (BEFORE showing AI output)

Show log statistics table, then ask:
- "Based on this data, what process inefficiencies or bottlenecks would you expect to find?" (free text, required)
- "If you were asked to improve this process, what would be your first recommendation?" (free text, required)

#### Section 4: Process A - AI Recommendation

Brief methodology explanation, then show recommendation with:
- Description
- Rationale
- Proposed change
- Simulated impact

Questions:
- Problem validity (1-5)
- Practicality (1-5)
- Expected impact (1-5)
- Alignment with your suggestion (Yes exactly / Yes similar / Partially / No different)
- What would you have done differently? (free text, optional)
- Additional comments (free text, optional)

#### Sections 5-6: Repeat for Process B

Same structure as Sections 3-4.

#### Section 7: Overall Assessment

| Question | Type |
|----------|------|
| How useful would this AI-assisted approach be for process improvement work? | Linear scale 1-5 |
| What did the system do well? | Free text (required) |
| What did the system miss or get wrong? | Free text (required) |
| Would you consider using a tool like this in your practice? | Multiple choice: Yes / Maybe / No |
| Why or why not? | Free text (optional) |
| Any additional feedback? | Free text (optional) |

### 4.3 Creating the Form

**Step 1: Run the main pipeline for each log**

```bash
cd src
poetry run python main.py --log "o2c/745347473/processed_745347473.csv" --max_iterations 3
poetry run python main.py --log "o2c/744204018/processed_744204018.csv" --max_iterations 3
```

This generates the improvement reports in `reports/`.

**Step 2: Generate the Google Forms content**

```bash
python -m src.evaluation.qualitative.run_qualitative_evaluation
```

This creates `reports/qualitative/google_forms_content.txt` with log statistics and form structure.

**Step 3: Manually add AI recommendations**

Copy the relevant AI recommendation sections from the pipeline reports (`reports/improvement_report_*.txt`) into the Google Forms content where indicated.

**Step 4: Create Google Form**

1. Go to [forms.google.com](https://forms.google.com)
2. Copy content section by section from `google_forms_content.txt`
3. Configure settings:
   - Collect email addresses: Optional
   - Show progress bar: Yes
   - Require responses for key questions
4. Distribute to expert evaluators

---

## 5. Data Analysis

### 5.1 Quantitative Metrics

| Metric | What it measures |
|--------|------------------|
| Mean scores per dimension | Overall perceived quality |
| Standard deviation | Agreement among experts |
| % "Would have suggested same/similar" | Alignment with expert thinking |
| Correlation: expertise level vs. scores | Does experience affect perception? |

### 5.2 Qualitative Analysis

**Thematic coding of free-text responses:**
- What improvements did experts suggest independently?
- Where did expert suggestions align/differ from AI?
- What did experts identify as system strengths?
- What gaps or limitations did experts note?

### 5.3 Claims for Paper

Based on results, the paper can make claims such as:
- "X% of domain experts rated the AI recommendations as addressing real problems (≥4/5)"
- "Y% of experts indicated they would have proposed similar improvements"
- "Experts identified [themes] as key strengths of the approach"
- "Experts noted [themes] as areas requiring improvement"

---

## 6. Scripts and Usage

### 6.1 Log Processor

```bash
# Process O2C logs to standard format
python -m src.evaluation.qualitative.log_processor
```

### 6.2 Generate Evaluation Package

```bash
# Generate Google Forms content from log statistics
python -m src.evaluation.qualitative.run_qualitative_evaluation
```

### 6.3 Output Files

After running, find outputs in `reports/qualitative/`:

| File | Purpose |
|------|---------|
| `evaluation_data.json` | Structured log statistics for analysis |
| `google_forms_content.txt` | Ready-to-paste Google Forms content |

Pipeline reports are in `reports/`:

| File | Purpose |
|------|---------|
| `improvement_report_*.txt` | Full improvement report per log (run via main.py) |

---

## 7. Relationship to Quantitative Evaluation

| Aspect | Quantitative | Qualitative |
|--------|--------------|-------------|
| **Data** | Synthetic fault injection | Real-world O2C logs |
| **Ground truth** | Known injected faults | Expert judgment |
| **Metrics** | Precision, recall, RMSE | Likert scores, thematic analysis |
| **Strength** | Controlled, reproducible | Ecological validity |
| **Weakness** | Artificial scenarios | Subjective, smaller sample |

**Complementary claims:**
- **Quantitative:** "The system can detect process faults accurately"
- **Qualitative:** "Domain experts find the recommendations actionable and useful"

---

## 8. Limitations

1. **Sample size:** 2 O2C logs from different industries may limit generalizability
2. **Expert pool:** Internal recruitment may introduce organizational bias
3. **Subjectivity:** Expert opinions inherently vary
4. **Single recommendation per log:** Does not test system behavior across multiple recommendations

---

## 9. File Structure

```
src/evaluation/qualitative/
├── README.md                       # This document
├── log_processor.py                # O2C log transformation script
├── run_qualitative_evaluation.py   # Generates Google Forms content
└── __init__.py

reports/                            # Pipeline reports (from main.py)
├── improvement_report_*.txt        # Full improvement reports

reports/qualitative/                # Generated after running evaluation script
├── evaluation_data.json            # Structured log statistics
└── google_forms_content.txt        # Ready-to-paste form content

logs/o2c/
├── 744204018/
│   └── processed_744204018.csv
└── 745347473/
    └── processed_745347473.csv
```
