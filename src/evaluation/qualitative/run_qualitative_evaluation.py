"""
Qualitative Evaluation Runner

Generates Google Forms content for human expert evaluation of
AI-generated process improvement recommendations.

This script generates the evaluation package (forms content, log statistics).
The actual pipeline should be run separately via src/main.py.

Usage:
1. Run the main pipeline for each log:
   python src/main.py --log "o2c/745347473/processed_745347473.csv" --max_iterations 3
   python src/main.py --log "o2c/744204018/processed_744204018.csv" --max_iterations 3

2. Run this script to generate evaluation forms:
   python -m src.evaluation.qualitative.run_qualitative_evaluation
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent.parent
sys.path.insert(0, str(src_dir))


# Log descriptions for the evaluation
LOG_DESCRIPTIONS = {
    '745347473': """**Process A: Order-to-Cash Process (Specialty Chemicals Manufacturing)**

This log captures an Order-to-Cash (O2C) process from a specialty chemicals and organic intermediates manufacturer. The process focuses on billing and warehouse fulfillment operations.

**Industry:** Specialty Chemicals / Chemical Manufacturing

**Process scope:**
- Fulfillment: Delivery creation, picking, and goods issue
- Billing: Invoice creation and billing document changes
- Warehouse operations: Goods movement and delivery processing
- Sales: Quotation creation

**Data characteristics:**
- 4-month observation period (May - August 2025)
- 9,553 events across 2,890 cases (sampled from ~16k original cases)
- 12 distinct activities
- 9 resources across 3 organizational roles: BillingClerk (3), WarehouseStaff (3), SalesRep (3)
- Average 3.3 events per case

**Top activities by frequency:**
- Create Invoice (3,099), Post Goods Issue (1,373), Create Delivery (1,357)
- Change Billing Document variants (2,484 combined)
- Goods Movement/Picking/Delivery changes (1,137 combined)

The log represents a high-volume O2C process with established workflows dominated by billing and warehouse operations.""",

    '744204018': """**Process B: Sales-Focused Order-to-Cash Process**

This log captures a sales-intensive Order-to-Cash (O2C) process with comprehensive coverage of sales order management and billing operations.

**Industry:** General Manufacturing / Distribution

**Process scope:**
- Sales order management: Order creation, modifications, delivery date changes
- Billing: Invoice creation, pro forma invoices, invoice cancellation
- Order lifecycle: Document date changes, status updates, price changes

**Data characteristics:**
- 4-month observation period (April - August 2025, ~114 days)
- 1,756 events across 592 cases
- 16 distinct activities
- 6 resources across 2 organizational roles: SalesRep (3), BillingClerk (3)
- Average 3.0 events per case

**Top activities by frequency:**
- Create Invoice (401), Create Pro Forma Invoice (284)
- Change Sales Order variants: Delivery Date (197), Document Date (166), All Items (151)
- Create Sales Order/Item (271 combined)

The log represents an O2C process with significant sales order management activity, including frequent order modifications and delivery date adjustments."""
}


def extract_log_context(log_path: str) -> Dict[str, Any]:
    """
    Extract process context from a log file for evaluation.

    Returns summary statistics that evaluators need to understand the process.
    """
    df = pd.read_csv(log_path)

    # Basic statistics
    n_cases = df['case_id'].nunique()
    n_events = len(df)
    n_activities = df['activity_name'].nunique()
    n_resources = df['resource'].nunique() if 'resource' in df.columns else 0

    # Time range
    if 'start_timestamp' in df.columns:
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], utc=True)
        time_min = df['start_timestamp'].min()
        time_max = df['start_timestamp'].max()
        timespan_days = (time_max - time_min).days
    else:
        time_min = time_max = None
        timespan_days = 0

    # Activity frequency
    activity_freq = df['activity_name'].value_counts().head(10).to_dict()

    # Resource distribution
    if 'resource' in df.columns:
        resource_freq = df['resource'].value_counts().head(5).to_dict()
    else:
        resource_freq = {}

    # Events per case distribution
    events_per_case = df.groupby('case_id').size()
    avg_events_per_case = events_per_case.mean()

    return {
        'log_path': log_path,
        'log_name': os.path.basename(log_path),
        'n_cases': int(n_cases),
        'n_events': int(n_events),
        'n_activities': int(n_activities),
        'n_resources': int(n_resources),
        'timespan_days': int(timespan_days),
        'time_range': {
            'start': str(time_min) if time_min else None,
            'end': str(time_max) if time_max else None
        },
        'avg_events_per_case': round(avg_events_per_case, 2),
        'top_activities': activity_freq,
        'top_resources': resource_freq
    }


def extract_recommendations(validated_improvements: List[Dict]) -> List[Dict[str, Any]]:
    """
    Extract individual recommendations from validated improvements.

    Each recommendation becomes a separate evaluation item.
    """
    recommendations = []

    for idx, imp in enumerate(validated_improvements):
        hypothesis = imp.get('hypothesis', {}) or {}
        impact = imp.get('impact') or imp.get('improvement', {}) or {}
        changes = (hypothesis.get('changes', {}) or {}).get('specific_changes', []) or []
        first_change = changes[0] if changes and isinstance(changes[0], dict) else {}

        des_validation = impact.get('des_validation', {}) or {}
        abs_validation = impact.get('abs_validation', {}) or {}

        # Extract metric improvements
        metrics = {}
        for metric, data in impact.get('improvements', {}).items():
            if isinstance(data, dict):
                metrics[metric] = {
                    'baseline': data.get('baseline'),
                    'new': data.get('new'),
                    'change_pct': data.get('change_pct')
                }

        rec = {
            'recommendation_id': idx + 1,
            'iteration': imp.get('iteration', idx + 1),
            'description': hypothesis.get('description', 'N/A'),
            'improvement_type': hypothesis.get('improvement_type', 'N/A'),
            'rationale': hypothesis.get('rationale', 'N/A'),
            'target': first_change.get('target', 'N/A'),
            'change': first_change.get('change', 'N/A'),
            'change_value': first_change.get('value', 'N/A'),
            'impact_summary': impact.get('summary', 'N/A'),
            'des_summary': des_validation.get('summary', 'N/A'),
            'abs_summary': abs_validation.get('summary', 'N/A'),
            'metrics': metrics,
            'is_improvement': bool(impact.get('is_improvement', False))
        }
        recommendations.append(rec)

    return recommendations


def format_recommendation_for_eval(rec: Dict, context: Dict = None) -> str:
    """
    Format a recommendation for human evaluation.

    Creates a readable text block that can be pasted into forms.
    """
    text = f"""
=== RECOMMENDATION {rec['recommendation_id']} ===

DESCRIPTION:
{rec['description']}

TYPE: {rec['improvement_type']}

RATIONALE:
{rec['rationale']}

PROPOSED CHANGE:
- Target: {rec['target']}
- Change: {rec['change']}
- Value: {rec['change_value']}

SIMULATION RESULTS:
- DES (Discrete Event Simulation): {rec['des_summary']}
- ABS (Agent-Based Simulation): {rec['abs_summary']}

IMPACT SUMMARY:
{rec['impact_summary']}
"""

    if rec['metrics']:
        text += "\nMETRIC CHANGES:\n"
        for metric, data in rec['metrics'].items():
            if data.get('baseline') is not None and data.get('new') is not None:
                text += f"  - {metric}: {data['baseline']:.2f} → {data['new']:.2f} ({data['change_pct']:+.1f}%)\n"

    return text.strip()


def format_context_for_eval(context: Dict) -> str:
    """
    Format process context for human/LLM evaluation.
    """
    text = f"""
=== PROCESS CONTEXT ===

LOG: {context['log_name']}

STATISTICS:
- Total Cases: {context['n_cases']:,}
- Total Events: {context['n_events']:,}
- Unique Activities: {context['n_activities']}
- Unique Resources: {context['n_resources']}
- Timespan: {context['timespan_days']} days
- Avg Events per Case: {context['avg_events_per_case']}

TOP ACTIVITIES (by frequency):
"""
    for activity, count in list(context['top_activities'].items())[:7]:
        text += f"  - {activity}: {count:,} events\n"

    if context['top_resources']:
        text += "\nTOP RESOURCES:\n"
        for resource, count in context['top_resources'].items():
            text += f"  - {resource}: {count:,} events\n"

    return text.strip()


def generate_evaluation_package(results: List[Dict], output_dir: Path) -> None:
    """
    Generate all files needed for the evaluation.

    Creates:
    1. Combined evaluation data JSON
    2. Google Forms-ready text file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Save combined evaluation data (for analysis)
    eval_data = {
        'generated_at': datetime.now().isoformat(),
        'n_logs': len(results),
        'n_total_recommendations': sum(r['n_recommendations'] for r in results),
        'logs': []
    }

    for result in results:
        log_data = {
            'log_id': result['log_id'],
            'context': result['context'],
            'baseline_metrics': result['baseline_metrics'],
            'recommendations': result['recommendations'],
            'status': result['status']
        }
        eval_data['logs'].append(log_data)

    with open(output_dir / 'evaluation_data.json', 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, default=str)

    # 3. Generate Google Forms-ready text
    forms_text = generate_google_forms_content(results)
    with open(output_dir / 'google_forms_content.txt', 'w', encoding='utf-8') as f:
        f.write(forms_text)

    print(f"\n{'='*60}")
    print("EVALUATION PACKAGE GENERATED")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"  - reports/          : {len([r for r in results if r['status']=='success'])} report files")
    print(f"  - evaluation_data.json : Combined data for analysis")
    print(f"  - google_forms_content.txt : Ready-to-paste form content")
    print(f"\nTotal recommendations to evaluate: {eval_data['n_total_recommendations']}")


def generate_google_forms_content(results: List[Dict]) -> str:
    """
    Generate text content structured for Google Forms.

    New structure with independent baseline design:
    1. Introduction
    2. Background (demographics)
    3-4. Process A: Context + Expert suggestion, then AI recommendation evaluation
    5-6. Process B: Context + Expert suggestion, then AI recommendation evaluation
    7. Overall Assessment
    """
    content = """
================================================================================
QUALITATIVE EVALUATION - GOOGLE FORMS CONTENT
================================================================================

Copy this content into Google Forms (forms.google.com) section by section.
Estimated completion time: 12-15 minutes

================================================================================
SECTION 1: INTRODUCTION
================================================================================

Form Title: Evaluation of AI-Generated Process Improvement Recommendations

Form Description:
Thank you for participating in this evaluation!

We are developing an AI-assisted approach for business process improvement that:
1. Analyzes event logs from enterprise systems
2. Identifies potential inefficiencies
3. Proposes concrete improvements
4. Validates proposals through simulation

Your expert feedback will help us understand:
- Whether AI-generated recommendations address real problems
- How AI suggestions compare to expert intuition
- The practical utility of this approach

**What you'll do:**
- Review 2 real-world Order-to-Cash process logs
- Share what improvements YOU would suggest (before seeing AI output)
- Evaluate the AI-generated recommendations
- Provide overall feedback

**Time required:** ~15 minutes

All responses are anonymous and will be used for academic research only.

================================================================================
SECTION 2: YOUR BACKGROUND
================================================================================

Q1. What is your current role? (Multiple choice)
- Process Analyst
- Business Consultant
- SAP Functional Expert
- BPM Researcher
- Process Mining Specialist
- Other: ____________

Q2. Years of experience with business processes (Multiple choice)
- 0-2 years
- 3-5 years
- 6-10 years
- 10+ years

Q3. How familiar are you with Order-to-Cash (O2C) processes? (Linear scale 1-5)
1 = Not familiar at all ... 5 = Expert

Q4. How familiar are you with process mining tools? (Linear scale 1-5)
1 = Not familiar at all ... 5 = Expert

"""

    # Process each log - map log_id to process label
    log_id_to_label = {
        '745347473': 'A',  # Full O2C (Specialty Chemicals)
        '744204018': 'B',  # Sales-Focused O2C
    }

    for idx, result in enumerate(results):
        if result['status'] != 'success':
            continue

        log_id = result['log_id']
        context = result['context']
        recommendations = result['recommendations']
        process_label = log_id_to_label.get(log_id, f'{idx+1}')

        # Get log description
        log_description = LOG_DESCRIPTIONS.get(log_id, "No description available.")

        # Section: Process Context + Expert's Own Suggestion (BEFORE AI)
        section_num = 3 + (idx * 2)
        content += f"""
================================================================================
SECTION {section_num}: PROCESS {process_label} - CONTEXT & YOUR SUGGESTIONS
================================================================================

--- PROCESS DESCRIPTION ---

{log_description}

--- PROCESS STATISTICS ---

| Metric | Value |
|--------|-------|
| Total Cases | {context['n_cases']:,} |
| Total Events | {context['n_events']:,} |
| Unique Activities | {context['n_activities']} |
| Unique Resources | {context['n_resources']} |
| Observation Period | {context['timespan_days']} days |
| Avg Events per Case | {context['avg_events_per_case']} |

**Top Activities (by frequency):**
"""
        for activity, count in list(context['top_activities'].items())[:7]:
            content += f"- {activity}: {count:,} events\n"

        if context['top_resources']:
            content += "\n**Resources:**\n"
            for resource, count in list(context['top_resources'].items()):
                content += f"- {resource}: {count:,} events\n"

        content += f"""

--- YOUR ANALYSIS (Before seeing AI output) ---

Q{section_num}.1 Based on this data, what process inefficiencies or bottlenecks would you expect to find?
[Long text field - REQUIRED]
(Hint: Consider activity frequencies, resource distribution, process flow patterns)

Q{section_num}.2 If you were asked to improve this process, what would be your first recommendation?
[Long text field - REQUIRED]
(Be as specific as possible about what you would change and why)

"""

        # Section: AI Recommendation Evaluation
        section_num_ai = section_num + 1

        content += f"""
================================================================================
SECTION {section_num_ai}: PROCESS {process_label} - AI RECOMMENDATION EVALUATION
================================================================================

--- HOW THE AI WORKS ---

Our system analyzes the event log and:
1. Discovers process patterns and resource behaviors
2. Identifies potential bottlenecks using process mining
3. Generates improvement hypotheses using an LLM
4. Validates each hypothesis through discrete-event and agent-based simulation
5. Reports improvements that show positive impact in simulation

Below is the recommendation the system generated for Process {process_label}.

"""

        if recommendations:
            rec = recommendations[0]  # Single recommendation per log
            content += f"""--- AI-GENERATED RECOMMENDATION ---

**Description:**
{rec['description']}

**Type:** {rec['improvement_type']}

**Rationale:**
{rec['rationale']}

**Proposed Change:**
- Target: {rec['target']}
- Change: {rec['change']}
- Value: {rec['change_value']}

**Simulation Results:**
- Discrete Event Simulation (DES): {rec['des_summary']}
- Agent-Based Simulation (ABS): {rec['abs_summary']}

**Impact Summary:**
{rec['impact_summary']}
"""
            if rec['metrics']:
                content += "\n**Predicted Metric Changes:**\n"
                for metric, data in rec['metrics'].items():
                    if data.get('baseline') is not None and data.get('new') is not None:
                        content += f"- {metric}: {data['baseline']:.2f} → {data['new']:.2f} ({data['change_pct']:+.1f}%)\n"
        else:
            content += """--- AI-GENERATED RECOMMENDATION ---

[No recommendation was generated for this process. The system did not identify improvements that showed positive impact in simulation.]
"""

        content += f"""

--- EVALUATION QUESTIONS ---

Q{section_num_ai}.1 Problem Validity: Does this recommendation address a real problem in the process?
(Linear scale 1-5)
1 = Not at all ... 5 = Definitely yes

Q{section_num_ai}.2 Practicality: Is this recommendation practical to implement in a real organization?
(Linear scale 1-5)
1 = Not practical at all ... 5 = Very practical

Q{section_num_ai}.3 Expected Impact: Would implementing this meaningfully improve the process?
(Linear scale 1-5)
1 = No improvement expected ... 5 = Significant improvement expected

Q{section_num_ai}.4 Alignment: Would you have suggested the same or similar improvement?
(Multiple choice)
- Yes, exactly the same
- Yes, similar direction
- Partially overlaps with my thinking
- No, I would suggest something different

Q{section_num_ai}.5 What would you have done differently? (Optional)
[Long text field]

Q{section_num_ai}.6 Any other comments about this recommendation? (Optional)
[Long text field]

"""

    # Overall Assessment Section
    content += """
================================================================================
SECTION 7: OVERALL ASSESSMENT
================================================================================

Q7.1 How useful would this AI-assisted approach be for process improvement work?
(Linear scale 1-5)
1 = Not useful at all ... 5 = Extremely useful

Q7.2 What did the system do well? (Required)
[Long text field]

Q7.3 What did the system miss or get wrong? (Required)
[Long text field]

Q7.4 Would you consider using a tool like this in your practice?
(Multiple choice)
- Yes, definitely
- Maybe, with improvements
- No

Q7.5 Why or why not? (Optional)
[Long text field]

Q7.6 Any additional feedback or suggestions? (Optional)
[Long text field]

================================================================================
SECTION 8: THANK YOU
================================================================================

Thank you for completing this evaluation!

Your expert feedback is invaluable for understanding the real-world utility of
AI-assisted process improvement approaches.

If you have any questions about this research, please contact: [researcher email]

================================================================================
END OF FORM CONTENT
================================================================================
"""

    return content


def main():
    """
    Generate Google Forms content from existing pipeline reports.

    Usage:
    1. First run the main pipeline for each log:
       python src/main.py --log "o2c/745347473/processed_745347473.csv" --max_iterations 3
       python src/main.py --log "o2c/744181388/processed_744181388.csv" --max_iterations 3

    2. Then run this script to generate evaluation package:
       python -m src.evaluation.qualitative.run_qualitative_evaluation
    """
    parser = argparse.ArgumentParser(
        description='Generate qualitative evaluation package from existing reports'
    )
    parser.add_argument('--logs-dir', type=str, default='logs/o2c',
                        help='Directory containing O2C log folders')
    parser.add_argument('--output-dir', type=str, default='reports/qualitative',
                        help='Output directory for evaluation package')

    args = parser.parse_args()

    # Resolve paths
    project_root = script_dir.parent.parent.parent
    logs_dir = project_root / args.logs_dir
    output_dir = project_root / args.output_dir

    # Find processed logs and build results from log context only
    # (reports are generated separately via main.py)
    results = []

    # Target logs for evaluation
    target_logs = ['745347473', '744204018']

    for log_id in target_logs:
        log_dir = logs_dir / log_id
        if not log_dir.exists():
            print(f"WARNING: Log directory not found: {log_dir}")
            continue

        # Find processed CSV
        processed_files = list(log_dir.glob('processed_*.csv'))
        if not processed_files:
            print(f"WARNING: No processed log found in {log_dir}")
            continue

        log_path = str(processed_files[0])
        print(f"Processing: {log_path}")

        # Extract context from log
        context = extract_log_context(log_path)

        result = {
            'log_id': log_id,
            'log_path': log_path,
            'context': context,
            'baseline_metrics': {},  # Will be in the main report
            'recommendations': [],   # Will be in the main report
            'full_report': '',
            'n_recommendations': 0,
            'status': 'success'
        }
        results.append(result)

    if not results:
        print("ERROR: No logs found. Check logs/o2c/ directory.")
        return

    print(f"\nFound {len(results)} logs for evaluation")

    # Generate evaluation package (Google Forms content)
    generate_evaluation_package(results, output_dir)

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Run the main pipeline for each log to generate reports:")
    print('   cd src && poetry run python main.py --log "o2c/745347473/processed_745347473.csv" --max_iterations 3')
    print('   cd src && poetry run python main.py --log "o2c/744204018/processed_744204018.csv" --max_iterations 3')
    print("\n2. Copy the AI recommendations from the generated reports into")
    print(f"   {output_dir / 'google_forms_content.txt'}")
    print("\n3. Create the Google Form at forms.google.com")


if __name__ == '__main__':
    main()
