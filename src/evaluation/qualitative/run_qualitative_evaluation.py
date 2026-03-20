"""
Qualitative Evaluation Runner

Runs the improvement validation pipeline on O2C logs and extracts
structured data for human/LLM evaluation.

Outputs:
1. Full reports (saved to reports/qualitative/)
2. Structured evaluation data (JSON) for Google Forms / LLM-as-judge
3. Process context summaries for each log
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent.parent
sys.path.insert(0, str(src_dir))

from improvement_validation_pipeline import ImprovementValidationAgent


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


def format_recommendation_for_eval(rec: Dict, context: Dict) -> str:
    """
    Format a recommendation for human/LLM evaluation.

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


def run_pipeline_for_log(
    log_path: str,
    model: str = 'gpt-4.1',
    temperature: float = 0.0,
    max_iterations: int = 3,
    des_runs: int = 5,
    abs_runs: int = 5
) -> Dict[str, Any]:
    """
    Run the improvement pipeline on a single log and extract evaluation data.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {log_path}")
    print(f"{'='*60}")

    # Extract context before running pipeline
    context = extract_log_context(log_path)
    print(f"  Cases: {context['n_cases']}, Events: {context['n_events']}, Activities: {context['n_activities']}")

    # Initialize agent
    agent = ImprovementValidationAgent(
        model_name=model,
        temperature=temperature,
        max_tokens=4096,
        max_iterations=max_iterations,
        use_sudo=False,  # Windows
        des_runs=des_runs,
        abs_runs=abs_runs,
        des_constrain_total_cases=False
    )

    # Run pipeline
    try:
        report = agent.run_improvement_pipeline(log_path)

        # Extract structured data from agent
        baseline_metrics = getattr(agent, 'baseline_metrics', {})
        validated_improvements = getattr(agent, 'validated_improvements', [])

        recommendations = extract_recommendations(validated_improvements)

        result = {
            'log_id': Path(log_path).parent.name,
            'log_path': log_path,
            'context': context,
            'baseline_metrics': baseline_metrics,
            'recommendations': recommendations,
            'full_report': report,
            'n_recommendations': len(recommendations),
            'status': 'success'
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        result = {
            'log_id': Path(log_path).parent.name,
            'log_path': log_path,
            'context': context,
            'baseline_metrics': {},
            'recommendations': [],
            'full_report': f"Pipeline failed: {str(e)}",
            'n_recommendations': 0,
            'status': 'error',
            'error': str(e)
        }

    return result


def generate_evaluation_package(results: List[Dict], output_dir: Path) -> None:
    """
    Generate all files needed for the evaluation.

    Creates:
    1. Individual report files
    2. Combined evaluation data JSON
    3. Google Forms-ready text file
    4. LLM-as-judge prompt file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save individual reports
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(exist_ok=True)

    for result in results:
        if result['status'] == 'success':
            report_path = reports_dir / f"report_{result['log_id']}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(result['full_report'])

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

    # 4. Generate LLM-as-judge prompts
    judge_prompts = generate_llm_judge_prompts(results)
    with open(output_dir / 'llm_judge_prompts.json', 'w', encoding='utf-8') as f:
        json.dump(judge_prompts, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("EVALUATION PACKAGE GENERATED")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"  - reports/          : {len([r for r in results if r['status']=='success'])} report files")
    print(f"  - evaluation_data.json : Combined data for analysis")
    print(f"  - google_forms_content.txt : Ready-to-paste form content")
    print(f"  - llm_judge_prompts.json : Prompts for LLM evaluation")
    print(f"\nTotal recommendations to evaluate: {eval_data['n_total_recommendations']}")


def generate_google_forms_content(results: List[Dict]) -> str:
    """
    Generate text content structured for Google Forms.
    """
    content = """
================================================================================
QUALITATIVE EVALUATION - GOOGLE FORMS CONTENT
================================================================================

This file contains the content to copy into Google Forms for human evaluation.

FORM STRUCTURE:
1. Introduction section (copy once)
2. For each log: Context section + Recommendation sections
3. Closing section (copy once)

================================================================================
SECTION 1: INTRODUCTION (Copy this to Form Description)
================================================================================

Thank you for participating in this evaluation of LLM-generated process
improvement recommendations.

You will be shown recommendations generated by an AI system for improving
business processes. For each recommendation, you'll see:
- The process context (statistics about the event log)
- The recommendation details (what change is proposed and why)
- Simulation results (predicted impact)

Please rate each recommendation on 5 dimensions using a 1-5 scale:

1. CORRECTNESS: Is the insight factually accurate given the process data?
   1=Incorrect, 5=Fully accurate

2. RELEVANCE: Does it address a real bottleneck or inefficiency?
   1=Irrelevant, 5=Highly relevant

3. ACTIONABILITY: Can this be implemented in practice?
   1=Impossible, 5=Immediately actionable

4. SPECIFICITY: Is the recommendation concrete or vague?
   1=Very vague, 5=Highly specific

5. POTENTIAL IMPACT: Would implementing this meaningfully improve KPIs?
   1=No impact, 5=Transformative

================================================================================
"""

    rec_counter = 0
    for result in results:
        if result['status'] != 'success' or not result['recommendations']:
            continue

        log_id = result['log_id']
        context = result['context']

        content += f"""
================================================================================
LOG: {log_id}
================================================================================

--- PROCESS CONTEXT (Show this before recommendations for this log) ---

{format_context_for_eval(context)}

"""

        for rec in result['recommendations']:
            rec_counter += 1
            content += f"""
--- RECOMMENDATION {rec_counter} (Log: {log_id}, Rec: {rec['recommendation_id']}) ---

{format_recommendation_for_eval(rec, context)}

[EVALUATION QUESTIONS - Create as Likert scale 1-5]

Q1. CORRECTNESS: Is this insight factually accurate given the process data?
    (1=Incorrect ... 5=Fully accurate)

Q2. RELEVANCE: Does it address a real bottleneck or inefficiency?
    (1=Irrelevant ... 5=Highly relevant)

Q3. ACTIONABILITY: Can this recommendation be implemented in practice?
    (1=Impossible ... 5=Immediately actionable)

Q4. SPECIFICITY: Is the recommendation concrete or vague?
    (1=Very vague ... 5=Highly specific)

Q5. POTENTIAL IMPACT: Would implementing this meaningfully improve KPIs?
    (1=No impact ... 5=Transformative)

Q6. COMMENTS (Optional): Any additional observations about this recommendation?
    [Free text field]

"""

    content += f"""
================================================================================
SECTION: CLOSING
================================================================================

Thank you for completing this evaluation!

Your responses will help us understand:
1. The quality of AI-generated process improvement recommendations
2. How well automated evaluation methods align with expert judgment

Total recommendations evaluated: {rec_counter}

================================================================================
"""

    return content


def generate_llm_judge_prompts(results: List[Dict]) -> List[Dict]:
    """
    Generate prompts for LLM-as-judge evaluation.

    Returns a list of prompts, one per recommendation.
    """
    prompts = []

    system_prompt = """You are an expert Business Process Management consultant evaluating process improvement recommendations.

Your task is to score each recommendation on 5 dimensions using a 1-5 scale.

Scoring criteria:
1. CORRECTNESS (1-5): Is the insight factually accurate given the process data?
   - 1: Contains factual errors or misinterprets the data
   - 3: Mostly accurate with minor issues
   - 5: Completely accurate and well-grounded in the data

2. RELEVANCE (1-5): Does it address a real bottleneck or inefficiency?
   - 1: Addresses non-issues or irrelevant aspects
   - 3: Addresses a moderate concern
   - 5: Targets a critical bottleneck

3. ACTIONABILITY (1-5): Can this be implemented in practice?
   - 1: Impossible or requires unrealistic changes
   - 3: Feasible with significant effort
   - 5: Can be implemented immediately

4. SPECIFICITY (1-5): Is the recommendation concrete or vague?
   - 1: Very vague, no clear action
   - 3: Somewhat specific
   - 5: Highly specific with clear parameters

5. POTENTIAL IMPACT (1-5): Would implementing this meaningfully improve KPIs?
   - 1: Negligible or no impact
   - 3: Moderate improvement
   - 5: Transformative impact

Respond in JSON format:
{
  "correctness": <1-5>,
  "correctness_justification": "<brief explanation>",
  "relevance": <1-5>,
  "relevance_justification": "<brief explanation>",
  "actionability": <1-5>,
  "actionability_justification": "<brief explanation>",
  "specificity": <1-5>,
  "specificity_justification": "<brief explanation>",
  "impact": <1-5>,
  "impact_justification": "<brief explanation>"
}"""

    for result in results:
        if result['status'] != 'success' or not result['recommendations']:
            continue

        context = result['context']
        context_text = format_context_for_eval(context)

        for rec in result['recommendations']:
            rec_text = format_recommendation_for_eval(rec, context)

            user_prompt = f"""Evaluate the following process improvement recommendation.

{context_text}

{rec_text}

Provide your evaluation scores and justifications in JSON format."""

            prompts.append({
                'log_id': result['log_id'],
                'recommendation_id': rec['recommendation_id'],
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'recommendation_data': rec
            })

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description='Run qualitative evaluation on O2C logs'
    )
    parser.add_argument('--logs-dir', type=str, default='logs/o2c',
                        help='Directory containing O2C log folders')
    parser.add_argument('--output-dir', type=str, default='reports/qualitative',
                        help='Output directory for evaluation package')
    parser.add_argument('--model', type=str, default='gpt-4.1',
                        help='LLM model for hypothesis generation')
    parser.add_argument('--max-iterations', type=int, default=3,
                        help='Max improvement iterations per log')
    parser.add_argument('--des-runs', type=int, default=5,
                        help='Number of DES runs')
    parser.add_argument('--abs-runs', type=int, default=5,
                        help='Number of ABS runs')
    parser.add_argument('--single-log', type=str,
                        help='Process only this log ID')
    parser.add_argument('--skip-pipeline', action='store_true',
                        help='Skip pipeline, only generate forms from existing data')

    args = parser.parse_args()

    # Resolve paths
    project_root = script_dir.parent.parent.parent
    logs_dir = project_root / args.logs_dir
    output_dir = project_root / args.output_dir

    if args.skip_pipeline:
        # Load existing evaluation data
        eval_data_path = output_dir / 'evaluation_data.json'
        if not eval_data_path.exists():
            print(f"ERROR: No existing data at {eval_data_path}")
            return

        with open(eval_data_path, 'r') as f:
            eval_data = json.load(f)

        # Reconstruct results format
        results = []
        for log_data in eval_data['logs']:
            results.append({
                'log_id': log_data['log_id'],
                'log_path': log_data['context']['log_path'],
                'context': log_data['context'],
                'baseline_metrics': log_data['baseline_metrics'],
                'recommendations': log_data['recommendations'],
                'full_report': '',
                'n_recommendations': len(log_data['recommendations']),
                'status': log_data['status']
            })

        generate_evaluation_package(results, output_dir)
        return

    # Find processed logs
    processed_logs = []
    for subdir in sorted(logs_dir.iterdir()):
        if not subdir.is_dir():
            continue

        if args.single_log and subdir.name != args.single_log:
            continue

        # Find processed CSV
        processed_files = list(subdir.glob('processed_*.csv'))
        if processed_files:
            processed_logs.append(str(processed_files[0]))

    if not processed_logs:
        print("ERROR: No processed logs found. Run log_processor.py first.")
        return

    print(f"Found {len(processed_logs)} processed logs")

    # Run pipeline on each log
    results = []
    for log_path in processed_logs:
        result = run_pipeline_for_log(
            log_path=log_path,
            model=args.model,
            temperature=0.0,
            max_iterations=args.max_iterations,
            des_runs=args.des_runs,
            abs_runs=args.abs_runs
        )
        results.append(result)

    # Generate evaluation package
    generate_evaluation_package(results, output_dir)


if __name__ == '__main__':
    main()
