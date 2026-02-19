import os
import json
import tempfile
import shutil
import time
import subprocess
import threading
from datetime import datetime
from dotenv import load_dotenv
import sys
import re
import pandas as pd
from typing import Dict

# Ensure src directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from simulator import SIMODDiscoverer, ProsimosRunner, AgentSimulatorRunner, EnsembleArbiter
from llm import HypothesisGenerator
from validation import ImprovementValidator
from reporting import ReportGenerator, EnsembleReportSynthesizer


class ImprovementValidationAgent:
    """
    Agent that generates process improvement hypotheses and validates them through simulation.
    
    Modular architecture with separate components for:
    - SIMOD discovery (SIMODDiscoverer)
    - Prosimos simulation (ProsimosRunner)
    - LLM hypothesis generation (HypothesisGenerator)
    - Improvement validation (ImprovementValidator)
    - Report generation (ReportGenerator)
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4.1", 
        temperature: float = 0.0, 
        max_tokens: int = 4096,
        max_iterations: int = 3,
        docker_image: str = "nokal/simod:latest",
        use_sudo: bool = True
    ):
        """Initialize the improvement validation agent."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.docker_image = docker_image
        self.use_sudo = use_sudo
        self.work_dir = None
        self.history = []
        self.interaction_log = []
        self.log_file = None
        self.log_basename = None  # Store log filename without extension
        self.execution_times = {}  # Track execution times
        
        # Initialize modules
        self.discoverer = SIMODDiscoverer(docker_image, use_sudo)
        self.simulator = ProsimosRunner(docker_image, use_sudo)
        self.agent_simulator = AgentSimulatorRunner()  # Agent-based simulator (optional)
        self.arbiter = EnsembleArbiter()  # Conflict resolution between simulators
        self.hypothesis_gen = HypothesisGenerator(model_name, temperature, max_tokens)
        self.validator = ImprovementValidator()
        self.reporter = ReportGenerator(model_name, temperature, max_tokens)
        self.ensemble_reporter = EnsembleReportSynthesizer()  # Reality check reports
    
    def run_improvement_pipeline(self, log_path: str) -> str:
        """
        Main pipeline to discover, improve, and validate process improvements.
        
        Args:
            log_path: Path to the XES event log file
            
        Returns:
            A comprehensive report of validated improvements
        """
        pipeline_start_time = time.time()
        
        print(f"\n{'='*80}")
        print("PROCESS IMPROVEMENT VALIDATION PIPELINE")
        print(f"{'='*80}\n")
        
        # Setup working directory
        self.work_dir = tempfile.mkdtemp(prefix="simod_improvement_")
        print(f"Working directory: {self.work_dir}")
        
        # Store log basename without extension for later use
        self.log_basename = os.path.splitext(os.path.basename(log_path))[0]
        self.log_path = log_path
        self.abs_case_count = None
        self.time_horizon_seconds = self._compute_time_horizon_seconds(log_path)
        if self.time_horizon_seconds:
            days = self.time_horizon_seconds / 86400
            print(f"✓ Time horizon from baseline log: {self.time_horizon_seconds:.0f} seconds ({days:.1f} days)")
            self._log_interaction(f"Time horizon (seconds): {self.time_horizon_seconds:.0f} ({days:.1f} days)")
        
        # Setup interaction log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.work_dir, f"interaction_log_{timestamp}.txt")
        self._log_interaction("="*80)
        self._log_interaction("PROCESS IMPROVEMENT VALIDATION - INTERACTION LOG")
        self._log_interaction(f"Timestamp: {datetime.now()}")
        self._log_interaction(f"Event Log: {log_path}")
        self._log_interaction(f"Model: {self.model_name}")
        self._log_interaction("="*80 + "\n")
        
        try:
            # Step 1: Discover/Load baseline with optional calibration
            print("\n[STEP 1] Setting up baseline simulation model...")
            self._log_interaction("\n" + "="*80)
            self._log_interaction("[STEP 1] BASELINE MODEL SETUP (DISCOVERY + CALIBRATION)")
            self._log_interaction("="*80)
            
            step1_start = time.time()
            
            # Try to load dataset-specific calibrated model first, then generic fallback.
            log_dataset = os.path.basename(os.path.dirname(log_path))
            calibrated_candidates = [
                f"simod_model_{log_dataset.lower()}_calibrated.json",
                "simod_model_calibrated.json"
            ]
            calibrated_model_path = None
            for candidate in calibrated_candidates:
                if os.path.exists(candidate):
                    calibrated_model_path = candidate
                    break
            baseline_model = None
            baseline_metrics = None
            calibration_factor = None
            pm4py_baseline_metrics = None
            
            if calibrated_model_path:
                print(f"✓ Found calibrated SIMOD model: {calibrated_model_path}")
                with open(calibrated_model_path) as f:
                    baseline_model = json.load(f)
                
                # Extract PM4PY baseline metrics
                from calibration.simod_calibrator import SIMODCalibrator
                try:
                    calibrator = SIMODCalibrator(log_path)
                    pm4py_baseline_metrics = calibrator.extract_baseline_metrics()
                    
                    # Create synthetic baseline_metrics from PM4PY data
                    baseline_metrics = {
                        'avg_throughput': pm4py_baseline_metrics['num_traces'],
                        'avg_cycle_time': pm4py_baseline_metrics['case_mean_cycle_time'],
                        'avg_waiting_time': 0,  # Will compute from event data if needed
                        'activity_stats': pm4py_baseline_metrics['activity_stats'],
                        'resource_utilization': {},
                        'num_events': pm4py_baseline_metrics['num_events'],
                        'num_activities': pm4py_baseline_metrics['num_activities'],
                        'ground_truth_source': 'pm4py'
                    }
                    
                    print(f"✓ Loaded PM4PY baseline metrics:")
                    print(f"  - Cases: {baseline_metrics['avg_throughput']}")
                    print(f"  - Mean cycle time: {baseline_metrics['avg_cycle_time']:.0f}s ({baseline_metrics['avg_cycle_time']/3600:.2f} hours)")
                    print(f"  - Activities: {baseline_metrics['num_activities']}")
                    
                except Exception as e:
                    print(f"⚠️  Could not load PM4PY metrics: {e}")
                    # Fall back to SIMOD discovery
                    baseline_model = None
            
            # If no calibrated model, discover baseline
            if baseline_model is None:
                print("Discovering baseline simulation model with SIMOD...")
                baseline_model, baseline_metrics = self.discoverer.discover(log_path, self.work_dir)
            
            step1_time = time.time() - step1_start
            self.execution_times['step_1_baseline_discovery'] = step1_time
            
            if not baseline_model or not baseline_metrics:
                return "ERROR: Failed to discover baseline model"
            
            print(f"Baseline metrics: {json.dumps(baseline_metrics, indent=2)}")
            print(f"⏱️  Baseline discovery took {step1_time:.1f} seconds ({step1_time/60:.1f} minutes)\n")
            
            # VALIDATION: Check if resources are real (not fake)
            resource_util = baseline_metrics.get('resource_utilization', {})
            fake_resource_count = sum(1 for r in resource_util.keys() if str(r).startswith('Resource_'))
            real_resource_count = len(resource_util) - fake_resource_count
            
            if resource_util:
                print(f"✓ Resource Analysis: {real_resource_count} real resources, {fake_resource_count} synthetic")
                if real_resource_count > 0:
                    print(f"  Sample real resources: {list(str(r) for r in list(resource_util.keys())[:5])}")
                if fake_resource_count > 0:
                    print(f"  ⚠️  WARNING: Found {fake_resource_count} fake Resource_X entries - simulation may be using fallback")
            else:
                print(f"⚠️  WARNING: No resource utilization extracted - check event log format")
            
            self._log_interaction("\nBaseline Model Structure:")
            self._log_interaction(json.dumps({
                'has_prosimos_params': 'prosimos_params' in baseline_model,
                'model_keys': list(baseline_model.keys())
            }, indent=2))
            self._log_interaction("\nBaseline Metrics:")
            self._log_interaction(json.dumps(baseline_metrics, indent=2))
            self._log_interaction(f"\n⏱️  Baseline discovery time: {step1_time:.1f} seconds ({step1_time/60:.1f} minutes)")
            
            self.history.append({
                "step": "baseline",
                "metrics": baseline_metrics,
                "model": baseline_model,
                "execution_time": step1_time
            })

            # Set case count from baseline metrics - use as Prosimos upper bound
            # (actual throughput will be determined by arrival rate over time horizon)
            baseline_case_count = baseline_metrics.get('avg_throughput', 1000)
            try:
                self.abs_case_count = int(round(float(baseline_case_count)))
                # Increase Prosimos case limit to 2x baseline to ensure time horizon is reached
                # (metrics extraction will filter to time horizon anyway)
                prosimos_max_cases = self.abs_case_count * 2
                print(f"✓ Baseline: {self.abs_case_count} cases over {self.time_horizon_seconds/86400:.1f} days")
                print(f"✓ Prosimos will simulate up to {prosimos_max_cases} cases (2x baseline) to ensure time horizon coverage")
                print(f"✓ Both simulators will be filtered to {self.time_horizon_seconds:.0f}s time horizon during metrics extraction")
                self.abs_case_count = prosimos_max_cases
            except (TypeError, ValueError):
                self.abs_case_count = 2000
                print(f"⚠️  Could not parse case count, defaulting to {self.abs_case_count} cases")

            # Build simulator-relative baseline for validation (preferred for relative comparisons).
            baseline_des_metrics = None
            baseline_abs_metrics = None
            baseline_unified_metrics = None
            try:
                baseline_des_metrics, baseline_abs_metrics, baseline_unified_metrics = self._simulate_baseline(
                    baseline_model,
                    log_path,
                    baseline_metrics
                )
                if baseline_unified_metrics:
                    print("✓ Validation mode: relative (simulator baseline)")
                    print(f"  - Baseline cycle time: {baseline_unified_metrics.get('avg_cycle_time', 0):.0f}s")
                    print(f"  - Baseline throughput: {baseline_unified_metrics.get('avg_throughput', 0):.2f}")
                    self._log_interaction("\nValidation mode: relative (simulator baseline)")
                    self._log_interaction(json.dumps(baseline_unified_metrics, indent=2))
                else:
                    print("⚠️  Simulator baseline unavailable, using PM4PY baseline fallback")
                    self._log_interaction("\nValidation mode: PM4PY fallback (simulator baseline unavailable)")
            except Exception as baseline_sim_error:
                print(f"⚠️  Baseline simulation failed: {baseline_sim_error}")
                print("   Falling back to PM4PY baseline for validation")
                self._log_interaction(f"\nBaseline simulation failed: {baseline_sim_error}")
                self._log_interaction("Validation mode: PM4PY fallback")
                baseline_des_metrics = None
                baseline_abs_metrics = None
                baseline_unified_metrics = None

            working_model = baseline_model
            working_abs_log_path = log_path
            target_hints = self._extract_target_hints(
                baseline_model,
                baseline_metrics,
                baseline_abs_metrics
            )
            attempt_history = []
            
            # Step 2-5: Iterative improvement
            validated_improvements = []
            iterations_times = []
            
            for iteration in range(self.max_iterations):
                iteration_start = time.time()
                
                print(f"\n{'='*80}")
                print(f"ITERATION {iteration + 1}/{self.max_iterations}")
                print(f"{'='*80}\n")
                
                self._log_interaction(f"\n{'='*80}")
                self._log_interaction(f"ITERATION {iteration + 1}/{self.max_iterations}")
                self._log_interaction(f"{'='*80}")
                
                # Generate hypothesis
                print(f"[STEP 2.{iteration+1}] Generating improvement hypothesis...")
                self._log_interaction(f"\n[STEP 2.{iteration+1}] LLM HYPOTHESIS GENERATION")
                self._log_interaction("-"*40)
                
                hyp_start = time.time()
                hypothesis = None
                strategic_screening_notes = []
                screening_attempts = []
                for proposal_try in range(1, 4):
                    generation_history = attempt_history + screening_attempts
                    candidate = self.hypothesis_gen.generate(
                        baseline_metrics,
                        generation_history,
                        target_hints
                    )
                    is_strategic, strategic_reason = self._is_strategic_hypothesis(
                        candidate,
                        baseline_metrics,
                        generation_history,
                        target_hints
                    )
                    if is_strategic:
                        hypothesis = candidate
                        if proposal_try > 1:
                            strategic_screening_notes.append(
                                f"Accepted proposal on attempt {proposal_try}/3 after strategic screening."
                            )
                        break
                    strategic_screening_notes.append(
                        f"Rejected proposal {proposal_try}/3: {strategic_reason}"
                    )
                    screening_attempts.append({
                        "hypothesis": candidate,
                        "improvement": {
                            "is_improvement": False,
                            "summary": f"Strategic screening rejected: {strategic_reason}"
                        }
                    })

                if hypothesis is None:
                    hypothesis = self.hypothesis_gen.generate(
                        baseline_metrics,
                        attempt_history + screening_attempts,
                        target_hints
                    )
                    strategic_screening_notes.append(
                        "No proposal passed strategic screening; using final constrained fallback proposal."
                    )
                hyp_time = time.time() - hyp_start
                
                print(f"Hypothesis: {hypothesis['description']}")
                print(f"Proposed changes: {json.dumps(hypothesis['changes'], indent=2)}")
                print(f"⏱️  Hypothesis generation took {hyp_time:.1f} seconds")
                
                self._log_interaction("\nLLM Generated Hypothesis:")
                self._log_interaction(json.dumps(hypothesis, indent=2))
                if strategic_screening_notes:
                    self._log_interaction("\nStrategic screening notes:")
                    for note in strategic_screening_notes:
                        self._log_interaction(f"- {note}")
                self._log_interaction(f"\n⏱️  Hypothesis generation time: {hyp_time:.1f} seconds")
                
                # Simulate with modifications
                print(f"\n[STEP 3.{iteration+1}] Running simulation with modifications...")
                self._log_interaction(f"\n[STEP 3.{iteration+1}] PARALLEL SIMULATIONS (DES + ABS)")
                self._log_interaction("-"*40)
                
                sim_start = time.time()
                
                # Run Prosimos (DES) in main thread
                print(f"  ├─ Prosimos (Structural/DES)...")
                des_metrics = self.simulator.simulate(
                    working_model,
                    hypothesis['changes'],
                    self.work_dir,
                    total_cases=self.abs_case_count,
                    time_horizon_seconds=self.time_horizon_seconds,
                    baseline_metrics=baseline_metrics
                )
                
                # Run AgentSimulator (ABS) in parallel thread
                abs_metrics = None
                abs_thread_error = None
                def run_agent_simulator():
                    nonlocal abs_metrics, abs_thread_error
                    try:
                        print(f"  ├─ AgentSimulator (Behavioral/ABS)...")
                        abs_metrics = self.agent_simulator.simulate(
                            working_abs_log_path,
                            hypothesis['changes'],
                            self.work_dir,
                            enabled=True,
                            time_horizon_seconds=self.time_horizon_seconds,
                            num_simulations=1
                        )[0]
                    except Exception as e:
                        abs_thread_error = e

                agent_thread = threading.Thread(target=run_agent_simulator, daemon=False)
                agent_thread.start()
                agent_thread.join(timeout=300)
                
                if agent_thread and agent_thread.is_alive():
                    raise RuntimeError("AgentSimulator timed out (>5 min)")

                if abs_thread_error:
                    raise RuntimeError(f"AgentSimulator failed: {abs_thread_error}")
                
                sim_time = time.time() - sim_start
                
                if not des_metrics:
                    print("Warning: DES simulation failed, skipping hypothesis")
                    print(f"⏱️  Simulation took {sim_time:.1f} seconds (DES failed)")
                    self._log_interaction("\nDES Simulation FAILED - no metrics returned")
                    self._log_interaction(f"\n⏱️  Simulation time: {sim_time:.1f} seconds")
                    iteration_time = time.time() - iteration_start
                    iterations_times.append({'iteration': iteration + 1, 'total_time': iteration_time, 'status': 'failed'})
                    continue
                
                print(f"  └─ DES Metrics: {json.dumps(des_metrics, indent=2)[:100]}...")
                if abs_metrics:
                    print(f"  └─ ABS Metrics: {json.dumps(abs_metrics, indent=2)[:100]}...")
                else:
                    raise RuntimeError("ABS metrics unavailable; cannot proceed.")
                
                print(f"⏱️  Parallel simulations took {sim_time:.1f} seconds")
                
                self._log_interaction("\nDES (Prosimos) Results:")
                self._log_interaction(json.dumps(des_metrics, indent=2))
                self._log_interaction("\nABS (AgentSimulator) Results:")
                self._log_interaction(json.dumps(abs_metrics, indent=2))
                self._log_interaction(f"\n⏱️  Parallel simulation time: {sim_time:.1f} seconds")
                
                # STEP 3.5: Ensemble Arbitration (Stereoscopic Analysis)
                print(f"\n[STEP 3.5.{iteration+1}] Ensemble Arbitration (Stereoscopic Analysis)...")
                self._log_interaction(f"\n[STEP 3.5.{iteration+1}] ENSEMBLE ARBITRATION")
                self._log_interaction("-"*40)
                
                arb_start = time.time()
                arbitration_result = self.arbiter.arbitrate(
                    hypothesis,
                    des_metrics,
                    abs_metrics,
                    threshold=2.0
                )
                arb_time = time.time() - arb_start
                
                # Use unified metrics from arbitration
                new_metrics = arbitration_result['unified_metrics'].copy()
                
                print(f"Arbitration Strategy: {arbitration_result['arbitration_strategy'].upper()}")
                print(f"Primary Source: {arbitration_result['primary_source']}")
                print(f"Hypothesis Type: {arbitration_result['hypothesis_type']}")
                print(f"Confidence: {arbitration_result['confidence']*100:.1f}%")
                print(f"Behavioral Gap: {arbitration_result['behavioral_gap']}")
                if arbitration_result['warnings']:
                    for warning in arbitration_result['warnings']:
                        print(f"  ⚠️  {warning}")
                
                self._log_interaction("\nArbitration Result:")
                self._log_interaction(f"Strategy: {arbitration_result['arbitration_strategy']}")
                self._log_interaction(f"Primary Source: {arbitration_result['primary_source']}")
                self._log_interaction(f"Confidence: {arbitration_result['confidence']*100:.1f}%")
                self._log_interaction(f"Behavioral Gap: {arbitration_result['behavioral_gap']}")
                self._log_interaction(f"Insight: {arbitration_result['insight']}")
                if arbitration_result['warnings']:
                    self._log_interaction("\nWarnings:")
                    for warning in arbitration_result['warnings']:
                        self._log_interaction(f"  {warning}")
                self._log_interaction(f"\n⏱️  Arbitration time: {arb_time:.1f} seconds")
                
                # Validate improvement
                print(f"\n[STEP 4.{iteration+1}] Validating improvement...")
                self._log_interaction(f"\n[STEP 4.{iteration+1}] VALIDATION & COMPARISON")
                self._log_interaction("-"*40)
                
                val_start = time.time()
                improvement_result = self.validator.validate(
                    baseline_unified_metrics or baseline_metrics,
                    new_metrics,
                    hypothesis,
                    threshold=1.0,
                    min_improvements=1,
                    max_regression_pct=8.0,
                    max_regressions=1,
                    require_throughput_non_decrease=True,
                    des_metrics=des_metrics,
                    abs_metrics=abs_metrics,
                    check_agreement=True,
                    agreement_threshold=10.0,
                    arbitration_result=arbitration_result
                )
                val_time = time.time() - val_start
                
                print(f"⏱️  Validation took {val_time:.1f} seconds")
                
                self._log_interaction("\nMetric Comparison:")
                for metric_name, metric_data in improvement_result.get('improvements', {}).items():
                    self._log_interaction(f"\n{metric_name}:")
                    self._log_interaction(f"  Baseline: {metric_data.get('baseline', 'N/A')}")
                    self._log_interaction(f"  New:      {metric_data.get('new', 'N/A')}")
                    self._log_interaction(f"  Change:   {metric_data.get('change_pct', 0):.1f}%")
                
                self._log_interaction(f"\nValidation Result: {'✓ IMPROVEMENT' if improvement_result['is_improvement'] else '✗ NO IMPROVEMENT'}")
                self._log_interaction(f"Summary: {improvement_result['summary']}")
                if 'arbitration_info' in improvement_result:
                    self._log_interaction(f"Arbitration: {improvement_result['arbitration_info']}")
                
                # Log agreement check if available
                if 'agreement_check' in improvement_result:
                    agreement = improvement_result['agreement_check']
                    self._log_interaction(f"\nSimulator Agreement: {'✓ AGREE' if agreement['agree'] else '✗ CONTRADICT'}")
                    self._log_interaction(f"  {agreement['reason']}")
                    if agreement.get('details'):
                        self._log_interaction("  Details:")
                        for metric, info in agreement['details'].items():
                            if info['status'] == 'agree':
                                self._log_interaction(f"    {metric}: {info['direction']} (DES: {info['des_change']:.1f}%, ABS: {info['abs_change']:.1f}%)")
                            elif info['status'] == 'contradict':
                                self._log_interaction(f"    {metric}: CONTRADICTION (DES: {info['des_direction']} {info['des_change']:.1f}%, ABS: {info['abs_direction']} {info['abs_change']:.1f}%)")
                
                self._log_interaction(f"\n⏱️  Validation time: {val_time:.1f} seconds")
                
                iteration_time = time.time() - iteration_start
                iterations_times.append({
                    'iteration': iteration + 1,
                    'total_time': iteration_time,
                    'hypothesis_time': hyp_time,
                    'simulation_time': sim_time,
                    'arbitration_time': arb_time,
                    'validation_time': val_time,
                    'status': 'validated' if improvement_result['is_improvement'] else 'not_improved'
                })
                
                print(f"⏱️  Iteration {iteration + 1} total time: {iteration_time:.1f} seconds ({iteration_time/60:.1f} minutes)")
                
                self.history.append({
                    "step": f"iteration_{iteration+1}",
                    "hypothesis": hypothesis,
                    "metrics": new_metrics,
                    "des_metrics": des_metrics,
                    "abs_metrics": abs_metrics,
                    "arbitration": arbitration_result,
                    "improvement": improvement_result,
                    "execution_time": iteration_time
                })

                attempt_history.append({
                    "hypothesis": hypothesis,
                    "unified_metrics": new_metrics,
                    "des_metrics": des_metrics,
                    "abs_metrics": abs_metrics,
                    "arbitration": arbitration_result,
                    "improvement": improvement_result
                })
                if len(attempt_history) > 5:
                    attempt_history.pop(0)
                
                # === CUMULATIVE IMPROVEMENT: Update baseline for next iteration if validated ===
                if improvement_result['is_improvement']:
                    print(f"\n✅ IMPROVEMENT VALIDATED - Updating baseline for next iteration")
                    self._log_interaction(f"\n✅ IMPROVEMENT VALIDATED - Updating baseline for next iteration")
                    
                    # Apply modifications to working_model for next iteration
                    working_model = self.simulator.apply_modifications_to_model(
                        working_model,
                        hypothesis['changes']
                    )

                    # Apply modifications to ABS log for cumulative ABS scenario
                    working_abs_log_path = self.agent_simulator.apply_modifications_to_log(
                        working_abs_log_path,
                        hypothesis['changes']
                    )
                    
                    # Use new metrics as baseline for next iteration
                    baseline_metrics = new_metrics.copy()
                    
                    # Also update baseline_des/abs if they exist
                    if des_metrics:
                        baseline_des_metrics = des_metrics.copy()
                    if abs_metrics:
                        baseline_abs_metrics = abs_metrics.copy()
                    
                    # Update baseline_unified for next iteration
                    if 'unified_metrics' in arbitration_result:
                        baseline_unified_metrics = arbitration_result['unified_metrics'].copy()
                    
                    validated_improvements.append({
                        "iteration": iteration + 1,
                        "hypothesis": hypothesis,
                        "improvement": improvement_result,
                        "des_metrics": des_metrics,
                        "abs_metrics": abs_metrics,
                        "arbitration": arbitration_result,
                        "metrics_change": {
                            "cycle_time": improvement_result['improvements'].get('cycle_time', {}).get('change_pct', 0),
                            "throughput": improvement_result['improvements'].get('throughput', {}).get('change_pct', 0),
                            "waiting_time": improvement_result['improvements'].get('waiting_time', {}).get('change_pct', 0),
                        }
                    })
                    
                    print(f"  New baseline: cycle_time={baseline_metrics.get('avg_cycle_time', 0):.0f}s, "
                          f"throughput={baseline_metrics.get('avg_throughput', 0):.0f}, "
                          f"waiting_time={baseline_metrics.get('avg_waiting_time', 0):.0f}s")
                    self._log_interaction(f"  New baseline: cycle_time={baseline_metrics.get('avg_cycle_time', 0):.0f}s, "
                                         f"throughput={baseline_metrics.get('avg_throughput', 0):.0f}, "
                                         f"waiting_time={baseline_metrics.get('avg_waiting_time', 0):.0f}s")
                
                if improvement_result['is_improvement']:
                    print(f"✓ VALIDATED IMPROVEMENT!")
                    print(f"  Impact: {improvement_result['summary']}")
                else:
                    print(f"✗ No improvement: {improvement_result['summary']}")
            
            # Step 6: Generate final report
            print(f"\n[STEP 6] Generating final reality check report...")
            self._log_interaction(f"\n{'='*80}")
            self._log_interaction("[STEP 6] REALITY CHECK REPORT GENERATION")
            self._log_interaction(f"{'='*80}")
            self._log_interaction(f"\nTotal validated improvements: {len(validated_improvements)}")
            
            report_start = time.time()
            
            # Generate ensemble report for each validated improvement
            ensemble_reports = []
            for idx, improvement in enumerate(validated_improvements, 1):
                ensemble_report = self.ensemble_reporter.synthesize_report(
                    improvement['hypothesis'],
                    baseline_metrics,
                    improvement['des_metrics'],
                    improvement['abs_metrics'],
                    improvement['arbitration'],
                    self.execution_times
                )
                ensemble_reports.append(ensemble_report)
                self._log_interaction(f"\n{'='*80}")
                self._log_interaction(f"ENSEMBLE REPORT - IMPROVEMENT {idx}")
                self._log_interaction(f"{'='*80}")
                self._log_interaction(ensemble_report)
            
            # Also generate traditional report for structured comparison
            report = self.reporter.generate(
                baseline_metrics,
                validated_improvements,
                log_path,
                len(self.history),
                self.execution_times,
                iterations_times
            )
            
            report_time = time.time() - report_start
            self.execution_times['step_6_report_generation'] = report_time
            
            total_time = time.time() - pipeline_start_time
            self.execution_times['total_pipeline'] = total_time
            
            # Log execution times (including arbitration)
            self._log_interaction("\n" + "="*80)
            self._log_interaction("EXECUTION TIME SUMMARY")
            self._log_interaction("="*80)
            self._log_interaction(f"\nTotal Pipeline Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            self._log_interaction(f"Baseline Discovery: {self.execution_times.get('step_1_baseline_discovery', 0):.1f} seconds")
            for it_time in iterations_times:
                self._log_interaction(f"\nIteration {it_time['iteration']} ({it_time['status']}): {it_time['total_time']:.1f} seconds")
                if 'hypothesis_time' in it_time:
                    self._log_interaction(f"  - Hypothesis Generation: {it_time['hypothesis_time']:.1f} seconds")
                    self._log_interaction(f"  - Parallel Simulations (DES+ABS): {it_time['simulation_time']:.1f} seconds")
                    self._log_interaction(f"  - Ensemble Arbitration: {it_time.get('arbitration_time', 0):.1f} seconds")
                    self._log_interaction(f"  - Validation: {it_time['validation_time']:.1f} seconds")
            self._log_interaction(f"\nReport Generation: {report_time:.1f} seconds")
            
            print(f"\n⏱️  Total pipeline time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            
            return report
            
        finally:
            self._save_and_cleanup()
    
    def _save_and_cleanup(self):
        """Save interaction log and clean up working directory."""
        # Save interaction log
        if self.log_file and os.path.exists(self.log_file):
            # Create report directory
            report_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "reports", "interaction_logs"
            )
            os.makedirs(report_dir, exist_ok=True)
            
            # Use stored log basename without extension
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"interaction_log_{self.log_basename}_{timestamp}.txt"
            final_log_path = os.path.join(report_dir, log_name)
            
            shutil.copy2(self.log_file, final_log_path)
            print(f"\nInteraction log saved to: {final_log_path}")

        # Clean up per-log simulation folders created under the log directory
        try:
            if getattr(self, "log_path", None):
                log_dir = os.path.dirname(self.log_path)
                if log_dir and os.path.isdir(log_dir):
                    for entry in os.listdir(log_dir):
                        if entry.startswith("agent_simulated_"):
                            path = os.path.join(log_dir, entry)
                            if os.path.isdir(path):
                                shutil.rmtree(path, ignore_errors=True)
        except Exception as cleanup_err:
            print(f"Warning: Could not clean agent_simulated_* folders: {cleanup_err}")
        
        # Cleanup
        if self.work_dir and os.path.exists(self.work_dir):
            try:
                shutil.rmtree(self.work_dir)
                print(f"Cleaned up working directory: {self.work_dir}")
            except PermissionError:
                try:
                    if self.use_sudo:
                        subprocess.run(['sudo', 'rm', '-rf', self.work_dir], check=True)
                    else:
                        subprocess.run(['chmod', '-R', '755', self.work_dir], check=True)
                        shutil.rmtree(self.work_dir)
                    print(f"Cleaned up working directory: {self.work_dir}")
                except Exception as cleanup_err:
                    print(f"Warning: Could not fully clean up {self.work_dir}: {cleanup_err}")
    
    def _log_interaction(self, message: str):
        """Log interaction to file and memory."""
        self.interaction_log.append(message)
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(message + "\n")
            except Exception as e:
                print(f"Warning: Could not write to interaction log: {e}")

    def _extract_target_hints(self, baseline_model: Dict, baseline_metrics: Dict = None, baseline_abs_metrics: Dict = None) -> Dict:
        """Extract resource, activity, and calendar names for grounded hypotheses."""
        resources = set()
        activities = set()
        calendars = set()
        gateways = set()

        prosimos_params = baseline_model.get("prosimos_params", {}) if baseline_model else {}

        for profile in prosimos_params.get("resource_profiles", []) or []:
            res_id = profile.get("id") or profile.get("name") or profile.get("resource_id")
            if res_id:
                resources.add(str(res_id))

        for task in prosimos_params.get("task_resource_distribution", []) or []:
            task_id = task.get("task_id") or task.get("activity")
            if task_id:
                activities.add(str(task_id))

        calendars_data = prosimos_params.get("calendars")
        if isinstance(calendars_data, dict):
            calendars.update(calendars_data.keys())
        elif isinstance(calendars_data, list):
            for cal in calendars_data:
                cal_id = cal.get("id") or cal.get("name")
                if cal_id:
                    calendars.add(str(cal_id))

        model_calendars = baseline_model.get("calendars") if baseline_model else None
        if isinstance(model_calendars, dict):
            calendars.update(model_calendars.keys())

        gateway_probs = prosimos_params.get("gateway_branching_probabilities")
        if isinstance(gateway_probs, dict):
            gateways.update(str(gw_id) for gw_id in gateway_probs.keys())
        elif isinstance(gateway_probs, list):
            for gw in gateway_probs:
                if isinstance(gw, dict):
                    gw_id = gw.get("id") or gw.get("gateway_id")
                    if gw_id:
                        gateways.add(str(gw_id))

        return {
            "resources": sorted(resources),
            "activities": sorted(activities),
            "calendars": sorted(calendars),
            "gateways": sorted(gateways),
            "role_redesign_hints": self._build_role_redesign_hints(baseline_metrics, baseline_abs_metrics)
        }

    def _simulate_baseline(self, baseline_model: Dict, log_path: str, baseline_metrics: Dict = None):
        """Run baseline simulations to align validation with simulator scales."""
        baseline_changes = {"parameter_type": None, "specific_changes": []}

        print("\n[STEP 1.5] Simulating baseline (DES + ABS)...")
        baseline_abs_metrics, abs_output_dir = self.agent_simulator.simulate(
            log_path,
            None,
            self.work_dir,
            enabled=True,
            time_horizon_seconds=None
        )

        # Align DES case count to ABS baseline to reduce structural mismatch.
        if baseline_abs_metrics:
            abs_cases = baseline_abs_metrics.get("simulated_case_count")
            if abs_cases is None:
                abs_cases = baseline_abs_metrics.get("avg_throughput")
            if abs_cases:
                try:
                    self.abs_case_count = int(round(float(abs_cases)))
                    self._log_interaction(
                        f"Aligned DES total_cases to ABS baseline: {self.abs_case_count}"
                    )
                except (TypeError, ValueError):
                    self.abs_case_count = None

        abs_horizon = self._compute_time_horizon_from_abs_outputs(abs_output_dir)
        if abs_horizon:
            self.time_horizon_seconds = abs_horizon
            self._log_interaction(f"Time horizon (ABS baseline, seconds): {self.time_horizon_seconds:.0f}")

        baseline_des_metrics = self.simulator.simulate(
            baseline_model,
            baseline_changes,
            self.work_dir,
            total_cases=self.abs_case_count,
            time_horizon_seconds=self.time_horizon_seconds,
            baseline_metrics=baseline_metrics
        )

        if not baseline_des_metrics:
            return None, baseline_abs_metrics, None

        baseline_unified_metrics = self._combine_metrics(
            baseline_des_metrics,
            baseline_abs_metrics
        )

        return baseline_des_metrics, baseline_abs_metrics, baseline_unified_metrics

    def _build_role_redesign_hints(self, baseline_metrics: Dict = None, baseline_abs_metrics: Dict = None) -> str:
        """Build concise role redesign hints from baseline (and ABS when available)."""
        role_utils = {}

        baseline_resources = (baseline_metrics or {}).get("resource_utilization", {})
        abs_resources = (baseline_abs_metrics or {}).get("resource_utilization", {})

        for role, util in baseline_resources.items():
            role_utils[str(role)] = float(util)

        # Blend ABS only if it has non-flat signal
        abs_vals = [float(v) for v in abs_resources.values()] if abs_resources else []
        abs_has_signal = len(set(round(v, 3) for v in abs_vals)) > 2
        if abs_has_signal:
            for role, util in abs_resources.items():
                role = str(role)
                if role in role_utils:
                    role_utils[role] = (role_utils[role] + float(util)) / 2.0
                else:
                    role_utils[role] = float(util)

        if not role_utils:
            return "No role utilization hints available."

        sorted_roles = sorted(role_utils.items(), key=lambda x: x[1], reverse=True)
        over = [f"{r}({u:.2f})" for r, u in sorted_roles[:3]]
        under = [f"{r}({u:.2f})" for r, u in sorted_roles[-3:]]

        return (
            f"Overloaded candidates: {', '.join(over)}. "
            f"Underutilized candidates: {', '.join(under)}. "
            "Prefer role redesign: create specialist role or shift responsibilities from overloaded to underutilized roles before adding headcount."
        )

    @staticmethod
    def _compute_time_horizon_from_abs_outputs(output_dir: str) -> float:
        """Compute time horizon from ABS output CSVs."""
        if not output_dir or not os.path.isdir(output_dir):
            return None

        time_candidates = [
            "end_timestamp",
            "end_time",
            "time:timestamp",
            "timestamp",
            "start_timestamp",
            "start_time"
        ]

        min_time = None
        max_time = None
        try:
            csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
            for name in csv_files[:5]:
                path = os.path.join(output_dir, name)
                df = pd.read_csv(path)
                if df.empty:
                    continue

                time_col = None
                for candidate in time_candidates:
                    if candidate in df.columns:
                        time_col = candidate
                        break
                if time_col is None:
                    for col in df.columns:
                        if "time" in col.lower():
                            time_col = col
                            break
                if time_col is None:
                    continue

                series = pd.to_datetime(df[time_col], errors="coerce")
                series_min = series.min()
                series_max = series.max()
                if pd.isna(series_min) or pd.isna(series_max):
                    continue

                min_time = series_min if min_time is None else min(min_time, series_min)
                max_time = series_max if max_time is None else max(max_time, series_max)
        except Exception:
            return None

        if min_time is None or max_time is None:
            return None

        horizon = (max_time - min_time).total_seconds()
        return horizon if horizon and horizon > 0 else None

    @staticmethod
    def _compute_time_horizon_seconds(log_path: str) -> float:
        """Compute time horizon in seconds from a CSV log."""
        if not log_path or not log_path.endswith(".csv"):
            return None
        try:
            df = pd.read_csv(log_path)
        except Exception:
            return None
        if df.empty:
            return None

        time_candidates = [
            "end_timestamp",
            "end_time",
            "time:timestamp",
            "timestamp",
            "start_timestamp",
            "start_time"
        ]
        time_col = None
        columns = df.columns.tolist()
        for candidate in time_candidates:
            if candidate in columns:
                time_col = candidate
                break
        if time_col is None:
            for col in columns:
                if "time" in col.lower():
                    time_col = col
                    break
        if time_col is None:
            return None

        series = pd.to_datetime(df[time_col], errors="coerce")
        min_time = series.min()
        max_time = series.max()
        if pd.isna(min_time) or pd.isna(max_time):
            return None
        horizon = (max_time - min_time).total_seconds()
        return horizon if horizon and horizon > 0 else None

    def _is_strategic_hypothesis(
        self,
        hypothesis: Dict,
        baseline_metrics: Dict,
        attempt_history: list,
        target_hints: Dict
    ):
        """Cheap pre-simulation gate to reject low-likelihood or unsafe proposals."""
        changes = (hypothesis or {}).get("changes", {})
        parameter_type = changes.get("parameter_type")
        specific_changes = changes.get("specific_changes", [])
        expected = (hypothesis or {}).get("expected_impact", {})

        if parameter_type not in {"resources", "role_redesign", "activity_durations", "calendar"}:
            return False, f"unsupported parameter_type '{parameter_type}'"

        if not isinstance(specific_changes, list) or len(specific_changes) != 1:
            return False, "must contain exactly one specific change"

        throughput_pct = self._extract_pct(expected.get("throughput", "0"))
        if throughput_pct < 0:
            return False, "expected throughput is negative"

        recent_failed_types = []
        for attempt in attempt_history[-2:]:
            if attempt.get("improvement", {}).get("is_improvement") is False:
                failed_type = (
                    attempt.get("hypothesis", {})
                    .get("changes", {})
                    .get("parameter_type")
                )
                if failed_type:
                    recent_failed_types.append(failed_type)
        if len(recent_failed_types) == 2 and recent_failed_types[0] == recent_failed_types[1] == parameter_type:
            return False, f"parameter_type '{parameter_type}' failed in last two attempts"

        change = specific_changes[0]
        target = str(change.get("target", ""))
        change_name = str(change.get("change", ""))

        resources = set(target_hints.get("resources", []))
        activities = set(target_hints.get("activities", []))
        calendars = set(target_hints.get("calendars", []))

        if parameter_type == "resources":
            if change_name not in {"add", "increase_count"}:
                return False, "resource changes must be add/increase_count"
            if resources and target not in resources:
                return False, f"resource target '{target}' is not in known targets"

        if parameter_type == "activity_durations":
            if change_name != "reduce_duration":
                return False, "activity duration changes must reduce_duration"
            if activities and target not in activities:
                return False, f"activity target '{target}' is not in known targets"

        if parameter_type == "calendar":
            if change_name != "increase_hours":
                return False, "calendar changes must be increase_hours"
            if calendars and target not in calendars:
                return False, f"calendar target '{target}' is not in known targets"

        if parameter_type == "role_redesign":
            if change_name not in {"shift_responsibility", "create_specialist_role"}:
                return False, "role redesign must be shift_responsibility/create_specialist_role"
            util = (baseline_metrics or {}).get("resource_utilization", {}) or {}
            overloaded = {str(r) for r, u in util.items() if float(u) >= 0.8}
            underutilized = {str(r) for r, u in util.items() if float(u) <= 0.25}

            if "→" in target:
                source, destination = [s.strip().replace("_profile", "") for s in target.split("→", 1)]
                if overloaded and source not in overloaded:
                    return False, f"role redesign source '{source}' is not overloaded"
                if underutilized and destination not in underutilized:
                    return False, f"role redesign destination '{destination}' is not underutilized"

        return True, "accepted"

    @staticmethod
    def _extract_pct(value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
            if match:
                return float(match.group(0))
        return 0.0

    @staticmethod
    def _combine_metrics(des_metrics: Dict, abs_metrics: Dict) -> Dict:
        """Combine DES/ABS metrics with a simple consensus average."""
        if not des_metrics:
            return {}
        if not abs_metrics:
            return des_metrics.copy()
        
        combined = {}
        for key in ["avg_cycle_time", "avg_throughput", "avg_waiting_time", "total_cost"]:
            if key in des_metrics and key in abs_metrics:
                combined[key] = (des_metrics[key] + abs_metrics[key]) / 2
            elif key in des_metrics:
                combined[key] = des_metrics[key]
            elif key in abs_metrics:
                combined[key] = abs_metrics[key]
        return combined
