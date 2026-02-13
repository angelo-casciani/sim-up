import os
import json
import tempfile
import shutil
import time
import subprocess
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from simulator import SIMODDiscoverer, ProsimosRunner
from llm import HypothesisGenerator
from validation import ImprovementValidator
from reporting import ReportGenerator


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
        self.hypothesis_gen = HypothesisGenerator(model_name, temperature, max_tokens)
        self.validator = ImprovementValidator()
        self.reporter = ReportGenerator(model_name, temperature, max_tokens)
    
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
            # Step 1: Discover baseline
            print("\n[STEP 1] Discovering baseline simulation model with SIMOD...")
            self._log_interaction("\n" + "="*80)
            self._log_interaction("[STEP 1] SIMOD BASELINE DISCOVERY")
            self._log_interaction("="*80)
            
            step1_start = time.time()
            baseline_model, baseline_metrics = self.discoverer.discover(log_path, self.work_dir)
            step1_time = time.time() - step1_start
            self.execution_times['step_1_baseline_discovery'] = step1_time
            
            if not baseline_model or not baseline_metrics:
                return "ERROR: Failed to discover baseline model"
            
            print(f"Baseline metrics: {json.dumps(baseline_metrics, indent=2)}")
            print(f"⏱️  Baseline discovery took {step1_time:.1f} seconds ({step1_time/60:.1f} minutes)\n")
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
                hypothesis = self.hypothesis_gen.generate(
                    baseline_metrics,
                    validated_improvements
                )
                hyp_time = time.time() - hyp_start
                
                print(f"Hypothesis: {hypothesis['description']}")
                print(f"Proposed changes: {json.dumps(hypothesis['changes'], indent=2)}")
                print(f"⏱️  Hypothesis generation took {hyp_time:.1f} seconds")
                
                self._log_interaction("\nLLM Generated Hypothesis:")
                self._log_interaction(json.dumps(hypothesis, indent=2))
                self._log_interaction(f"\n⏱️  Hypothesis generation time: {hyp_time:.1f} seconds")
                
                # Simulate with modifications
                print(f"\n[STEP 3.{iteration+1}] Running simulation with modifications...")
                self._log_interaction(f"\n[STEP 3.{iteration+1}] SIMULATION")
                self._log_interaction("-"*40)
                
                sim_start = time.time()
                new_metrics = self.simulator.simulate(
                    baseline_model,
                    hypothesis['changes'],
                    self.work_dir
                )
                sim_time = time.time() - sim_start
                
                if not new_metrics:
                    print("Warning: Simulation failed, skipping this hypothesis")
                    print(f"⏱️  Simulation took {sim_time:.1f} seconds (failed)")
                    self._log_interaction("\nSimulation FAILED - no metrics returned")
                    self._log_interaction(f"\n⏱️  Simulation time: {sim_time:.1f} seconds (failed)")
                    iteration_time = time.time() - iteration_start
                    iterations_times.append({'iteration': iteration + 1, 'total_time': iteration_time, 'status': 'failed'})
                    continue
                
                print(f"New metrics: {json.dumps(new_metrics, indent=2)}")
                print(f"⏱️  Simulation took {sim_time:.1f} seconds")
                self._log_interaction("\nSimulation Results (New Metrics):")
                self._log_interaction(json.dumps(new_metrics, indent=2))
                self._log_interaction(f"\n⏱️  Simulation time: {sim_time:.1f} seconds")
                
                # Validate improvement
                print(f"\n[STEP 4.{iteration+1}] Validating improvement...")
                self._log_interaction(f"\n[STEP 4.{iteration+1}] VALIDATION & COMPARISON")
                self._log_interaction("-"*40)
                
                val_start = time.time()
                improvement_result = self.validator.validate(
                    baseline_metrics,
                    new_metrics,
                    hypothesis
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
                self._log_interaction(f"\n⏱️  Validation time: {val_time:.1f} seconds")
                
                iteration_time = time.time() - iteration_start
                iterations_times.append({
                    'iteration': iteration + 1,
                    'total_time': iteration_time,
                    'hypothesis_time': hyp_time,
                    'simulation_time': sim_time,
                    'validation_time': val_time,
                    'status': 'validated' if improvement_result['is_improvement'] else 'not_improved'
                })
                
                print(f"⏱️  Iteration {iteration + 1} total time: {iteration_time:.1f} seconds ({iteration_time/60:.1f} minutes)")
                
                self.history.append({
                    "step": f"iteration_{iteration+1}",
                    "hypothesis": hypothesis,
                    "metrics": new_metrics,
                    "improvement": improvement_result,
                    "execution_time": iteration_time
                })
                
                if improvement_result['is_improvement']:
                    print(f"✓ VALIDATED IMPROVEMENT!")
                    print(f"  Impact: {improvement_result['summary']}")
                    validated_improvements.append({
                        "hypothesis": hypothesis,
                        "baseline_metrics": baseline_metrics,
                        "new_metrics": new_metrics,
                        "impact": improvement_result
                    })
                else:
                    print(f"✗ No improvement: {improvement_result['summary']}")
            
            # Step 6: Generate final report
            print(f"\n[STEP 6] Generating final report...")
            self._log_interaction(f"\n{'='*80}")
            self._log_interaction("[STEP 6] FINAL REPORT GENERATION")
            self._log_interaction(f"{'='*80}")
            self._log_interaction(f"\nTotal validated improvements: {len(validated_improvements)}")
            
            report_start = time.time()
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
            
            # Log execution times
            self._log_interaction("\n" + "="*80)
            self._log_interaction("EXECUTION TIME SUMMARY")
            self._log_interaction("="*80)
            self._log_interaction(f"\nTotal Pipeline Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            self._log_interaction(f"Baseline Discovery: {self.execution_times.get('step_1_baseline_discovery', 0):.1f} seconds")
            for it_time in iterations_times:
                self._log_interaction(f"\nIteration {it_time['iteration']} ({it_time['status']}): {it_time['total_time']:.1f} seconds")
                if 'hypothesis_time' in it_time:
                    self._log_interaction(f"  - Hypothesis Generation: {it_time['hypothesis_time']:.1f} seconds")
                    self._log_interaction(f"  - Simulation: {it_time['simulation_time']:.1f} seconds")
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
