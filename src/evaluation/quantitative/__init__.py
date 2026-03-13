# Quantitative evaluation: LLM vs Grid Search comparison
#
# This module implements the comprehensive evaluation framework for:
# 1. Resource Allocation - Adding resources to bottleneck roles
# 2. Calendar Adjustment - Extending working hours for constrained resources
# 3. Role Redesign - Redistributing workload between roles

# Global random seed for reproducibility and fair benchmarking.
# Using the same seed ensures that identical simulation inputs
# (e.g., Grid Search testing count=3 and LLM testing count=3)
# yield the exact same cycle time. This eliminates random variance
# as a confounding factor in the comparison.
GLOBAL_SEED = 42

# Resource Allocation components
from evaluation.quantitative.resource_fault_injector import FaultInjector
from evaluation.quantitative.resource_grid_search import GridSearch, GridSearchHistory
from evaluation.quantitative.resource_llm_search import LLMResourceSearch, LLMSearchHistory

# Calendar Adjustment components
from evaluation.quantitative.calendar_fault_injector import CalendarFaultInjector
from evaluation.quantitative.calendar_grid_search import CalendarGridSearch, CalendarGridSearchHistory
from evaluation.quantitative.calendar_llm_search import CalendarLLMSearch, CalendarLLMSearchHistory

# Role Redesign components
from evaluation.quantitative.role_redesign_fault_injector import RoleRedesignFaultInjector
from evaluation.quantitative.role_redesign_grid_search import (
    RoleRedesignGridSearch, RoleRedesignGridSearchHistory
)
from evaluation.quantitative.role_redesign_llm_search import (
    RoleRedesignLLMSearch, RoleRedesignLLMHistory
)

# Comprehensive evaluation
from evaluation.quantitative.quantitative_evaluation_runner import (
    ComprehensiveEvaluator,
    ImprovementType,
    InterventionCost,
    ExperimentConfig,
    AggregatedResult,
    EvaluationReport,
    run_comprehensive_evaluation
)

# Results reporting
from evaluation.quantitative.quantitative_evaluation_results_reporter import (
    ComprehensiveResultsReporter,
    generate_report_from_file
)

# Visualization
from evaluation.quantitative.results_visualizer import ResultsVisualizer

__all__ = [
    # Constants
    "GLOBAL_SEED",

    # Resource Allocation
    "FaultInjector",
    "GridSearch",
    "GridSearchHistory",
    "LLMResourceSearch",
    "LLMSearchHistory",

    # Calendar Adjustment
    "CalendarFaultInjector",
    "CalendarGridSearch",
    "CalendarGridSearchHistory",
    "CalendarLLMSearch",
    "CalendarLLMSearchHistory",

    # Role Redesign
    "RoleRedesignFaultInjector",
    "RoleRedesignGridSearch",
    "RoleRedesignGridSearchHistory",
    "RoleRedesignLLMSearch",
    "RoleRedesignLLMHistory",

    # Comprehensive Evaluation
    "ComprehensiveEvaluator",
    "ImprovementType",
    "InterventionCost",
    "ExperimentConfig",
    "AggregatedResult",
    "EvaluationReport",
    "run_comprehensive_evaluation",

    # Reporting
    "ComprehensiveResultsReporter",
    "generate_report_from_file",

    # Visualization
    "ResultsVisualizer",
]
