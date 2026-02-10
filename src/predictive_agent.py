from typing import Dict, Any
from gen_ai_hub.proxy.native.openai import chat

from analysis import (
    import_xes,
    get_start_activities,
    get_end_activities,
    get_variants,
    get_case_arrival_average,
    get_cycle_time,
    get_stochastic_language,
    abstract_log_features,
    abstract_variants,
    discover_petri_net_inductive,
    abstract_petri_net,
    discover_declare,
    abstract_declare,
)


class PredictiveProcessAgent:
    def __init__(self, model_name: str = "gpt-4.1", temperature: float = 0.0, max_tokens: int = 32768):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run_analysis_pipeline(self, log_path: str) -> str:
        """
        Executes PM4Py functions to gather stats, then calls the LLM to generate a report about predictive insights.

        Args:
            log_path: Path to the XES event log file

        Returns:
            A string containing the produced report.
        """
        print(f"1. Loading event log: {log_path}")
        log = import_xes(log_path)
        
        print("2. Extracting stats")

        # general stats
        start_acts = get_start_activities(log)
        end_acts = get_end_activities(log)
        variants = get_variants(log)

        # temporal/predictive metrics
        arrival_rate = get_case_arrival_average(log)
        cycle_times = get_cycle_time(log)

        # stochastic model for predicting the probability of future paths
        try:
            stochastic_lang = get_stochastic_language(log)
        except Exception as e:
            print(f"Warning: Could not compute stochastic language: {e}")
            stochastic_lang = "Not available"

        # abstractions for llms
        print("    Computing LLM abstractions")
        try:
            log_features = abstract_log_features(log)
        except Exception as e:
            print(f"Warning: Could not abstract log features: {e}")
            log_features = "Not available"

        try:
            variants_abstraction = abstract_variants(log)
        except Exception as e:
            print(f"Warning: Could not abstract variants: {e}")
            variants_abstraction = "Not available"

        # Discover Petri net for further analysis
        print("    Discovering process model")
        try:
            net, initial_marking, final_marking = discover_petri_net_inductive(log)
            petri_net_abstraction = abstract_petri_net(net, initial_marking, final_marking)
        except Exception as e:
            print(f"Warning: Could not discover Petri net: {e}")
            petri_net_abstraction = "Not available"

        try:
            declare_model = discover_declare(log)
            declare_abstraction = abstract_declare(declare_model)
        except Exception as e:
            print(f"Warning: Could not discover DECLARE model: {e}")
            declare_abstraction = "Not available"

        print("3. Context engineering for LLM")
        context_payload = self._build_context(
            log_features=log_features,
            variants_abstraction=variants_abstraction,
            arrival_rate=arrival_rate,
            cycle_times=cycle_times,
            stochastic_lang=stochastic_lang,
            start_acts=start_acts,
            end_acts=end_acts,
            petri_net_abstraction=petri_net_abstraction,
            declare_abstraction=declare_abstraction,
            variants_count=len(variants) if variants else 0,
        )

        print("4. Generating the predictive report")
        report = self._call_llm(context_payload)
        return report

    def _build_context(self,
                       log_features: str,
                       variants_abstraction: str,
                       arrival_rate: float,
                       cycle_times: Dict[str, Any],
                       stochastic_lang: Any,
                       start_acts: Dict[str, int],
                       end_acts: Dict[str, int],
                       petri_net_abstraction: str,
                       declare_abstraction: str,
                       variants_count: int) -> str:
        context = f"""
=== PROCESS LOG ANALYSIS ===

LOG FEATURES ABSTRACTION:
{str(log_features)[:1500]}

VARIANTS ABSTRACTION:
{str(variants_abstraction)[:1000]}
Total Unique Variants: {variants_count}

TEMPORAL METRICS:
- Average Case Arrival Rate: {arrival_rate} seconds between cases
- Cycle Time Statistics: {cycle_times}

STOCHASTIC PROFILE (Probabilistic Paths):
{str(stochastic_lang)[:2000]}

PROCESS MODEL ABSTRACTION (Petri Net):
{str(petri_net_abstraction)[:1000]}

COMPLIANCE RULES (DECLARE):
{str(declare_abstraction)[:1000]}

START ACTIVITIES: {start_acts}
END ACTIVITIES: {end_acts}
"""
        return context

    def _call_llm(self, context_data: str) -> str:
        system_prompt = self._get_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Analyze this process data and generate a predictive analysis report:\n\n{context_data}",
            },
        ]

        kwargs = dict(model_name=self.model_name, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        response = chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _get_system_prompt(self) -> str:
        return """
You are an expert Process Mining Agent specialized in Predictive Analytics.
Your goal is NOT to describe what happened in the past, but to forecast risks and opportunities for the future.

You have access to the 'Stochastic Language' of the process, which tells you the probability of specific paths occurring.

Structure your response into these 3 sections:

1. EXECUTIVE FORECAST
   - Based on the arrival rate and cycle times, predict the system load for the next period.
   - Identify the "Dominant Path" (the one with highest probability in the stochastic profile).
   - Estimate throughput expectations.

2. PREDICTIVE BOTTLENECK ANALYSIS
   - Look at the stochastic probabilities. If the most probable path contains activities known for delays (based on cycle time), predict the impact.
   - Format: "Given that X% of cases follow Path A, and Activity Y is in Path A with cycle time Z, we project a bottleneck at Y."
   - Consider the Petri net model to identify synchronization points that may cause delays.

3. ANOMALY RISK ASSESSMENT
   - Look at the low-probability traces in the Stochastic profile.
   - Predict the risk: "There is a Z% chance of cases deviating into [Rare Path], which correlates with [Long Duration/Loop]."
   - Consider the DECLARE compliance rules to flag potential violations.

DO NOT simply list statistics. Use terms like "Projected," "Likely," "Forecast," and "Risk of."
Provide actionable recommendations for process improvement based on your predictions.
"""
