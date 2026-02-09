import os
from dotenv import load_dotenv
load_dotenv()
from predictive_agent import PredictiveProcessAgent


def test_predictive_process_agent():
    print("\n" + "=" * 80)
    print("Testing predictive agent on event log...")
    print("=" * 80 + "\n")

    agent = PredictiveProcessAgent(model_name="gpt-4.1")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "..", "logs", "road_traffic.xes")

    print(f"\nRunning predictive analysis on: {log_path}\n")
    report = agent.run_analysis_pipeline(log_path)

    print("\n" + "=" * 80)
    print("PREDICTIVE INSIGHT REPORT")
    print("=" * 80)
    print(report)
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    test_predictive_process_agent()



