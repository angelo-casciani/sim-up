from argparse import ArgumentParser
import os
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()
from predictive_agent import PredictiveProcessAgent


def parse_arguments():
    parser = ArgumentParser(description="Run LLM Generation.")
    parser.add_argument('--model', type=str, default='gpt-4.1', help='LLM model id')
    parser.add_argument('--log', type=str, default='road_traffic.xes', help='Event log to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for LLM')
    parser.add_argument('--max_tokens', type=int, default=32768, help='Max tokens for LLM')
    args = parser.parse_args()
    return args


def test_predictive_process_agent():
    print("\n" + "=" * 80)
    print("Testing predictive agent on event log...")
    print("=" * 80 + "\n")

    args = parse_arguments()
    model_name = args.model
    log_name = args.log
    temperature = args.temperature
    max_tokens = args.max_tokens
    
    agent = PredictiveProcessAgent(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "..", "logs", log_name)

    print(f"\nRunning predictive analysis on: {log_path}\n")
    report = agent.run_analysis_pipeline(log_path)

    print("\n" + "=" * 80)
    print("PREDICTIVE INSIGHT REPORT")
    print("=" * 80)
    print(report)
    print("\n" + "=" * 80 + "\n")

    report_dir = os.path.join(script_dir, "..", "report")
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"report_{model_name}_{log_name}_{temperature}_{max_tokens}_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        f.write("PREDICTIVE INSIGHT REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log File: {log_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Max Tokens: {max_tokens}\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    
    print(f"Report saved to: {report_file}")




if __name__ == "__main__":
    test_predictive_process_agent()



