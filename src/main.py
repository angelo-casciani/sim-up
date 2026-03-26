import os
import platform
from argparse import ArgumentParser
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import subprocess

from improvement_validation_pipeline import ImprovementValidationAgent


def parse_arguments():
    parser = ArgumentParser(description="Test Process Improvement Validation Agent")
    is_windows = platform.system() == "Windows"

    parser.add_argument('--model', type=str, default='anthropic--claude-4.5-opus', help='LLM model id')
    parser.add_argument('--log', type=str, default='LoanApp.csv', help='Event log file name in logs/ directory')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for LLM generation (default: 0.0)')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Max tokens for LLM responses (default: 4096)')
    parser.add_argument('--max_iterations', type=int, default=3, help='Maximum improvement iterations (default: 3)')
    parser.add_argument('--no_sudo', action='store_true', default=is_windows,
                        help='Do not use sudo for Docker commands (auto-detected: disabled on Windows)')
    parser.add_argument('--des_runs', type=int, default=10, help='Number of DES (Prosimos) runs to average (default: 10)')
    parser.add_argument('--abs_runs', type=int, default=10, help='Number of ABS (AgentSimulator) runs to request (default: 10)')
    parser.add_argument('--des_constrain_total_cases', action='store_true', help='Constrain DES by baseline-derived max case budget (default: off)')

    return parser.parse_args()


def check_docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"[OK] Docker found: {result.stdout.strip()}")
            return True
        else:
            print("[X] Docker not found or not working")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[X] Docker not found or not working")
        return False


def check_simod_image(image_name: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "images", "-q", image_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            print(f"[OK] SIMOD image '{image_name}' found")
            return True
        else:
            print(f"[X] SIMOD image '{image_name}' not found")
            print(f"  To pull it, run: docker pull {image_name}")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    print("\n" + "=" * 80)
    print("PROCESS IMPROVEMENT VALIDATION AGENT")
    print("=" * 80 + "\n")

    args = parse_arguments()
    use_sudo = not args.no_sudo

    print("Configuration:")
    print(f"  LLM Model: {args.model}")
    print(f"  Event Log: {args.log}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max Tokens: {args.max_tokens}")
    print(f"  Max Iterations: {args.max_iterations}")
    print(f"  Use Sudo: {use_sudo}")
    print(f"  DES Runs: {args.des_runs}")
    print(f"  ABS Runs: {args.abs_runs}")
    print(f"  DES Case-Budget Constraint: {args.des_constrain_total_cases}")
    print()

    print("Checking prerequisites...")
    docker_ok = check_docker_available()
    if docker_ok:
        simod_ok = check_simod_image("nokal/simod:latest")
        if not simod_ok:
            print("\nWarning: SIMOD image not found. Will use mock baseline data.")
            print("For full functionality, install SIMOD:")
            print(f"  docker pull nokal/simod:latest")
    else:
        print("\nWarning: Docker not available. Will use mock baseline data.")
        print("For full functionality, install Docker.")

    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "..", "logs", args.log)

    if not os.path.exists(log_path):
        print(f"ERROR: Log file not found: {log_path}")
        return

    print(f"Event log path: {log_path}")
    print()

    print("Initializing predictive agent for improvement hypotheses...")
    agent = ImprovementValidationAgent(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        use_sudo=use_sudo,
        des_runs=args.des_runs,
        abs_runs=args.abs_runs,
        des_constrain_total_cases=args.des_constrain_total_cases
    )

    print("Running improvement pipeline...")
    try:
        report = agent.run_improvement_pipeline(log_path)

        print("\n" + "=" * 80)
        print("FINAL IMPROVEMENT REPORT")
        print("=" * 80)
        print(report)
        print("\n" + "=" * 80 + "\n")

        print("Saving report...")
        report_dir = os.path.join(script_dir, "..", "reports")
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name_no_ext = os.path.splitext(args.log)[0].replace(os.sep, "_").replace("/", "_")
        report_filename = (f"improvement_report_{args.model}_{log_name_no_ext}_{args.temperature}_{args.max_tokens}_{timestamp}.txt")
        report_path = os.path.join(report_dir, report_filename)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Report saved to: {report_path}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
