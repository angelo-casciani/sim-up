import os
from argparse import ArgumentParser
from datetime import datetime
from dotenv import load_dotenv
import sys
import subprocess

load_dotenv()

from improvement_validation_pipeline import ImprovementValidationAgent


def parse_arguments():
    parser = ArgumentParser(description="Test Process Improvement Validation Agent")
    parser.add_argument('--model', type=str, default='gpt-4.1', help='LLM model id')
    parser.add_argument('--log', type=str, default='road_traffic.xes', help='Event log file name in logs/ directory')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for LLM generation (default: 0.0)')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Max tokens for LLM responses (default: 4096)')
    parser.add_argument('--max_iterations', type=int, default=3, help='Maximum improvement iterations (default: 3)')    
    parser.add_argument('--docker_image', type=str, default='nokal/simod:latest', help='SIMOD Docker image (default: nokal/simod:latest)')
    parser.add_argument('--no_sudo', action='store_true', help='Do not use sudo for Docker commands (default: use sudo)')
    
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
            print(f"✓ Docker found: {result.stdout.strip()}")
            return True
        else:
            print("✗ Docker not found or not working")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ Docker not found or not working")
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
            print(f"✓ SIMOD image '{image_name}' found")
            return True
        else:
            print(f"✗ SIMOD image '{image_name}' not found")
            print(f"  To pull it, run: docker pull {image_name}")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def test_improvement_agent():
    print("\n" + "=" * 80)
    print("PROCESS IMPROVEMENT VALIDATION AGENT - TEST")
    print("=" * 80 + "\n")
    
    args = parse_arguments()
    
    print("Configuration:")
    print(f"  LLM Model: {args.model}")
    print(f"  Event Log: {args.log}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max Tokens: {args.max_tokens}")
    print(f"  Max Iterations: {args.max_iterations}")
    print(f"  Docker Image: {args.docker_image}")
    print(f"  Use Sudo: {not args.no_sudo}")
    print()
    
    print("Checking prerequisites...")
    docker_ok = check_docker_available()
    if docker_ok:
        simod_ok = check_simod_image(args.docker_image)
        if not simod_ok:
            print("\nWarning: SIMOD image not found. Will use mock baseline data.")
            print("For full functionality, install SIMOD:")
            print(f"  docker pull {args.docker_image}")
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
        docker_image=args.docker_image,
        use_sudo=not args.no_sudo
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
        # Strip file extension from log name
        log_name_no_ext = os.path.splitext(args.log)[0]
        report_filename = (f"improvement_report_{args.model}_{log_name_no_ext}_{args.temperature}_{args.max_tokens}_{timestamp}.txt")
        report_path = os.path.join(report_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()


def simple_test():
    """
    Simple test with minimal configuration.
    Useful for quick verification without command-line arguments.
    """
    print("\n" + "=" * 80)
    print("SIMPLE IMPROVEMENT AGENT TEST")
    print("=" * 80 + "\n")
    
    # Use default settings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "..", "logs", "road_traffic.xes")
    
    if not os.path.exists(log_path):
        print(f"ERROR: Log file not found: {log_path}")
        return
    
    # Create agent with default settings
    agent = ImprovementValidationAgent(
        model_name="gpt-4.1",
        temperature=0.0,
        max_tokens=4096,
        max_iterations=2  # Fewer iterations for quick test
    )
    
    print(f"Running improvement pipeline on: {log_path}\n")
    
    try:
        report = agent.run_improvement_pipeline(log_path)
        
        print("\n" + "=" * 80)
        print("IMPROVEMENT REPORT")
        print("=" * 80)
        print(report)
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_improvement_agent()
    else:     # Show help
        print("\nUsage:")
        print("  python src/test_improvement_agent.py [options]")
        print("\nOptions:")
        print("  --model MODEL          LLM model (default: gpt-4.1)")
        print("  --log LOGFILE          Event log file (default: road_traffic.xes)")
        print("  --temperature TEMP     Temperature (default: 0.0)")
        print("  --max_tokens NUM       Max tokens (default: 4096)")
        print("  --max_iterations NUM   Max iterations (default: 3)")
        print("  --docker_image IMAGE   SIMOD Docker image (default: nokal/simod:latest)")
        print("\nExample:")
        print("  python src/test_improvement_agent.py --log sepsis.xes --max_iterations 5")
        print("\nOr run simple test (press Enter to continue, Ctrl+C to cancel):")
        
        try:
            input()
            simple_test()
        except KeyboardInterrupt:
            print("\nCancelled")
