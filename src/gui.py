import os
import sys
import io
import threading
from datetime import datetime
from contextlib import redirect_stdout

import gradio as gr
from improvement_validation_pipeline import ImprovementValidationAgent
from llm import DEFAULT_MODEL, SUPPORTED_MODELS, OllamaUnavailableError, check_ollama_model, model_slug

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "..", "logs")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")

# Supported LLMs (Ollama model tags)
SUPPORTED_LLMS = SUPPORTED_MODELS


def get_available_logs():
    """List log files in /logs folder, excluding o2c subfolder."""
    logs = []
    for f in os.listdir(LOGS_DIR):
        path = os.path.join(LOGS_DIR, f)
        if os.path.isfile(path) and f.lower().endswith((".csv", ".xes")):
            logs.append(f)
    return sorted(logs)


def get_latest_report(log_name: str, model: str) -> str:
    """Find and read the most recent report matching log and model."""
    log_stem = os.path.splitext(log_name)[0]
    matching = []
    for f in os.listdir(REPORTS_DIR):
        if f.startswith("improvement_report_") and log_stem in f and model_slug(model) in f:
            matching.append(f)
    if not matching:
        return ""
    matching.sort(reverse=True)  # newest first (timestamp in filename)
    report_path = os.path.join(REPORTS_DIR, matching[0])
    with open(report_path, "r", encoding="utf-8") as fp:
        return fp.read()


def run_pipeline(log_file, model, temperature, max_iterations, think):
    """Run the improvement pipeline and yield status updates."""
    log_path = os.path.join(LOGS_DIR, log_file)

    if not os.path.exists(log_path):
        yield f"Error: Log file not found: {log_path}", ""
        return

    try:
        check_ollama_model(model)
    except OllamaUnavailableError as e:
        yield f"Error: {e}", ""
        return

    yield "Initializing agent...", ""

    # Capture stdout for progress
    output_buffer = io.StringIO()
    report = ""
    error_msg = ""

    def run_agent():
        nonlocal report, error_msg
        try:
            agent = ImprovementValidationAgent(
                model_name=model,
                temperature=temperature,
                max_tokens=4096,
                think=bool(think),
                max_iterations=int(max_iterations),
                use_sudo=False,
                des_runs=10,
                abs_runs=10,
                des_constrain_total_cases=False,
            )
            report = agent.run_improvement_pipeline(log_path)
        except Exception as e:
            error_msg = str(e)

    # Run in thread with stdout capture
    old_stdout = sys.stdout
    sys.stdout = output_buffer

    thread = threading.Thread(target=run_agent)
    thread.start()

    # Poll for updates
    last_output = ""
    while thread.is_alive():
        current_output = output_buffer.getvalue()
        if current_output != last_output:
            last_output = current_output
            yield f"Running pipeline...\n\n{current_output[-2000:]}", ""
        thread.join(timeout=1.0)

    sys.stdout = old_stdout
    final_output = output_buffer.getvalue()

    if error_msg:
        yield f"Error: {error_msg}\n\nOutput:\n{final_output[-2000:]}", ""
        return

    # Save report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_stem = os.path.splitext(log_file)[0].replace(os.sep, "_").replace("/", "_")
    report_filename = f"improvement_report_{model_slug(model)}_{log_stem}_{temperature}_{4096}_{timestamp}.txt"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write(report)

    yield f"Done! Report saved to:\n{report_filename}\n\nOutput:\n{final_output[-1000:]}", report


# Build Gradio UI
with gr.Blocks(title="Process Improvement Agent") as demo:
    gr.Markdown("# Process Improvement Agent")

    with gr.Row():
        with gr.Column(scale=1):
            log_dropdown = gr.Dropdown(
                choices=get_available_logs(),
                label="Select Log File",
                value=get_available_logs()[0] if get_available_logs() else None,
            )
            model_dropdown = gr.Dropdown(
                choices=SUPPORTED_LLMS,
                label="Select LLM Model (Ollama)",
                value=DEFAULT_MODEL,
                allow_custom_value=True,
            )
            think_checkbox = gr.Checkbox(
                value=False,
                label="Enable thinking (models that support it)",
            )
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.0,
                label="Temperature",
            )
            max_iter_input = gr.Number(
                value=3,
                minimum=1,
                maximum=10,
                step=1,
                label="Max Iterations",
            )
            run_btn = gr.Button("Run Pipeline", variant="primary")
            status_box = gr.Textbox(
                label="Status",
                lines=10,
                interactive=False,
            )

        with gr.Column(scale=2):
            report_box = gr.Textbox(
                label="Improvement Report",
                lines=30,
                interactive=False,
            )

    run_btn.click(
        fn=run_pipeline,
        inputs=[log_dropdown, model_dropdown, temperature_slider, max_iter_input, think_checkbox],
        outputs=[status_box, report_box],
    )

if __name__ == "__main__":
    demo.launch()
