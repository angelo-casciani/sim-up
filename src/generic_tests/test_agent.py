import re
from typing import Dict, List, Any

from dotenv import load_dotenv
load_dotenv()
from gen_ai_hub.proxy.native.openai import chat


def call_llm(prompt: str, model_name: str = "gpt-4.1", temperature: float = 0.0, max_tokens: int = 2048) -> str:
    """Calls the LLM and returns the response content."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can use tools to answer questions."},
        {"role": "user", "content": prompt}
    ]
    kwargs = dict(model_name=model_name, messages=messages, temperature=temperature, max_tokens=max_tokens)
    response = chat.completions.create(**kwargs)
    return response.choices[0].message.content


# Tool Definitions
def calculator_tool(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def text_analyzer_tool(text: str) -> str:
    """Analyzes text and returns statistics."""
    words = text.split()
    sentences = text.split('.')
    return f"""Text Analysis:
- Word count: {len(words)}
- Character count: {len(text)}
- Sentence count: {len(sentences)}
- Average word length: {sum(len(word) for word in words) / len(words):.2f}"""


# Available tools for the agent
TOOLS = {
    "calculator": {
        "description": "Evaluates mathematical expressions",
        "input_schema": "string (e.g., '5 + 3')",
        "function": calculator_tool
    },
    "text_analyzer": {
        "description": "Analyzes text and provides statistics",
        "input_schema": "string (text to analyze)",
        "function": text_analyzer_tool
    }
}


def create_react_prompt(query: str, tools: Dict, history: List[str] = None) -> str:
    """
    Creates a ReAct-style prompt for the LLM.
    """
    tool_descriptions = "\n".join([
        f"- {name}: {info['description']} Input: {info['input_schema']}"
        for name, info in tools.items()
    ])
    
    history_text = "\n".join(history) if history else ""
    
    prompt = f"""You are a helpful assistant that can use tools to answer questions.

Available Tools:
{tool_descriptions}

Instructions:
1. Think step by step about how to answer the question
2. If you need to use a tool, respond with:
   Thought: [your reasoning]
   Action: [tool_name]
   Action Input: [just the input value as a plain string, no quotes]
3. After receiving an observation, either:
   - Use another tool if needed, OR
   - Provide the final answer with: Final Answer: [your answer]
4. IMPORTANT: After you get a tool result, you MUST either use another tool or give a Final Answer. Do not repeat the same action.

Example:
Thought: I need to calculate 5 + 3
Action: calculator
Action Input: 5 + 3
Observation: The result of 5 + 3 is 8
Thought: I have the answer from the calculation
Final Answer: The result is 8

Question: {query}

{history_text}

Let's think step by step:"""
    
    return prompt

def parse_llm_response(response: str) -> Dict[str, Any]:
    """
    Parses the LLM response to extract thought, action, and action input.
    """
    response = response.strip()
    
    # Check for final answer
    if "Final Answer:" in response:
        final_answer = response.split("Final Answer:")[-1].strip()
        return {"type": "final_answer", "content": final_answer}
    
    # Parse action
    thought_match = re.search(r"Thought:(.*?)(?=Action:|$)", response, re.DOTALL)
    action_match = re.search(r"Action:(.*?)(?=Action Input:|$)", response, re.DOTALL)
    action_input_match = re.search(r"Action Input:(.*?)(?=\n|$)", response, re.DOTALL)
    
    if action_match and action_input_match:
        action_input = action_input_match.group(1).strip()
        
        # Remove quotes if present
        if action_input.startswith('"') and action_input.endswith('"'):
            action_input = action_input[1:-1]
        elif action_input.startswith("'") and action_input.endswith("'"):
            action_input = action_input[1:-1]
        
        # Try to parse if it's a dictionary format
        if action_input.startswith('{'):
            try:
                import ast
                parsed_dict = ast.literal_eval(action_input)
                action_input = list(parsed_dict.values())[0]
            except:
                pass
        
        return {
            "type": "action",
            "thought": thought_match.group(1).strip() if thought_match else "",
            "action": action_match.group(1).strip(),
            "action_input": action_input
        }
    
    return {"type": "unknown", "content": response}

def react_agent(query: str, max_iterations: int = 5, verbose: bool = True) -> str:
    """
    ReAct-style agent loop.
    """
    history = []
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}")
            print(f"{'='*60}")
        
        # Generate prompt and get LLM response
        prompt = create_react_prompt(query, TOOLS, history)
        response = call_llm(prompt)
        
        if verbose:
            print(f"\nLLM Response:\n{response}")
        
        parsed = parse_llm_response(response)
        
        if parsed["type"] == "final_answer":
            if verbose:
                print(f"\nFinal Answer: {parsed['content']}")
            return parsed["content"]
        
        elif parsed["type"] == "action":
            # Execute tool
            tool_name = parsed["action"].lower().strip()
            tool_input = parsed["action_input"]
            
            if tool_name in TOOLS:
                observation = TOOLS[tool_name]["function"](tool_input)
                if verbose:
                    print(f"\nTool: {tool_name}")
                    print(f"Input: {tool_input}")
                    print(f"Observation: {observation}")
                
                # Add to history
                history.append(f"Thought: {parsed['thought']}")
                history.append(f"Action: {parsed['action']}")
                history.append(f"Action Input: {parsed['action_input']}")
                history.append(f"Observation: {observation}")
                    
            else:
                error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {list(TOOLS.keys())}"
                if verbose:
                    print(f"\n{error_msg}")
                history.append(f"Observation: {error_msg}")
        else:
            if verbose:
                print(f"\nCould not parse response, continuing...")
            history.append(f"Thought: {parsed.get('content', response)}")
    
    return "Max iterations reached without finding an answer."


def simple_agent_test(query: str, expected_tool: str) -> str:
    """
    A simplified agent test that demonstrates tool selection and execution.
    """
    print(f"\nQuery: {query}")
    
    # Simple tool selection logic
    if "analyze" in query.lower() and "text" in query.lower():
        tool_name = "text_analyzer"
        # Extract text between quotes
        text_match = re.search(r"['\"]([^'\"]+)['\"]|text[:\s]+(.+?)(?:\.|$)", query, re.IGNORECASE)
        if text_match:
            text = text_match.group(1) or text_match.group(2)
            text = text.strip()
            result = text_analyzer_tool(text)
            print(f"Selected Tool: {tool_name}")
            print(f"Input: {text[:50]}...")
            print(f"Result:\n{result}")
            return f"Using the text analyzer tool:\n{result}"
    
    elif "calculate" in query.lower() or any(op in query for op in ['+', '-', '*', '/']):
        tool_name = "calculator"
        # Extract mathematical expression
        # Look for mathematical expressions
        math_pattern = r'[0-9+\-*/\(\)\s]+'
        matches = re.findall(math_pattern, query)
        if matches:
            expression = max(matches, key=len).strip()
            result = calculator_tool(expression)
            print(f"Selected Tool: {tool_name}")
            print(f"Input: {expression}")
            print(f"Result: {result}")
            return f"Using the calculator tool: {result}"
    
    return "Could not determine appropriate tool for this query."


def main():
    """Main function to run all tests."""
    print("=" * 80)
    print("ReAct Agent Testing Suite")
    print("=" * 80)
    
    # Test 1: Calculator with ReAct Agent
    print("\n" + "#"*70)
    print("# TEST 1: Calculator Tool (ReAct Agent)")
    print("#"*70)
    
    query1 = "Calculate 234 * 17 + 89"
    print(f"\nQuery: {query1}\n")
    
    answer1 = react_agent(query1, verbose=True)
    
    print("\n" + "="*70)
    print(f"FINAL RESULT: {answer1}")
    print("="*70)
    
    # Test 2: Text Analyzer with ReAct Agent
    print("\n" + "#"*70)
    print("# TEST 2: Text Analyzer Tool (ReAct Agent)")
    print("#"*70)
    
    query2 = """Analyze this text: 'Artificial intelligence is transforming the way we work and live. 
It enables machines to learn from experience and perform human-like tasks. 
The future of AI holds immense potential for innovation.'"""
    
    print(f"\nQuery: {query2}\n")
    
    answer2 = react_agent(query2, verbose=True)
    
    print("\n" + "="*70)
    print(f"FINAL RESULT: {answer2}")
    print("="*70)
    
    # Simple Agent Tests
    print("\n" + "#"*70)
    print("# SIMPLE AGENT DEMONSTRATIONS")
    print("#"*70)
    
    # Test 3: Calculator (Simple)
    print("\n" + "="*50)
    print("TEST 3: Calculator Tool (Simple)")
    print("="*50)
    answer3 = simple_agent_test("Calculate 234 * 17 + 89", "calculator")
    print(f"\nFinal Answer: {answer3}")
    
    # Test 4: Text Analyzer (Simple)
    print("\n" + "="*50)
    print("TEST 4: Text Analyzer Tool (Simple)")
    print("="*50)
    answer4 = simple_agent_test("Analyze this text: 'Artificial intelligence is transforming the way we work and live. It enables machines to learn from experience and perform human-like tasks. The future of AI holds immense potential for innovation.'", "text_analyzer")
    print(f"\nFinal Answer: {answer4}")
    
    print("\n" + "#"*70)
    print("# ALL TESTS COMPLETE")
    print("#"*70)


if __name__ == "__main__":
    main()
