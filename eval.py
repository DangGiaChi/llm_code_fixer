import argparse
import subprocess
import sys
import tempfile
import os
import re
from datasets import load_dataset
from tqdm import tqdm
from langchain_core.messages import HumanMessage
from agent.graph import agent_graph

# Same timeout as the agent's tool
CODE_EXECUTION_TIMEOUT = 5

def check_solution(problem, generated_code: str) -> bool:
    test_code = problem['test']
    full_test_script = f"{generated_code}\n\n{test_code}"
    
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmpf:
        tmpf.write(full_test_script)
        tmpf_path = tmpf.name
        
    try:
        result = subprocess.run(
            [sys.executable, tmpf_path],
            capture_output=True,
            text=True,
            timeout=CODE_EXECUTION_TIMEOUT,
            check=False
        )
        
        if result.returncode == 0 and not result.stderr:
            return True
        else:
            return False
            
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        if os.path.exists(tmpf_path):
            os.remove(tmpf_path)

def extract_python_code(raw_output: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", raw_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    if raw_output.strip().startswith("def"):
         return raw_output.strip()
    
    # Forcing a failure
    return ""

def run_evaluation(subsample: int):
    print(f"Loading dataset 'bigcode/humanevalpack'...")
    dataset = load_dataset("bigcode/humanevalpack", split="test")
    
    if subsample:
        print(f"Using a subsample of {subsample} problems.")
        dataset = dataset.select(range(subsample))
    else:
        print(f"Using the full dataset ({len(dataset)} problems).")
        
    num_passed = 0
    total_problems = len(dataset)

    print("Starting evaluation...")
    for i, problem in enumerate(tqdm(dataset, desc="Evaluating problems")):
        prompt = problem['prompt']
        buggy_code = problem['buggy_solution']

        initial_input = f"""
Here is the function signature and docstring:
```python
{prompt}
```
Here is the buggy implementation:
```python
{buggy_code}
```
Please fix the implementation
"""
        # k=1 attempt for pass@1
        try:
            final_state = agent_graph.invoke({
                "messages": [HumanMessage(content=initial_input)]
            })
            raw_solution = final_state['messages'][-1].content
            generated_code = extract_python_code(raw_solution)
            
            if not generated_code:
                tqdm.write(f"Problem {i+1}/{total_problems}: FAILED (No code output)")
                continue

            is_pass = check_solution(problem, generated_code)
            
            if is_pass:
                num_passed += 1
                tqdm.write(f"Problem {i+1}/{total_problems}: PASSED")
            else:
                tqdm.write(f"Problem {i+1}/{total_problems}: FAILED")
                
        except Exception as e:
            tqdm.write(f"Problem {i+1}/{total_problems}: ERROR ({e})")

    # Report
    if total_problems == 0:
        print("No problems were evaluated.")
        return

    pass_at_1 = (num_passed / total_problems)
    print("\n--- Evaluation Complete ---")
    print(f"Total Problems: {total_problems}")
    print(f"Problems Passed: {num_passed}")
    print(f"pass@1 Score:   {pass_at_1:.4f} ({pass_at_1 * 100:.2f}%)")

def main(): 
    parser = argparse.ArgumentParser(description="Run HumanEvalFix evaluation for the AI code-fixing agent.") 
    parser.add_argument( "--subsample", type=int, default=20, help="Number of problems to run from the dataset. Set to 0 to run all." ) 
    args = parser.parse_args()
    run_evaluation(subsample=args.subsample)

if __name__ == "__main__":
    main()