# LLM Code-Fixing Agent

This project implements an LLM-based AI agent using `langgraph` to autonomously fix buggy Python code. The agent uses a ReAct-style loop and a sandboxed Python interpreter to test and validate its fixes.

The agent's performance is evaluated on a subsample of the `bigcode/humaneval-fix-python` benchmark using the `pass@1` metric.

## First steps:

### Setup Environment

First, you need to set up the project environment and install the required tools

**Install requirements.txt:**

    pip install -r requirements.txt

This project requires a model that supports tool calling. We will use `llama3.1`, which is an 8B model optimized for this task.

**Install Ollama:**
    Follow the instructions on the [official Ollama website](https://ollama.com/download) to download and install the application.

**Start the Ollama Server:**
    On macOS and Windows, running the application is usually enough. On Linux, or to be sure, you can run:

    ollama serve


**Pull the Model:**
    In a new terminal window, pull the `mistral:7b-instruct` model.

    ollama pull mistral:7b-instruct

### Configure the Agent

* Make sure the `model` parameter is set to **`"mistral:7b-instruct"`**:

    ```python
    # Inside agent/graph.py
    llm = ChatOllama(model="mistral:7b-instruct")
    ```

---

## Running the Evaluation

The main script `eval.py` runs the agent against the `HumanEvalFix` benchmark and calculates the `pass@1` score.

### Run on a Subsample (Recommended First)

By default, the script runs on a small **subsample of 20 problems**. This should take a while depending on your computer's computational capabilities.

    python -m evaluation.run_eval


### Run the Full Benchmark

To run the full evaluation on all 164 problems, use the `--subsample=0` flag.

    python -m evaluation.run_eval --subsample=0


### Expected Output

You will see a progress bar and status updates for each problem:

```
Loading dataset 'bigcode/humaneval-fix-python'...
Using a subsample of 20 problems.
Starting evaluation...
Problem 1/20: FAILED                                                                              
Problem 2/20: FAILED                                                                              
Problem 3/20: PASSED                                                                              
Problem 4/20: PASSED                                                                              
Problem 5/20: FAILED                                                                              
Problem 6/20: FAILED                                                                              
Problem 7/20: FAILED                                                                              
Problem 8/20: FAILED                                                                              
Problem 9/20: FAILED                                                                              
Problem 10/20: FAILED                                                                             
Problem 11/20: FAILED                                                                             
Problem 12/20: PASSED                                                                             
Problem 13/20: FAILED                                                                             
Problem 14/20: PASSED                                                                             
Problem 15/20: FAILED                                                                             
Problem 16/20: PASSED                                                                             
Problem 17/20: PASSED                                                                             
Problem 18/20: FAILED                                                                             
Problem 19/20: PASSED                                                                             
Problem 20/20: FAILED                                                                             
Evaluating problems: 100%|████████████████████████████████████████| 20/20 [06:58<00:00, 20.90s/it]

--- Evaluation Complete ---
Total Problems: 20
Problems Passed: 7
pass@1 Score:   0.3500 (35.00%)
```

---