import subprocess
import sys
import tempfile
import os
from langchain.tools import tool

CODE_EXECUTION_TIMEOUT = 5

@tool
def python_interpreter(code: str) -> str:
    """
    Placeholder
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmpf:
        tmpf.write(code)
        tmpf_path = tmpf.name
        
    try:
        completed_process = subprocess.run(
            [sys.executable, tmpf_path],
            capture_output=True,
            text=True,
            timeout=CODE_EXECUTION_TIMEOUT,
            check=False
        )
        
        stdout = completed_process.stdout
        stderr = completed_process.stderr
        
        if completed_process.returncode == 0:
            return f"Execution successful:\n[STDOUT]:\n{stdout}\n[STDERR]:\n{stderr}"
        else:
            return f"Execution failed with errors:\n[STDOUT]:\n{stdout}\n[STDERR]:\n{stderr}"

    except subprocess.TimeoutExpired:
        return f"Execution timed out after {CODE_EXECUTION_TIMEOUT} seconds."
    except Exception as e:
        return f"An unexpected error occurred during execution: {str(e)}"
    finally:
        if os.path.exists(tmpf_path):
            os.remove(tmpf_path)