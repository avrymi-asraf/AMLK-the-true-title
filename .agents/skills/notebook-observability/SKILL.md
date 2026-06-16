---
name: notebook-observability
description: Use Jupyter notebooks as live observability playgrounds for production systems and AI pipelines. Use this skill when you need to trace internal transitions, test configs, and iterate prompts while keeping the notebook synchronized with real production code.
---

# Notebook Observability

<notebook-observability>
Use Jupyter notebooks (such as Colab or local `.ipynb` files) as a live visual observability layer for production systems instead of treating them as throwaway scratchpads. This pattern keeps the notebook aligned with the real code path so you can inspect internal state, test configurations, and visualize deep loops on real hardware without creating a separate wrapper codebase.
</notebook-observability>

<alignment-rule>
The notebook must call the production library API directly. Do not hide real signatures behind adapter functions just to make the notebook easier to use. If the production API changes, the notebook should break and be updated alongside the tests and CLI; that is part of keeping the observability layer honest.
</alignment-rule>

<function-reuse>
**Use the exact same functions from production code.** Do not rewrite or reinterpret business logic inside the notebook. Instead, import and call the real production functions directly. If you must add a wrapper (e.g., to add logging or catch exceptions for visualization), keep it minimal—delegate to the real function immediately.

**Why this matters:**
- **Saves time:** No duplicate logic to maintain or debug.
- **Forces clarity:** If a function is hard to call from the notebook, it means the function signature is unclear. Fix the function, not the notebook.
- **Ensures efficiency:** The notebook will use the same optimizations, caching, and shortcuts as the CLI or server.
- **Makes the process intuitive:** The sequence of calls in the notebook reflects how the real system works, helping you understand the actual execution flow.

**Bad:**
```python
# Don't rewrite the function
def format_output_in_notebook(data):
    result = []
    for item in data:
        result.append({"id": item["_id"], "text": item["content"]})
    return result
```

**Good:**
```python
# Import and use the real function
from my_module.formatting import format_output

output = format_output(data)
```

If the real function doesn't exist yet, create it. The notebook becomes a forcing function that drives the codebase toward clearer, more reusable APIs.
</function-reuse>

<self-bootstrapping>
A good observability notebook should bootstrap itself in a fresh runtime. Clone the target repo, install dependencies, insert the source path, and import the real pipeline entry points you want to inspect.

```python
# 1. Clone the active repo
!git clone https://github.com/your-org/your-repo.git
%cd your-repo

# 2. Install dependencies quietly (including GitHub-hosted packages)
%pip install -q "git+https://github.com/huggingface/transformers.git" "accelerate" "pyyaml"

# 3. Path injection so the kernel prioritizes the cloned source over global packages
import sys
sys.path.insert(0, "src")

# 4. Import the real, production-ready pipeline
from your_module.pipeline import load_config, run_loop
```
</self-bootstrapping>

<secrets-and-headless>
Make the notebook safe for automated or headless execution by using a cascading secret hook: environment variables first, then Colab userdata, then an interactive prompt as a fallback.

```python
import os


def _get_secret(key: str) -> str:
    """Env var first, then Colab vault, then interactive input."""
    if os.environ.get(key):
        return os.environ[key]

    try:
        from google.colab import userdata
        val = userdata.get(key)
        if val:
            return val
    except ImportError:
        pass

    return input(f"Enter {key}: ")


os.environ["API_KEY"] = _get_secret("API_KEY")
```
</secrets-and-headless>

<visualization>
Render internal transitions with focused formatting helpers instead of raw dumps. Use small display routines that make removed steps, pruning reasons, or execution phases obvious to a human operator.

```python
def show_transition(row: dict) -> None:
    print("=" * 60)
    print("\n[GENERATED STEPS]")
    for i, step in enumerate(row["steps"]):
        marker = "  ✗" if row["removed_start"] <= i <= row["removed_end"] else "   "
        print(f"{marker} {i}: {step}")

    print("\n[REASON FOR PRUNING]")
    print(f"  {row['reason']}")
    print("=" * 60)
```
</visualization>

<rapid-iteration>
Use the notebook to iterate on prompts, configs, and tracing logic without restarting the whole system. Keep heavy state in memory, reload text files or prompt templates directly in notebook cells, and run the real pipeline over the updated input immediately.
</rapid-iteration>

<when-to-use>
Use this skill when you need to:
- set up a notebook to trace or visualize a complex backend or AI pipeline;
- create a live playground for production code in a repository;
- bridge local code limits with Colab or other remote GPU runtimes while keeping the notebook synchronized with real production APIs.
</when-to-use>