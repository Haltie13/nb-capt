---
trigger: model_decision
description: When moving files in the repository
---

# AGENTS.md - Repo Restructuring Protocol

## 1. Objective
Clean up the legacy directory structure by separating source code from training checkpoints:
- **Move all Python source code** scattered in `/data` and `/data/src` into a unified root `/src/` directory.
- **Move all trainer checkpoints** (e.g., `.pt`, `.pth`, `.ckpt`, `.h5`, `checkpoint-*` directories) to a unified root `/checkpoints/` directory.

## 2. File Classification Matrix
Before moving a file, classify it using these rules:
* Files ending in `.py` -> Move to `src/` (maintain their relative sub-folder structures if applicable).
* Files/Directories matching `*checkpoint*`, `*.pt`, `*.pth`, `*.ckpt` -> Move to `checkpoints/`.
* Raw datasets (`*.csv`, `*.json`, `*.parquet`, audio files, Annotation files) -> Leave inside `data/`.

## 3. Step-by-Step Execution Loop
You must execute this migration sequentially, file-by-file or module-by-module:

### Step 3.1: The Move
* Use atomic filesystem moves (`mv`) so git tracks the history.
* Create `src/` and `checkpoints/` at the root if they do not exist.

### Step 3.2: Import Path Resolution
Because we are moving code to a root `src/` directory, relative imports will break. 
* Update relative imports (`from . import x` or `from .. legacy import y`) based on the file's new location in `src/`.
* If using absolute imports, ensure `src` is treated as the root package or adjust paths accordingly.

### Step 3.3: Verification Loop (No Native Tests Available)
Since there is no test suite, you must use Python's compiler and Ruff to verify code integrity after **every single move**:
1. Run compilation check on the moved file and its dependents: 
   `uv run python -m py_compile <path_to_file.py>`
2. Run Ruff to check for unresolved imports or syntax errors:
   `uv run ruff check src/`
3. If Ruff reports `F401` (unused import) or `E402` (module level import not at top), fix it. If it reports an unresolved import error, halt and trace where that file went.

## 4. Safety Constraints
* Do NOT delete any `.pt` or `.pth` files. Checkpoint data is destructive if lost.
* Do NOT modify the actual training logic inside the files—only modify `import` statements and file paths.