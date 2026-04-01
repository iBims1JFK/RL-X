import argparse
import shutil
import sys
import os
import re
from pathlib import Path
import rl_x

# --- Default Gitignore Content ---
GITIGNORE_CONTENT = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Env
venv/
.env
.venv/

# RL-X Specific
wandb/
data/
checkpoints/
"""

def main():
    parser = argparse.ArgumentParser(description="RL-X Project Scaffolding")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: init
    init_parser = subparsers.add_parser("init", help="Initialize a new project")
    init_parser.add_argument("--algo", action="append", help="Algorithm to include (e.g. ppo)")
    init_parser.add_argument("--env", action="append", help="Environment type to include (e.g. gymnasium)")
    
    args = parser.parse_args()

    if args.command == "init":
        setup_project(args.algo, args.env)

def find_folder(base_path, options):
    """Helper to find a folder checking multiple naming options (singular/plural)"""
    for opt in options:
        candidate = base_path / opt
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None

def update_imports(root_dir, project_name):
    """
    Recursively updates imports in python files.
    Replaces 'rl_x.algorithms' -> 'project_name.algorithms' (except managers/types)
    Replaces 'rl_x.environments' -> 'project_name.environments' (except managers/types)
    """
    
    # Modules to keep as rl_x references (Framework Core)
    # These are checked against the start of the sub-module path
    KEEP_MODULES = {
        "algorithms.algorithm_manager",
        "algorithms.deep_learning_framework_type",
        "environments.environment_manager",
        "environments.action_space_type",
        "environments.observation_space_type",
        "environments.data_interface_type"
    }

    print(f"Refactoring imports in {root_dir.name}...")
    
    for path in root_dir.rglob("*.py"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_lines = []
        modified = False
        
        for line in content.splitlines(keepends=True):
            # Regex logic:
            # 1. Look for 'from rl_x.algorithms' or 'import rl_x.environments'
            # 2. Capture the immediate next segment (submodule) to check against whitelist
            match = re.search(r"(from|import)\s+rl_x\.(algorithms|environments)(\.[a-zA-Z0-9_]+)?", line)
            
            should_replace = False
            
            if match:
                # Extract the full path part detected, e.g., 'rl_x.algorithms.sac' or 'rl_x.algorithms.algorithm_manager'
                matched_str = match.group(0).split()[-1]
                
                # Convert 'rl_x.algorithms.sac' -> 'algorithms.sac'
                sub_path = matched_str.replace("rl_x.", "", 1)
                
                # Check whitelist
                is_core = False
                for keep in KEEP_MODULES:
                    # Match exact module or submodule (e.g. algorithms.algorithm_manager.some_func)
                    if sub_path == keep or sub_path.startswith(keep + "."):
                        is_core = True
                        break
                
                if not is_core:
                    should_replace = True

            if should_replace:
                # Perform safe string replacement for this line
                new_line = line.replace("rl_x.algorithms", f"{project_name}.algorithms")
                new_line = new_line.replace("rl_x.environments", f"{project_name}.environments")
                new_lines.append(new_line)
                modified = True
            else:
                new_lines.append(line)
            
        if modified:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
                
    print("Imports refactored.")

def setup_project(algos, envs):
    cwd = Path.cwd()
    project_name = cwd.name
    
    # 1. Locate Installed Paths
    package_root = Path(rl_x.__file__).parent
    site_packages = package_root.parent
    experiments_src = site_packages / "experiments"

    print(f"Initializing RL-X project: {project_name}")

    # 2. Determine Source Folders (Handle singular/plural mismatch)
    src_algo_root = find_folder(package_root, ["algorithms", "algorithm"])
    src_env_root = find_folder(package_root, ["environments", "environment"])

    if not src_algo_root:
        print(f"Error: Could not find an algorithms folder in {package_root}")
        return

    if not src_env_root:
        print(f"Error: Could not find an environments folder in {package_root}")
        return

    # 3. Create Project Structure
    # Creates /path/to/project_name/project_name (standard python src layout)
    target_src_root = cwd / project_name
    target_src_root.mkdir(parents=True, exist_ok=True)
    (target_src_root / "__init__.py").touch()
    
    # Destination containers (Always use Plural for user project)
    dst_algo_root = target_src_root / "algorithms"
    dst_env_root = target_src_root / "environments"
    
    dst_algo_root.mkdir(exist_ok=True)
    dst_env_root.mkdir(exist_ok=True)
    (dst_algo_root / "__init__.py").touch()
    (dst_env_root / "__init__.py").touch()

    # 4. Copy Experiments (to project root)
    if experiments_src.exists():
        target_exp = cwd / "experiments"
        if not target_exp.exists():
            shutil.copytree(experiments_src, target_exp)
            print("Copied experiments folder.")
        else:
            print("Experiments folder already exists, skipping.")
    else:
        print("Warning: 'experiments' folder not found in site-packages.")

    # 5. Copy Specific Algorithms
    if algos is None:
        algos = []
    for algo in algos:
        src = src_algo_root / algo
        dst = dst_algo_root / algo
        
        if src.exists():
            if dst.exists():
                print(f"Algorithm '{algo}' already exists.")
            else:
                shutil.copytree(src, dst)
                print(f"Added algorithm: {algo}")
        else:
            print(f"Error: Algorithm '{algo}' not found.")
            available = [x.name for x in src_algo_root.iterdir() if x.is_dir() and not x.name.startswith('__')]
            print(f"  Available: {available}")

    # 6. Copy Specific Environments
    if envs is None:
        envs = []
    for env in envs:
        src = src_env_root / env
        dst = dst_env_root / env
        
        if src.exists():
            if dst.exists():
                print(f"Environment '{env}' already exists.")
            else:
                shutil.copytree(src, dst)
                print(f"Added environment: {env}")
        else:
            print(f"Error: Environment '{env}' not found.")
            available = [x.name for x in src_env_root.iterdir() if x.is_dir() and not x.name.startswith('__')]
            print(f"  Available: {available}")

    # 7. Refactor Imports
    update_imports(target_src_root, project_name)

    # 8. Create .gitignore
    if not (cwd / ".gitignore").exists():
        with open(cwd / ".gitignore", "w") as f:
            f.write(GITIGNORE_CONTENT)
        print("Created .gitignore")
    
    print(f"\nProject {project_name} ready!")

if __name__ == "__main__":
    main()