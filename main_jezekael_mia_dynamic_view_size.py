import subprocess

# Define paths to your scripts
script_paths = [
    "main_jezekael_dynamic_mia.py",
    "main_jezekael_dynamic_mia_neighbors2.py",
    "main_jezekael_dynamic_mia_neighbors10.py",
    "main_jezekael_dynamic_mia_neighbors20.py",
    "main_jezekael_dynamic_mia_neighborsfull.py",
]

# Iterate over each script and execute it
for script_path in script_paths:
    print(f"Executing script: {script_path}")
    subprocess.run(["python", script_path])