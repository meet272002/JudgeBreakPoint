import subprocess
import sys

scripts = [
    "Scripts/get_data.py",
    "Scripts/preprocess_data.py",
    "Scripts/generate_cases.py",
    "Scripts/generate_qa.py",   
    "Scripts/run_models.py",
    "Scripts/run_judge.py",
    "Scripts/compute_similarity.py",
    "Scripts/merge_results.py"
]

for script in scripts:
    print(f"\nRunning {script}")

    result = subprocess.run(
        [sys.executable, script],
        check=True
    )

print("\nPipeline completed successfully.")