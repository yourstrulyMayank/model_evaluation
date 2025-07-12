import os
import json
from collections import defaultdict

ML_BENCHMARKS = [
    "MLflow", "scikit-learn", "Yellowbrick", "Evidently AI", "Weights & Biases", "AutoML (TPOT, H2O)"
]

def load_ml_history(model_name, category="supervised"):
    """
    Load ML evaluation history for a given model and category.
    """
    history_file = f"evaluation_results/ml_history_{category}.json"
    history_data = []
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            all_history = json.load(f)
        # Filter for this model
        model_history = [entry for entry in all_history if entry.get("model_name") == model_name]
        # Sort by run number or timestamp
        model_history.sort(key=lambda x: x.get("run", 0))
        history_data = model_history

    # Calculate benchmark averages
    benchmark_averages = {}
    benchmark_counts = defaultdict(int)
    benchmark_sums = defaultdict(float)
    for entry in history_data:
        for benchmark, score in entry.get("scores", {}).items():
            if score != 'N/A' and isinstance(score, (int, float)):
                benchmark_sums[benchmark] += score
                benchmark_counts[benchmark] += 1
    for benchmark in ML_BENCHMARKS:
        if benchmark_counts[benchmark] > 0:
            benchmark_averages[benchmark] = benchmark_sums[benchmark] / benchmark_counts[benchmark]
        else:
            benchmark_averages[benchmark] = 'N/A'

    # Calculate summary statistics
    all_scores = []
    benchmarks_tested = set()
    for entry in history_data:
        for benchmark, score in entry.get("scores", {}).items():
            if score != 'N/A' and isinstance(score, (int, float)):
                all_scores.append(score)
                benchmarks_tested.add(benchmark)
    benchmark_stats = {
        'benchmarks_tested': len(benchmarks_tested),
        'overall_average': sum(all_scores) / len(all_scores) if all_scores else 0,
        'best_score': max(all_scores) if all_scores else 0
    }

    return history_data, ML_BENCHMARKS, benchmark_averages, benchmark_stats