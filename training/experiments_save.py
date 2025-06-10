import os
import json
import pickle
import pandas as pd
from datetime import datetime
import numpy as np

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_model_results(classical_results, mse_mse, mse_r2, npp_mse, npp_r2, 
                      save_dir="model_results", experiment_name=None):
    """
    Save all model results for later exploration
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate experiment name with timestamp if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    # Compile all results - convert numpy types to native Python types
    all_results = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "neural_networks": {
            "MSE": {"mse": float(mse_mse), "r2": float(mse_r2)},
            "NPP": {"mse": float(npp_mse), "r2": float(npp_r2)}
        },
        "classical_models": {}
    }
    
    # Add classical model results (excluding the model objects)
    for name, results in classical_results.items():
        all_results["classical_models"][name] = {
            "mse": float(results["mse"]),
            "r2": float(results["r2"])
        }
    
    # Convert any remaining numpy types
    all_results = convert_numpy_types(all_results)
    
    # Save summary as JSON
    json_path = os.path.join(save_dir, f"{experiment_name}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save detailed results with predictions as pickle
    detailed_results = {
        "neural_networks": {
            "MSE": {"mse": mse_mse, "r2": mse_r2},
            "NPP": {"mse": npp_mse, "r2": npp_r2}
        },
        "classical_models": classical_results  # Includes models and predictions
    }
    
    pickle_path = os.path.join(save_dir, f"{experiment_name}_detailed.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(detailed_results, f)
    
    # Create a CSV for easy viewing
    model_names = ["MSE", "NPP"] + list(classical_results.keys())
    mses = [float(mse_mse), float(npp_mse)] + [float(classical_results[k]["mse"]) for k in classical_results]
    r2s = [float(mse_r2), float(npp_r2)] + [float(classical_results[k]["r2"]) for k in classical_results]
    
    df = pd.DataFrame({
        'Model': model_names,
        'MSE': mses,
        'R2': r2s
    })
    df = df.sort_values('MSE')  # Sort by performance
    
    csv_path = os.path.join(save_dir, f"{experiment_name}_results.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to:")
    print(f"  Summary (JSON): {json_path}")
    print(f"  Detailed (PKL): {pickle_path}")
    print(f"  CSV: {csv_path}")
    
    return {
        "json_path": json_path,
        "pickle_path": pickle_path,
        "csv_path": csv_path,
        "experiment_name": experiment_name
    }

def load_results(experiment_name, save_dir="model_results"):
    """
    Load saved results for analysis
    """
    json_path = os.path.join(save_dir, f"{experiment_name}_summary.json")
    pickle_path = os.path.join(save_dir, f"{experiment_name}_detailed.pkl")
    csv_path = os.path.join(save_dir, f"{experiment_name}_results.csv")
    
    results = {}
    
    # Load JSON summary
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results['summary'] = json.load(f)
    
    # Load detailed pickle
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            results['detailed'] = pickle.load(f)
    
    # Load CSV
    if os.path.exists(csv_path):
        results['dataframe'] = pd.read_csv(csv_path)
    
    return results

def compare_experiments(experiment_names, save_dir="model_results"):
    """
    Compare results across multiple experiments
    """
    all_data = []
    
    for exp_name in experiment_names:
        results = load_results(exp_name, save_dir)
        if 'dataframe' in results:
            df = results['dataframe'].copy()
            df['Experiment'] = exp_name
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        print("No results found for the specified experiments")
