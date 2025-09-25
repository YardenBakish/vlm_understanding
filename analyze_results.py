import os
import json
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pandas as pd

def find_model_directories(eval_dir):
    """Find all model directories that contain a 2.2B subdirectory"""
    model_dirs = []
    
    if not os.path.exists(eval_dir):
        print(f"Directory {eval_dir} does not exist")
        return model_dirs
    
    for item in os.listdir(eval_dir):
        item_path = os.path.join(eval_dir, item)
        if os.path.isdir(item_path):
            # Check if this model directory has a 2.2B subdirectory
            subdir_path = os.path.join(item_path, "2.2B")
            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                model_dirs.append((item, subdir_path))
    
    return model_dirs

def parse_filename(filename):
    """Parse filename to extract test type and epoch"""
    # Pattern: res_{test}_{epoch}.json
    pattern = r'res_(.+?)_(\d+)\.json'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def parse_tempcomp_results(data):
    """Parse tempcomp test results"""
    total_correct = 0
    total_precise = 0
    total_num = 0
    results_by_key = {}
    
    for key, value in data.items():
        if isinstance(value, dict) and all(k in value for k in ["num_correct", "num_correct_precise", "num_total"]):
            num_correct = value["num_correct"]
            num_correct_precise = 0#value["num_correct_precise"]
            num_total = value["num_total"]
            
            # Individual key performance
            key_performance = (num_correct_precise + num_correct) / num_total if num_total > 0 else 0
            results_by_key[key] = {
                'performance': key_performance,
                'num_correct': num_correct,
                'num_correct_precise': num_correct_precise,
                'num_total': num_total
            }
            
            # Accumulate for overall performance
            total_correct += num_correct
            total_precise += num_correct_precise
            total_num += num_total
    
    # Overall performance
    overall_performance = (total_precise + total_correct) / total_num if total_num > 0 else 0
    
    return {
        'overall_performance': overall_performance,
        'by_key': results_by_key,
        'totals': {
            'total_correct': total_correct,
            'total_precise': total_precise,
            'total_num': total_num
        }
    }

def parse_blur_video_results(data):
    """Parse blur_video test results"""
    if "num_total" not in data:
        return None
    
    num_total = data["num_total"]
    sim_values = {}
    
    # Find all keys that contain dictionaries with "sim" values
    for key, value in data.items():
        if key != "num_total" and isinstance(value, dict) and "sim" in value:
            sim_values[key] = value["sim"]
        #else:
        #    sim_values[key] = value
    
    if not sim_values:
        return None
    
    K = len(sim_values)
    
    # Calculate normalized sim values
    normalized_sims = {}
    for key, sim in sim_values.items():
        normalized_sims[key] = sim / (num_total / K) if num_total > 0 else 0
    
    # Calculate AUC (assuming keys can be sorted numerically)
    try:
        # Try to sort keys as numbers
        print(sim_values)

        sorted_keys = sorted(sim_values.keys(), key=lambda x: float(x))
        x_values = [float(k) for k in sorted_keys]
        y_values = [normalized_sims[k] for k in sorted_keys]

        

        
        # Calculate AUC
        if len(x_values) > 1:
            auc_score = auc(x_values, y_values)
        else:
            auc_score = y_values[0] if y_values else 0
    except ValueError:
        print("HERE")
        exit(1)

        # If keys can't be converted to numbers, use string sorting
        sorted_keys = sorted(sim_values.keys())
        x_values = list(range(len(sorted_keys)))
        y_values = [normalized_sims[k] for k in sorted_keys]
        
        if len(x_values) > 1:
            auc_score = auc(x_values, y_values)
        else:
            auc_score = y_values[0] if y_values else 0
    
    return {
        'overall_performance': auc_score,  # AUC as overall performance metric
        'normalized_sims': normalized_sims,
        'num_total': num_total,
        'K': K,
        'sorted_keys': sorted_keys,
        'auc_score': auc_score
    }

def collect_results(eval_dir, target_tests=["tempcomp","tempcomp_yes_no","tempcomp_caption_matching", "blur_video_ker", "blur_video"]):
    """Collect all results from the evaluation directory"""
    model_dirs = find_model_directories(eval_dir)
    
    if not model_dirs:
        print("No model directories with 2.2B subdirectory found")
        return {}
    
    all_results = defaultdict(lambda: defaultdict(dict))  # test_type -> model -> epoch -> results
    
    for model_name, model_path in model_dirs:
        print(f"Processing model: {model_name}")
        
        # Look for JSON files in the 2.2B directory
        for filename in os.listdir(model_path):
            if filename.endswith('.json'):
                test_type, epoch = parse_filename(filename)
                
                if test_type in target_tests and epoch is not None:
                    filepath = os.path.join(model_path, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        # Parse based on test type
                        if "tempcomp" in test_type:
                            results = parse_tempcomp_results(data)
                        elif "blur_video" in test_type:
                            results = parse_blur_video_results(data)
                        else:
                            continue
                        
                        if results:
                            all_results[test_type][model_name][epoch] = results
                            
                    except Exception as e:
                        exit(1)
                        print(f"Error processing {filepath}: {e}")
    
    return dict(all_results)




def find_missing_combinations(results):
    """
    Find which test_types are missing model_name, epoch combinations
    compared to the test_type with the most complete data.
    """
    # Collect all model-epoch combinations for each test_type
    test_type_combinations = {}
    
    for test_type in results.keys():
        combinations = set()
        for model_name, epochs_data in results[test_type].items():
            for epoch in epochs_data.keys():
                combinations.add((model_name, epoch))
        test_type_combinations[test_type] = combinations
    
    # Find the test_type with the most combinations
    most_complete_test_type = max(test_type_combinations.keys(), 
                                  key=lambda x: len(test_type_combinations[x]))
    reference_combinations = test_type_combinations[most_complete_test_type]
    
    print(f"Reference test_type (most complete): {most_complete_test_type}")
    print(f"Total combinations in reference: {len(reference_combinations)}")
    print(f"{'='*80}")
    
    # Check what each other test_type is missing
    for test_type in sorted(test_type_combinations.keys()):
        if test_type == most_complete_test_type:
            continue
            
        current_combinations = test_type_combinations[test_type]
        missing_combinations = reference_combinations - current_combinations
        
        print(f"\nTest type: {test_type}")
        print(f"Has {len(current_combinations)} combinations (missing {len(missing_combinations)})")
        
        if missing_combinations:
            print("Missing combinations:")
            for model_name, epoch in sorted(missing_combinations):
                print(f"  - Model: {model_name}, Epoch: {epoch}")
        else:
            print("✓ No missing combinations")
    
    return most_complete_test_type, reference_combinations, test_type_combinations



def print_results(results):
    """Print results in decreasing order of performance"""

    #find_missing_combinations(results)
    #exit(1)

    for test_type in sorted(results.keys()):
        print(f"\n{'='*60}")
        print(f"TEST TYPE: {test_type.upper()}")
        print(f"{'='*60}")
        
        # Collect all model-epoch combinations with their performance
        performance_data = []
        
        for model_name, epochs_data in results[test_type].items():
            for epoch, result_data in epochs_data.items():
                performance = result_data['overall_performance']
                performance_data.append((performance, model_name, epoch, result_data))
        
        # Sort by performance (descending)
        performance_data.sort(key=lambda x: x[0], reverse=True)
        
        # Print results
        print(f"num types: {len(performance_data)}")
        for i, (performance, model_name, epoch, result_data) in enumerate(performance_data, 1):
            print(f"\n{i:2d}. Model: {model_name:20s} | Epoch: {epoch:3d} | Performance: {performance:.4f}")
            
            if "tempcomp" in test_type:
                print(f"     Total: {result_data['totals']['total_correct'] + result_data['totals']['total_precise']}/{result_data['totals']['total_num']}")
                print(f"     By key:")
                for key, key_data in sorted(result_data['by_key'].items()):
                    #if test_type!= "tempcomp_motion_bench":
                    #    if (key != "direction") and (key!="speed"):
                    #        continue
                    print(f"       {key}: {key_data['performance']:.4f} ({key_data['num_correct_precise'] + key_data['num_correct']}/{key_data['num_total']})")
            
            elif "blur_video" in test_type:
                print(f"     NUM TOTAL: {result_data['num_total']:.1f}")

                print(f"     AUC Score: {result_data['auc_score']:.4f}")
                print(f"     Normalized similarities:")
                for key in result_data['sorted_keys']:
                    print(f"       {key}: {result_data['normalized_sims'][key]:.4f}")

def create_visualization(results):
    """Create visualizations for the results"""
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 6 * len(results)))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (test_type, test_results) in enumerate(results.items()):
        ax = axes[idx]
        
        # Collect data for plotting
        models = []
        epochs = []
        performances = []
        
        for model_name, epochs_data in test_results.items():
            for epoch, result_data in epochs_data.items():
                models.append(f"{model_name}_e{epoch}")
                epochs.append(epoch)
                performances.append(result_data['overall_performance'])
        
        # Sort by performance
        sorted_data = sorted(zip(performances, models, epochs), reverse=True)
        sorted_performances, sorted_models, sorted_epochs = zip(*sorted_data) if sorted_data else ([], [], [])
        
        # Create bar plot
        bars = ax.bar(range(len(sorted_performances)), sorted_performances)
        ax.set_xlabel('Model (Epoch)')
        ax.set_ylabel('Performance')
        ax.set_title(f'{test_type.upper()} - Performance by Model and Epoch')
        ax.set_xticks(range(len(sorted_models)))
        ax.set_xticklabels(sorted_models, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, perf in zip(bars, sorted_performances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{perf:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()




def extract_best_variant_epoch_combinations(data_dict, target_keys=['speed', 'direction']):
    """
    Extracts variant-epoch combinations and ranks them by summed scores across all test_types.
    
    Args:
        data_dict: Dictionary with structure {test_type: {variant: {epoch_num: {'by_key': {key: {'num_correct': x, 'num_total': y}}}}}}
        target_keys: List of keys to sum scores for (default: ['speed', 'direction'])
    
    Returns:
        List of tuples: [(variant, epoch_num, total_score), ...] sorted by descending score
    """
    combination_scores = {}
    all_keys = set()
    
    # First pass: discover all keys
    for test_type, test_data in data_dict.items():
        for variant, variant_data in test_data.items():
            for epoch_num, epoch_data in variant_data.items():
                if 'by_key' in epoch_data:
                    all_keys.update(epoch_data['by_key'].keys())
    
    # Initialize structure for each discovered key + global
    for key in list(all_keys) + ['global']:
        combination_scores[key] = {}
    
    # Iterate through all test_types
    for test_type, test_data in data_dict.items():
        # Iterate through variants
        for variant, variant_data in test_data.items():
            # Iterate through epoch numbers
            for epoch_num, epoch_data in variant_data.items():
                # Create combination key
                combo_key = (variant, epoch_num)
                
                # Extract scores for target keys from 'by_key' section
                if 'by_key' in epoch_data:
                    by_key_data = epoch_data['by_key']
                    
                    # Process each target key separately
                    for target_key in target_keys:
                        if target_key in by_key_data:
                            # Initialize score for this combination if not seen before
                            if combo_key not in combination_scores[target_key]:
                                combination_scores[target_key][combo_key] = {}
                                combination_scores[target_key][combo_key]['num_correct'] = 0
                                combination_scores[target_key][combo_key]['num_total'] = 0
                            
                            key_data = by_key_data[target_key]
                            
                            # Accumulate num_correct and num_total
                            if 'num_correct' in key_data and 'num_total' in key_data:
                                if key_data['num_total'] > 0:  # Avoid division by zero
                                    combination_scores[target_key][combo_key]['num_correct'] += key_data['num_correct']
                                    combination_scores[target_key][combo_key]['num_total'] += key_data['num_total']
    
    # Calculate final averages and create ranked lists for each key
    results = {}
    for target_key in target_keys:
        ranked_combinations = []
        
        for (variant, epoch_num), score_data in combination_scores[target_key].items():
            if score_data['num_total'] > 0:
                average_score = score_data['num_correct'] / score_data['num_total']
                ranked_combinations.append((variant, epoch_num, average_score))
        
        # Sort by average score in descending order
        ranked_combinations.sort(key=lambda x: x[2], reverse=True)
        results[target_key] = ranked_combinations

    
    top_n = 5
    for key_name, ranked_combinations in results.items():
        if top_n:
            combinations_to_print = ranked_combinations[:top_n]
        else:
            combinations_to_print = ranked_combinations
        
        print(f"\nVariant-Epoch Combinations Ranked by '{key_name.upper()}' Score (Descending):")
        print("=" * 70)
        
        for i, (variant, epoch_num, average_score) in enumerate(combinations_to_print, 1):
            print(f"{i:2d}. Variant: {variant:15} | Epoch: {epoch_num:8} | {key_name.capitalize()} Score: {average_score:.4f}")
        
        print()  # Add blank line between different key rankings

    exit(1)


def extract_all_combinations(data_dict):
    """
    Extracts variant-epoch combinations and ranks them by individual key scores across all test_types.
    Automatically detects all keys in 'by_key' and calculates averages for each, plus a global average.
    
    Args:
        data_dict: Dictionary with structure {test_type: {variant: {epoch_num: {'by_key': {key: {'num_correct': x, 'num_total': y}}}}}}
    
    Returns:
        Dictionary with separate rankings for each key plus global: {key: [(variant, epoch_num, score), ...], 'global': [...]}
    """
    # Dictionary to store aggregated scores for each variant-epoch combination by key
    combination_scores = {}
    all_keys = set()
    
    # First pass: discover all keys
    for test_type, test_data in data_dict.items():
        for variant, variant_data in test_data.items():
            for epoch_num, epoch_data in variant_data.items():
                if 'by_key' in epoch_data:
                    all_keys.update(epoch_data['by_key'].keys())
    
    # Initialize structure for each discovered key + global
    for key in list(all_keys) + ['global']:
        combination_scores[key] = {}
    
    # Second pass: accumulate scores
    for test_type, test_data in data_dict.items():
        for variant, variant_data in test_data.items():
            for epoch_num, epoch_data in variant_data.items():
                combo_key = (variant, epoch_num)
                
                if 'by_key' in epoch_data:
                    by_key_data = epoch_data['by_key']
                    
                    # Initialize if not seen before
                    for key in list(all_keys) + ['global']:
                        if combo_key not in combination_scores[key]:
                            combination_scores[key][combo_key] = {'num_correct': 0, 'num_total': 0}
                    
                    # Process each key individually
                    for key_name, key_data in by_key_data.items():
                        if 'num_correct' in key_data and 'num_total' in key_data:
                            if key_data['num_total'] > 0:
                                # Add to individual key
                                combination_scores[key_name][combo_key]['num_correct'] += key_data['num_correct']
                                combination_scores[key_name][combo_key]['num_total'] += key_data['num_total']
                                
                                # Add to global total
                                combination_scores['global'][combo_key]['num_correct'] += key_data['num_correct']
                                combination_scores['global'][combo_key]['num_total'] += key_data['num_total']
    
    # Calculate final averages and create ranked lists for each key
    results = {}
    for key_name in list(all_keys) + ['global']:
        ranked_combinations = []
        
        for (variant, epoch_num), score_data in combination_scores[key_name].items():
            if score_data['num_total'] > 0:
                average_score = score_data['num_correct'] / score_data['num_total']
                ranked_combinations.append((variant, epoch_num, average_score))
        
        # Sort by average score in descending order
        ranked_combinations.sort(key=lambda x: x[2], reverse=True)
        results[key_name] = ranked_combinations
    
    top_n = 10
    for key_name, ranked_combinations in results.items():
        if top_n:
            combinations_to_print = ranked_combinations[:top_n]
        else:
            combinations_to_print = ranked_combinations
        
        print(f"\nVariant-Epoch Combinations Ranked by '{key_name.upper()}' Score (Descending):")
        print("=" * 70)
        
        for i, (variant, epoch_num, average_score) in enumerate(combinations_to_print, 1):
            print(f"{i:2d}. Variant: {variant:15} | Epoch: {epoch_num:8} | {key_name.capitalize()} Score: {average_score:.4f}")
        
        print()  # Add blank line 
    
    exit(1)


def main():
    """Main function to run the evaluation parser"""
    eval_dir = "eval"  # Change this path as needed
    target_tests = ["tempcomp","tempcomp_det_check","tempcomp_yes_no", "tempcomp_det_ker_check","tempcomp_caption_matching","tempcomp_motion_bench", "tempcomp_motion_bench_sports", "tempcomp_mvbench", "blur_video_ker", "blur_video"]
    target_tests = ["tempcomp_motion_bench_sports"]
    #target_tests = ["blur_video_ker"]
    target_tests = ["tempcomp_det_ker_check"]

    target_tests = ["tempcomp_caption_matching", "tempcomp", "tempcomp_yes_no" ] #"tempcomp_caption_matching", # tempcomp_motion_bench_sports

    #target_tests = ["blur_video_ker", "blur_video", "blur_video_ker_sorted", "blur_video_sorted"]

    #target_tests = ["tempcomp_motion_bench_sports"]


    #print("Collecting results...")
    results = collect_results(eval_dir, target_tests)
    #extract_all_combinations(results)
    #exit(1)
    
    
    
    if not results:
        print("No results found!")
        return
    
    print("Printing results...")
    print_results(results)
    
    print("\nCreating visualizations...")
    #create_visualization(results)
    
    return results

if __name__ == "__main__":
    # Run the main function
    results = main()