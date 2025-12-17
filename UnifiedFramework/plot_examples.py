"""
Example usage of advanced plotting functions with existing results
This script shows how to use the plotting functions with your existing results data
"""

import json
import glob
import os
from evaluation.advanced_plots import AdvancedPlots


def load_existing_results():
    """Load existing experimental results from the results directory"""
    results_files = glob.glob('results/results_*.json')
    all_results = []
    
    for file_path in results_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_results.extend(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_results


def process_results_for_plotting(results):
    """Process results data for plotting functions"""
    # Group by attack method
    attack_data = {}
    defense_data = {}
    
    for result in results:
        attack = result['attack']
        defense = result['defense']
        
        if attack not in attack_data:
            attack_data[attack] = {
                'acc_before': [],
                'asr_before': [],
                'acc_after': [],
                'asr_after': []
            }
        
        attack_data[attack]['acc_before'].append(result.get('ACC_before_defense', 0))
        attack_data[attack]['asr_before'].append(result.get('ASR_before_defense', 0))
        attack_data[attack]['acc_after'].append(result.get('ACC_after_defense', 0))
        attack_data[attack]['asr_after'].append(result.get('ASR_after_defense', 0))
        
        if defense not in defense_data:
            defense_data[defense] = {
                'acc_improvement': [],
                'asr_reduction': []
            }
        
        defense_data[defense]['acc_improvement'].append(result.get('ACC_improvement', 0))
        defense_data[defense]['asr_reduction'].append(result.get('ASR_reduction', 0))
    
    return attack_data, defense_data


def create_custom_plots():
    """Create custom plots using existing results"""
    plotter = AdvancedPlots()
    
    # Load existing results
    results = load_existing_results()
    
    if not results:
        print("No existing results found. Generating example plots with synthetic data...")
        # Generate all plots with synthetic data
        from generate_plots import AdvancedExperimentPlotter
        plotter_instance = AdvancedExperimentPlotter()
        plotter_instance.generate_comprehensive_analysis()
        return
    
    print(f"Found {len(results)} experimental results")
    
    # Process results
    attack_data, defense_data = process_results_for_plotting(results)
    
    # 1. Create defense effectiveness comparison
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Defense effectiveness bar chart
    defenses = list(defense_data.keys())
    acc_improvements = [np.mean(defense_data[d]['acc_improvement']) for d in defenses]
    asr_reductions = [np.mean(defense_data[d]['asr_reduction']) for d in defenses]
    
    x = np.arange(len(defenses))
    width = 0.35
    
    ax1.bar(x - width/2, acc_improvements, width, label='ACC Improvement', alpha=0.8)
    ax1.bar(x + width/2, asr_reductions, width, label='ASR Reduction', alpha=0.8)
    ax1.set_xlabel('Defense Methods')
    ax1.set_ylabel('Improvement (%)')
    ax1.set_title('Defense Method Effectiveness')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.upper() for d in defenses])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Attack method comparison
    attacks = list(attack_data.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(attacks)))
    
    for i, attack in enumerate(attacks):
        acc_before = np.mean(attack_data[attack]['acc_before'])
        asr_before = np.mean(attack_data[attack]['asr_before'])
        ax2.scatter(acc_before, asr_before, c=[colors[i]], s=100, 
                   label=attack.upper(), alpha=0.7)
    
    ax2.set_xlabel('Clean Accuracy Before Defense (%)')
    ax2.set_ylabel('Attack Success Rate Before Defense (%)')
    ax2.set_title('Attack Methods Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Defense trade-off analysis
    for defense in defenses:
        if defense != 'none':  # Skip baseline
            acc_imp = defense_data[defense]['acc_improvement']
            asr_red = defense_data[defense]['asr_reduction']
            ax3.scatter(acc_imp, asr_red, s=60, alpha=0.7, label=defense.upper())
    
    ax3.set_xlabel('Accuracy Improvement (%)')
    ax3.set_ylabel('ASR Reduction (%)')
    ax3.set_title('Defense Trade-off Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Overall performance summary
    overall_scores = []
    for defense in defenses:
        if defense != 'none':
            score = np.mean(defense_data[defense]['acc_improvement']) + \
                   np.mean(defense_data[defense]['asr_reduction'])
            overall_scores.append(score)
        else:
            overall_scores.append(0)
    
    bars = ax4.bar(defenses, overall_scores, alpha=0.8)
    ax4.set_ylabel('Overall Defense Score')
    ax4.set_title('Overall Defense Performance')
    ax4.set_xticklabels([d.upper() for d in defenses])
    ax4.grid(True, alpha=0.3)
    
    # Color bars based on performance
    for i, bar in enumerate(bars):
        if overall_scores[i] > 40:
            bar.set_color('green')
        elif overall_scores[i] > 20:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig('figures/results_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("Custom analysis plot saved to 'figures/results_analysis.png'")


def quick_demo():
    """Quick demonstration of all plotting functions"""
    print("Quick Demo: Generating all plots with synthetic data...")
    
    # Create plotter instance
    plotter = AdvancedPlots(figures_dir='demo_figures')
    
    # 1. Poisoning rate analysis
    print("1. Poisoning rate analysis...")
    plotter.plot_poisoning_rate_performance()
    
    # 2. Finetuning analysis (with synthetic data)
    print("2. Finetuning analysis...")
    epochs, asr_data, acc_data = plotter._generate_synthetic_finetuning_data()
    plotter.plot_finetuning_analysis(epochs, asr_data, acc_data)
    
    # 3. Training loss analysis
    print("3. Training loss analysis...")
    training_data = plotter._generate_synthetic_training_data()
    plotter.plot_training_loss_analysis(training_data)
    
    print("Demo complete! Check 'demo_figures' directory for output plots.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        quick_demo()
    else:
        create_custom_plots()