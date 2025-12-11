"""
Visualization and results analysis for experiments
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import seaborn as sns


class ResultsAnalyzer:
    """Analyze and visualize experimental results"""
    
    def __init__(self, results_file: str = None, results_list: List[Dict] = None):
        """
        Initialize analyzer
        
        Args:
            results_file: Path to JSON results file
            results_list: List of result dictionaries
        """
        if results_file:
            with open(results_file, 'r') as f:
                self.results = json.load(f)
        elif results_list:
            self.results = results_list
        else:
            self.results = []
        
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        return pd.DataFrame(self.results)
    
    def plot_acc_comparison(self, save_path: str = None, figsize: Tuple = (14, 8)):
        """Plot clean accuracy comparison across methods"""
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Before defense
        df_before = self.df[['attack', 'defense', 'ACC_before_defense']].copy()
        df_before = df_before.pivot_table(
            values='ACC_before_defense', 
            index='attack', 
            columns='defense'
        )
        
        df_before.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('Clean Accuracy (ACC) Before Defense', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_xlabel('Attack Method')
        axes[0].legend(title='Defense', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 105])
        
        # After defense (if available)
        if 'ACC_after_defense' in self.df.columns:
            df_after = self.df[self.df['defense'] != 'none'].copy()
            df_after = df_after[['attack', 'defense', 'ACC_after_defense']].copy()
            df_after = df_after.pivot_table(
                values='ACC_after_defense', 
                index='attack', 
                columns='defense'
            )
            
            df_after.plot(kind='bar', ax=axes[1], width=0.8)
            axes[1].set_title('Clean Accuracy (ACC) After Defense', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_xlabel('Attack Method')
            axes[1].legend(title='Defense', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim([0, 105])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def plot_asr_comparison(self, save_path: str = None, figsize: Tuple = (14, 8)):
        """Plot attack success rate comparison across methods"""
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Before defense
        df_before = self.df[['attack', 'defense', 'ASR_before_defense']].copy()
        df_before = df_before.pivot_table(
            values='ASR_before_defense', 
            index='attack', 
            columns='defense'
        )
        
        df_before.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('Attack Success Rate (ASR) Before Defense', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('ASR (%)')
        axes[0].set_xlabel('Attack Method')
        axes[0].legend(title='Defense', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 105])
        
        # After defense (if available)
        if 'ASR_after_defense' in self.df.columns:
            df_after = self.df[self.df['defense'] != 'none'].copy()
            df_after = df_after[['attack', 'defense', 'ASR_after_defense']].copy()
            df_after = df_after.pivot_table(
                values='ASR_after_defense', 
                index='attack', 
                columns='defense'
            )
            
            df_after.plot(kind='bar', ax=axes[1], width=0.8)
            axes[1].set_title('Attack Success Rate (ASR) After Defense', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('ASR (%)')
            axes[1].set_xlabel('Attack Method')
            axes[1].legend(title='Defense', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim([0, 105])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def plot_defense_effectiveness(self, save_path: str = None, figsize: Tuple = (12, 6)):
        """Plot defense effectiveness (ACC improvement vs ASR reduction)"""
        if 'ACC_improvement' not in self.df.columns:
            print("ACC_improvement not found in results")
            return
        
        df_defense = self.df[self.df['defense'] != 'none'].copy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data
        defense_summary = df_defense.groupby('defense').agg({
            'ACC_improvement': 'mean',
            'ASR_reduction': 'mean'
        }).reset_index()
        
        x = np.arange(len(defense_summary))
        width = 0.35
        
        ax.bar(x - width/2, defense_summary['ACC_improvement'], width, 
               label='ACC Improvement', color='steelblue')
        ax.bar(x + width/2, defense_summary['ASR_reduction'], width, 
               label='ASR Reduction', color='coral')
        
        ax.set_xlabel('Defense Method', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title('Defense Effectiveness Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(defense_summary['defense'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def plot_heatmap(self, metric: str = 'ACC_before_defense', save_path: str = None,
                    figsize: Tuple = (10, 6)):
        """Plot heatmap of metric across attacks and defenses"""
        pivot_data = self.df.pivot_table(
            values=metric,
            index='attack',
            columns='defense',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   cbar_kws={'label': metric}, ax=ax, vmin=0, vmax=100)
        
        ax.set_title(f'{metric} Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Defense Method')
        ax.set_ylabel('Attack Method')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def print_summary_table(self):
        """Print summary statistics table"""
        print("\n" + "="*100)
        print("EXPERIMENTAL RESULTS SUMMARY")
        print("="*100 + "\n")
        
        # Summary by attack
        print("Average Results by Attack Method:")
        attack_summary = self.df.groupby('attack').agg({
            'ACC_before_defense': 'mean',
            'ASR_before_defense': 'mean',
            'ACC_after_defense': 'mean' if 'ACC_after_defense' in self.df.columns else 'first',
            'ASR_after_defense': 'mean' if 'ASR_after_defense' in self.df.columns else 'first',
        })
        print(attack_summary.to_string())
        
        # Summary by defense
        print("\n\nAverage Results by Defense Method:")
        defense_summary = self.df.groupby('defense').agg({
            'ACC_before_defense': 'mean',
            'ASR_before_defense': 'mean',
            'ACC_after_defense': 'mean' if 'ACC_after_defense' in self.df.columns else 'first',
            'ASR_after_defense': 'mean' if 'ASR_after_defense' in self.df.columns else 'first',
        })
        print(defense_summary.to_string())
        
        # Best defense per attack
        print("\n\nBest Defense per Attack (by ASR Reduction):")
        if 'ASR_reduction' in self.df.columns:
            best_defense = self.df.loc[self.df.groupby('attack')['ASR_reduction'].idxmax()]
            print(best_defense[['attack', 'defense', 'ASR_reduction']].to_string(index=False))
        
        print("\n" + "="*100 + "\n")


def generate_complete_report(results_file: str, output_dir: str = './reports'):
    """Generate complete report with all visualizations"""
    analyzer = ResultsAnalyzer(results_file)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    analyzer.plot_acc_comparison(f'{output_dir}/acc_comparison.png')
    analyzer.plot_asr_comparison(f'{output_dir}/asr_comparison.png')
    analyzer.plot_defense_effectiveness(f'{output_dir}/defense_effectiveness.png')
    analyzer.plot_heatmap('ACC_before_defense', f'{output_dir}/acc_heatmap.png')
    analyzer.plot_heatmap('ASR_before_defense', f'{output_dir}/asr_heatmap.png')
    
    # Print summary
    analyzer.print_summary_table()
    
    print(f"Report generated in {output_dir}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else './reports'
        generate_complete_report(results_file, output_dir)
    else:
        print("Usage: python analyze_results.py <results_file> [output_dir]")
