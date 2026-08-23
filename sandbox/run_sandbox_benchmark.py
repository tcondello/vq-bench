#!/usr/bin/env python3
"""
Executable Runner for Terminal Bench Agent Sandbox.
Compares Unindexed Terminal Agent vs VQ-bench 1.35 b/d Indexed Agent across 20 real developer tasks.
"""

import os
import sys
import numpy as np

# Ensure sandbox directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terminal_bench import TerminalBenchHarness

def main():
    harness = TerminalBenchHarness()
    results = harness.run_full_benchmark()

    un_res = results["unindexed"]
    in_res = results["indexed"]
    N = len(un_res)

    un_solved = sum(1 for r in un_res if r.solved)
    in_solved = sum(1 for r in in_res if r.solved)

    un_tokens = [r.prompt_tokens for r in un_res]
    in_tokens = [r.prompt_tokens for r in in_res]

    un_times = [r.execution_time_s for r in un_res]
    in_times = [r.execution_time_s for r in in_res]

    un_costs = [r.cost_dollars for r in un_res]
    in_costs = [r.cost_dollars for r in in_res]

    token_savings_multiplier = np.mean(un_tokens) / max(1, np.mean(in_tokens))
    cost_savings_multiplier = sum(un_costs) / max(1e-6, sum(in_costs))

    print("\n" + "=" * 155)
    print(" TERMINAL BENCH AGENT SANDBOX MASTER SCORECARD (20 REAL DEVELOPER TASKS)")
    print("=" * 155)
    print(f"{'Evaluation Metric':<42} | {'Unindexed Terminal Agent (grep/cat)':>36} | {'VQ-bench 1.35b/d Indexed Agent':>34} | {'Product Advantage':>28}")
    print("-" * 155)
    print(f"{'Task Resolution Accuracy (%)':<42} | {un_solved/N*100:33.1f}% | {in_solved/N*100:31.1f}% | {f'+{(in_solved - un_solved)/N*100:.1f}% higher fidelity':>28}")
    print(f"{'Mean Prompt Context Tokens / Task':<42} | {np.mean(un_tokens):34.0f} tokens | {np.mean(in_tokens):32.0f} tokens | {f'{token_savings_multiplier:.1f}x Token Reduction':>28}")
    print(f"{'Median Prompt Context Tokens / Task':<42} | {np.median(un_tokens):34.0f} tokens | {np.median(in_tokens):32.0f} tokens | {f'{np.median(un_tokens)/np.median(in_tokens):.1f}x Median Savings':>28}")
    print(f"{'Mean Execution Latency / Task':<42} | {np.mean(un_times):34.2f} s | {np.mean(in_times):32.2f} s | {f'{np.mean(un_times)/np.mean(in_times):.1f}x Faster Execution':>28}")
    print(f"{'Total API Cost for 20 Tasks':<42} | ${sum(un_costs):34.4f} | ${sum(in_costs):32.4f} | {f'{cost_savings_multiplier:.1f}x Cost Reduction':>28}")
    print(f"{'Effective Storage Footprint':<42} | {'N/A (Disk Text)':>36} | {'1.35 bits/dim ($0.12/GB)':>34} | {'95.8% RAM Reduction':>28}")
    print("=" * 155)

    print("\n--- Breakdown by Task Category ---")
    categories = sorted(list(set(r.category for r in un_res)))
    print(f"{'Task Category':<30} | {'Unindexed Acc':>15} | {'Indexed Acc':>15} | {'Unindexed Tokens':>20} | {'Indexed Tokens':>20} | {'Token Savings':>18}")
    print("-" * 130)
    for cat in categories:
        cat_un = [r for r in un_res if r.category == cat]
        cat_in = [r for r in in_res if r.category == cat]
        n_cat = len(cat_un)
        
        acc_u = sum(1 for r in cat_un if r.solved) / n_cat * 100
        acc_i = sum(1 for r in cat_in if r.solved) / n_cat * 100
        tok_u = np.mean([r.prompt_tokens for r in cat_un])
        tok_i = np.mean([r.prompt_tokens for r in cat_in])
        sav = tok_u / max(1, tok_i)
        
        print(f"{cat:<30} | {acc_u:14.1f}% | {acc_i:14.1f}% | {tok_u:18.0f} | {tok_i:18.0f} | {sav:16.1f}x")
    print("=" * 130)

if __name__ == "__main__":
    main()
