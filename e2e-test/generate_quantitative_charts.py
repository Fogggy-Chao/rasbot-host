import matplotlib.pyplot as plt
import numpy as np
import os

# --- Mock Data Generation ---
# Replace this with your actual collected data later!

np.random.seed(42) # for reproducibility

# Tier 1: Simple, Single-Action (e.g., move_base, stop)
# Expected: High success, low latency, few tool calls
runs_per_tier = 10
tier1_latencies = np.random.normal(loc=1.5, scale=1.0, size=runs_per_tier) * 1000 # ms, around 5.5s
tier1_successes = np.random.choice([True, False], size=runs_per_tier, p=[0.9, 0.1]) # 90% success
tier1_tool_calls = np.random.randint(1, 3, size=runs_per_tier) # 1-2 tool calls

# Tier 2: Simple Multi-Step or Basic Perception (e.g., wave arm, find object)
# Expected: Good success, medium latency, moderate tool calls
tier2_latencies = np.random.normal(loc=3.0, scale=1.5, size=runs_per_tier) * 1000 # ms, around 8s
tier2_successes = np.random.choice([True, False], size=runs_per_tier, p=[0.8, 0.2]) # 80% success
tier2_tool_calls = np.random.randint(2, 5, size=runs_per_tier) # 2-4 tool calls

# Tier 3: Complex Multi-Step, Perception + Action (e.g., find and approach, find and attempt grasp)
# Expected: Moderate success, higher latency, more tool calls
tier3_latencies = np.random.normal(loc=7.0, scale=2.5, size=runs_per_tier) * 1000 # ms, around 12s
tier3_successes = np.random.choice([True, False], size=runs_per_tier, p=[0.7, 0.3]) # 70% success
tier3_tool_calls = np.random.randint(3, 7, size=runs_per_tier) # 3-6 tool calls

# Ensure latencies are positive
tier1_latencies = np.abs(tier1_latencies)
tier2_latencies = np.abs(tier2_latencies)
tier3_latencies = np.abs(tier3_latencies)

# --- Plotting Configuration ---
FIGURE_DIR = "quantitative_figures"
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

TABLEAU_COLORS = [
    '#4E79A7', # Blue
    '#F28E2B', # Orange
    '#E15759', # Red
    '#76B7B2', # Teal
    '#59A14F', # Green
]

# --- Plotting Functions ---

def plot_average_latency(tier_names, avg_latencies, std_latencies, filename):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tier_names, avg_latencies, yerr=std_latencies, capsize=5, color=TABLEAU_COLORS[0])
    plt.ylabel('Average Latency (ms)', fontsize=12)
    plt.xlabel('Command Tier', fontsize=12)
    plt.title('Average System Latency per Command Tier', fontsize=15, pad=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + np.std(avg_latencies)*0.1, f'{yval:.0f} ms', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()
    print(f"Generated: {os.path.join(FIGURE_DIR, filename)}")

def plot_success_rate(tier_names, success_rates, filename):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tier_names, success_rates, color=TABLEAU_COLORS[1])
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.xlabel('Command Tier', fontsize=12)
    plt.title('System Success Rate per Command Tier', fontsize=15, pad=15)
    plt.ylim(0, 100)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval:.0f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()
    print(f"Generated: {os.path.join(FIGURE_DIR, filename)}")

def plot_average_tool_calls(tier_names, avg_tool_calls, filename):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tier_names, avg_tool_calls, color=TABLEAU_COLORS[2])
    plt.ylabel('Average Number of Tool Calls', fontsize=12)
    plt.xlabel('Command Tier', fontsize=12)
    plt.title('Average Tool Calls per Command Tier', fontsize=15, pad=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f'{yval:.1f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()
    print(f"Generated: {os.path.join(FIGURE_DIR, filename)}")

def plot_latency_distribution(tier_name, latencies, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=8, color=TABLEAU_COLORS[3], edgecolor='black')
    plt.xlabel('Latency (ms)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Latency Distribution for {tier_name} Commands', fontsize=15, pad=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()
    print(f"Generated: {os.path.join(FIGURE_DIR, filename)}")

# --- Main Execution ---
if __name__ == '__main__':
    tier_names = ['Tier 1 (Simple)', 'Tier 2 (Medium)', 'Tier 3 (Complex)']

    # Calculate metrics from mock data
    avg_latencies = [
        np.mean(tier1_latencies),
        np.mean(tier2_latencies),
        np.mean(tier3_latencies)
    ]
    std_latencies = [
        np.std(tier1_latencies),
        np.std(tier2_latencies),
        np.std(tier3_latencies)
    ]
    success_rates = [
        (np.sum(tier1_successes) / runs_per_tier) * 100,
        (np.sum(tier2_successes) / runs_per_tier) * 100,
        (np.sum(tier3_successes) / runs_per_tier) * 100
    ]
    avg_tool_calls = [
        np.mean(tier1_tool_calls),
        np.mean(tier2_tool_calls),
        np.mean(tier3_tool_calls)
    ]

    # Plotting
    plot_average_latency(tier_names, avg_latencies, std_latencies, "avg_latency_per_tier.png")
    plot_success_rate(tier_names, success_rates, "success_rate_per_tier.png")
    plot_average_tool_calls(tier_names, avg_tool_calls, "avg_tool_calls_per_tier.png")
    
    plot_latency_distribution('Tier 1 (Simple)', tier1_latencies, "latency_dist_tier1.png")
    plot_latency_distribution('Tier 2 (Medium)', tier2_latencies, "latency_dist_tier2.png")
    plot_latency_distribution('Tier 3 (Complex)', tier3_latencies, "latency_dist_tier3.png")

    print(f"\nAll quantitative figures saved in '{FIGURE_DIR}' directory.") 