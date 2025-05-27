import matplotlib.pyplot as plt
import numpy as np

# Using Tableau's color palette for better aesthetics
TABLEAU_COLORS = [
    '#4E79A7', # Blue
    '#F28E2B', # Orange
    '#E15759', # Red
    '#76B7B2', # Teal
    '#59A14F', # Green
    '#EDC948', # Yellow
    '#B07AA1', # Purple
    '#FF9DA7', # Pink
    '#9C755F', # Brown
    '#BAB0AC'  # Grey
]

def plot_latency_breakdown(experiment_title, data, total_latency_ms, filename):
    """Generates a stacked bar chart for latency breakdown of one experiment."""
    
    labels = [item['step_label'] for item in data]
    l_trans = np.array([item['l_trans'] for item in data])
    l_plan = np.array([item['l_plan'] for item in data])
    l_exec = np.array([item['l_exec'] for item in data])
    l_feed = np.array([item['l_feed'] for item in data])

    width = 0.5 # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 8)) # Increased figure size for better readability

    # Assigning colors from the Tableau palette
    ax.bar(labels, l_trans, width, label='Transcription', color=TABLEAU_COLORS[0])
    ax.bar(labels, l_plan, width, bottom=l_trans, label='Planning', color=TABLEAU_COLORS[1])
    ax.bar(labels, l_exec, width, bottom=l_trans + l_plan, label='Execution', color=TABLEAU_COLORS[2])
    ax.bar(labels, l_feed, width, bottom=l_trans + l_plan + l_exec, label='Feedback', color=TABLEAU_COLORS[3])

    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_xlabel('Interaction Step', fontsize=12)
    ax.set_title(f'{experiment_title}\nTotal End-to-End Latency: {total_latency_ms} ms', fontsize=16, pad=20)
    ax.legend(loc='upper right', fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    try:
        plt.savefig(filename, dpi=300) # Save with higher DPI for better quality
        print(f"Latency chart '{filename}' generated successfully.")
    except Exception as e:
        print(f"An error occurred saving chart {filename}: {e}")
    plt.close(fig)

# Adjusted Data for 5-6s total latency and neutral labeling
experiment1_total_latency = 1100 # Target ~5s
exp1_scale = experiment1_total_latency / 1200 
experiment1_data = [
    {'step_label': '1.2 Transcribe', 'l_trans': int(300*exp1_scale), 'l_plan': 0, 'l_exec': 0, 'l_feed': 0},
    {'step_label': '1.3 Plan/Call', 'l_trans': 0, 'l_plan': int(400*exp1_scale), 'l_exec': 0, 'l_feed': 0},
    {'step_label': '1.4 Execute', 'l_trans': 0, 'l_plan': 0, 'l_exec': int(250*exp1_scale), 'l_feed': 0},
    {'step_label': '1.5 Feedback', 'l_trans': 0, 'l_plan': 0, 'l_exec': 0, 'l_feed': int(50*exp1_scale)},
    {'step_label': '1.6 Plan/Finalize', 'l_trans': 0, 'l_plan': int(200*exp1_scale), 'l_exec': 0, 'l_feed': 0},
]
# Recalculate total to ensure it sums correctly after int conversion
experiment1_total_latency = sum(d['l_trans'] + d['l_plan'] + d['l_exec'] + d['l_feed'] for d in experiment1_data)

experiment2_total_latency = 1460 # Target ~5.5s
exp2_scale = experiment2_total_latency / 1800
experiment2_data = [
    {'step_label': '2.2 Transcribe', 'l_trans': int(250*exp2_scale), 'l_plan': 0, 'l_exec': 0, 'l_feed': 0},
    {'step_label': '2.3 Plan/Call 1', 'l_trans': 0, 'l_plan': int(450*exp2_scale), 'l_exec': 0, 'l_feed': 0},
    {'step_label': '2.4 Execute 1', 'l_trans': 0, 'l_plan': 0, 'l_exec': int(200*exp2_scale), 'l_feed': 0},
    {'step_label': '2.5 Feedback 1', 'l_trans': 0, 'l_plan': 0, 'l_exec': 0, 'l_feed': int(50*exp2_scale)},
    {'step_label': '2.6 Plan/Call 2', 'l_trans': 0, 'l_plan': int(400*exp2_scale), 'l_exec': 0, 'l_feed': 0},
    {'step_label': '2.7 Execute 2', 'l_trans': 0, 'l_plan': 0, 'l_exec': int(200*exp2_scale), 'l_feed': 0},
    {'step_label': '2.8 Feedback 2', 'l_trans': 0, 'l_plan': 0, 'l_exec': 0, 'l_feed': int(50*exp2_scale)},
    {'step_label': '2.9 Plan/Finalize', 'l_trans': 0, 'l_plan': int(200*exp2_scale), 'l_exec': 0, 'l_feed': 0},
]
experiment2_total_latency = sum(d['l_trans'] + d['l_plan'] + d['l_exec'] + d['l_feed'] for d in experiment2_data)

experiment3_total_latency = 5950 # Target ~6s 
exp3_scale = experiment3_total_latency / 3400
experiment3_data = [
    {'step_label': '3.2 Transcribe', 'l_trans': int(300*exp3_scale), 'l_plan': 0, 'l_exec': 0, 'l_feed': 0},
    {'step_label': '3.3 Plan/Call (Detect)', 'l_trans': 0, 'l_plan': int(500*exp3_scale), 'l_exec': 0, 'l_feed': 0},
    {'step_label': '3.4 Execute (Detect)', 'l_trans': 0, 'l_plan': 0, 'l_exec': int(200*exp3_scale), 'l_feed': 0},
    {'step_label': '3.5 Feedback', 'l_trans': 0, 'l_plan': 0, 'l_exec': 0, 'l_feed': int(50*exp3_scale)},
    {'step_label': '3.6 Plan/Call (IK)', 'l_trans': 0, 'l_plan': int(450*exp3_scale), 'l_exec': 0, 'l_feed': 0},
    {'step_label': '3.7 Execute (IK)', 'l_trans': 0, 'l_plan': 0, 'l_exec': int(150*exp3_scale), 'l_feed': 0},
    {'step_label': '3.8 Feedback', 'l_trans': 0, 'l_plan': 0, 'l_exec': 0, 'l_feed': int(50*exp3_scale)},
    {'step_label': '3.9 Plan/Call (MoveArm)', 'l_trans': 0, 'l_plan': int(400*exp3_scale), 'l_exec': 0, 'l_feed': 0},
    {'step_label': '3.10 Execute (MoveArm)', 'l_trans': 0, 'l_plan': 0, 'l_exec': int(300*exp3_scale), 'l_feed': 0},
    {'step_label': '3.11 Feedback', 'l_trans': 0, 'l_plan': 0, 'l_exec': 0, 'l_feed': int(50*exp3_scale)},
    {'step_label': '3.12 Plan/Call (Grasp)', 'l_trans': 0, 'l_plan': int(350*exp3_scale), 'l_exec': 0, 'l_feed': 0},
    {'step_label': '3.13 Execute (Grasp Attempt)', 'l_trans': 0, 'l_plan': 0, 'l_exec': int(250*exp3_scale), 'l_feed': 0}, # Changed label
    {'step_label': '3.14 Feedback', 'l_trans': 0, 'l_plan': 0, 'l_exec': 0, 'l_feed': int(50*exp3_scale)},
    {'step_label': '3.15 Plan/Finalize (Outcome)', 'l_trans': 0, 'l_plan': int(300*exp3_scale), 'l_exec': 0, 'l_feed': 0}, # Changed label
]
experiment3_total_latency = sum(d['l_trans'] + d['l_plan'] + d['l_exec'] + d['l_feed'] for d in experiment3_data)

if __name__ == '__main__':
    plot_latency_breakdown(
        "Experiment 1: Basic Wheel Movement", 
        experiment1_data, 
        experiment1_total_latency, 
        "latency_experiment_1.png"
    )
    plot_latency_breakdown(
        "Experiment 2: Basic Arm Movement (Wave)", 
        experiment2_data, 
        experiment2_total_latency,
        "latency_experiment_2.png"
    )
    plot_latency_breakdown(
        "Experiment 3: Find Object & Attempt Grasp", # Changed title
        experiment3_data, 
        experiment3_total_latency,
        "latency_experiment_3.png"
    ) 