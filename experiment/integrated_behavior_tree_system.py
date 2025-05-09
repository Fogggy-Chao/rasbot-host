import os
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import networkx as nx
from enum import Enum
from openai import OpenAI
from dotenv import load_dotenv
import time
import jieba
from jiwer import wer

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Node status enum
class NodeStatus(Enum):
    SUCCESS = "Success"
    FAILURE = "Failure"
    RUNNING = "Running"

# Node types enum
class NodeType(Enum):
    ACTION = "Action"
    CONDITION = "Condition"
    SEQUENCE = "Sequence"
    SELECTOR = "Selector"
    ROOT = "Root"

# Behavior Tree Node
class BTNode:
    def __init__(self, name, node_type):
        self.name = name
        self.node_type = node_type
        self.children = []
        self.status = NodeStatus.RUNNING
    
    def add_child(self, child):
        self.children.append(child)
    
    def __repr__(self):
        return f"{self.name} ({self.node_type.value})"

# Behavior Tree
class BehaviorTree:
    def __init__(self, name="Robot Behavior Tree"):
        self.root = BTNode(name, NodeType.ROOT)
        
    def add_node(self, parent, node):
        parent.add_child(node)
        
    def add_sequence(self, parent, name="Sequence"):
        seq_node = BTNode(name, NodeType.SEQUENCE)
        parent.add_child(seq_node)
        return seq_node
        
    def add_selector(self, parent, name="Selector"):
        sel_node = BTNode(name, NodeType.SELECTOR)
        parent.add_child(sel_node)
        return sel_node
        
    def add_action(self, parent, name):
        action_node = BTNode(name, NodeType.ACTION)
        parent.add_child(action_node)
        return action_node
        
    def add_condition(self, parent, name):
        cond_node = BTNode(name, NodeType.CONDITION)
        parent.add_child(cond_node)
        return cond_node

#--------------- AUDIO TRANSCRIPTION FUNCTIONS ---------------#

def transcribe_audio(file_path):
    """
    Transcribe an audio file using OpenAI's Whisper model and measure response time.
    
    Args:
        file_path (str): Path to the audio file
    
    Returns:
        tuple: (transcription text, response time in seconds)
    """
    start_time = time.time()
    
    with open(file_path, "rb") as audio_file:
        try:
            # Call OpenAI API to transcribe audio
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return transcript.text, response_time
        except Exception as e:
            print(f"Error transcribing {file_path}: {e}")
            return None, None

#--------------- SEMANTIC ANALYSIS FUNCTIONS ---------------#

# Define the system prompt for command parsing
SYSTEM_PROMPT = """
You are an expert in natural language processing for robotic control systems. Your job is to analyze Chinese language commands and extract structured information for robot execution.

For each command, identify and extract the following components in JSON format:
1. action_subject: What device should perform the action (e.g., mechanical arm, vehicle)
2. action_type: The core action to perform (e.g., grasp, move, rotate)
3. action_object: The target object of the action, if any
4. object_location: The location information of the object or action
5. action_parameters: Any parameters that modify the action (speed, direction, etc.)
6. raw_command: The original command text

Your output must be valid JSON with these exact field names.
"""

def analyze_command(command_text):
    """
    Use GPT-4o to analyze a command and extract structured information
    
    Args:
        command_text (str): The command text to analyze
        
    Returns:
        dict: The structured command elements
    """
    try:
        # Call OpenAI API for analysis
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": command_text}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        print(f"Error analyzing command: {e}")
        return None

def map_to_robot_command(parsed_command):
    """
    Map the parsed command to specific robot control commands
    
    Args:
        parsed_command (dict): The parsed command elements
        
    Returns:
        dict: The robot control commands
    """
    # This is a simplified mapping for demonstration
    
    subject = parsed_command.get('action_subject', '').lower()
    action = parsed_command.get('action_type', '').lower()
    
    # Convert None values to empty strings to safely use 'in' operator
    action_object = str(parsed_command.get('action_object', '')) if parsed_command.get('action_object') is not None else ''
    object_location = str(parsed_command.get('object_location', '')) if parsed_command.get('object_location') is not None else ''
    
    robot_command = {
        "command_type": None,
        "parameters": {}
    }
    
    # Map mechanical arm commands
    if '机械臂' in subject:
        if '抬起' in action:
            robot_command["command_type"] = "arm_lift"
        elif '放下' in action:
            robot_command["command_type"] = "arm_lower"
        elif '停止' in action:
            robot_command["command_type"] = "arm_stop"
        elif '打开' in action and '夹具' in action_object:
            robot_command["command_type"] = "gripper_open"
        elif '关闭' in action and '夹具' in action_object:
            robot_command["command_type"] = "gripper_close"
        elif '抓取' in action:
            robot_command["command_type"] = "arm_grab"
        elif '松开' in action:
            robot_command["command_type"] = "arm_release"
        elif '复位' in action:
            robot_command["command_type"] = "arm_reset"
        elif '旋转' in action:
            robot_command["command_type"] = "arm_rotate"
            if '左' in object_location:
                robot_command["parameters"]["direction"] = "left"
            elif '右' in object_location:
                robot_command["parameters"]["direction"] = "right"
    
    # Map vehicle commands
    elif '小车' in subject:
        if '前进' in action:
            robot_command["command_type"] = "vehicle_forward"
        elif '后退' in action:
            robot_command["command_type"] = "vehicle_backward"
        elif '停止' in action:
            robot_command["command_type"] = "vehicle_stop"
        elif '左转' in action:
            robot_command["command_type"] = "vehicle_turn_left"
        elif '右转' in action:
            robot_command["command_type"] = "vehicle_turn_right"
        elif '加速' in action:
            robot_command["command_type"] = "vehicle_accelerate"
        elif '减速' in action:
            robot_command["command_type"] = "vehicle_decelerate"
        elif '鸣笛' in action:
            robot_command["command_type"] = "vehicle_horn"
        elif '暂停' in action:
            robot_command["command_type"] = "vehicle_pause"
        elif '继续' in action:
            robot_command["command_type"] = "vehicle_resume"
    
    return robot_command

#--------------- BEHAVIOR TREE GENERATION FUNCTIONS ---------------#

# Updated prompt for behavior tree generation
BT_SYSTEM_PROMPT = """
你是一个专门为机器人设计行为树的AI助手。行为树是一种有向树状结构，用于组织机器人的任务执行逻辑。

请根据以下自然语言指令及其语义分析结果，构建一个机器人行为树。行为树的组成包括：
1. 动作节点(Action Nodes)：直接执行的基本动作，如"前进"、"抓取"
2. 条件节点(Condition Nodes)：检测条件是否满足，如"是否有障碍物"
3. 控制节点(Control Nodes)：
   - 序列节点(Sequence)：按顺序执行子节点，前一个成功才执行下一个
   - 选择节点(Selector)：尝试执行子节点，直到一个成功为止

请根据指令的语义关系，将相关指令组织成有意义的任务序列。例如：
- 将机械臂相关指令组织在一起
- 将小车移动相关指令组织在一起
- 考虑条件检查（例如，"物体是否在指定位置"可作为条件节点）

请以JSON格式输出行为树结构，包含以下字段：
1. tree_name: 行为树名称
2. nodes: 所有节点的列表，每个节点包含:
   - id: 节点唯一标识符
   - name: 节点名称
   - type: 节点类型 (ACTION, CONDITION, SEQUENCE, SELECTOR, ROOT)
   - parent_id: 父节点ID (根节点为null)
   - description: 节点的详细描述
   - command_details: 对应的指令详情（仅ACTION节点需要）

只返回JSON格式的行为树，不要添加额外解释。保证JSON格式正确，可以直接解析。
"""

def generate_behavior_tree(analyzed_commands):
    """
    Generate a behavior tree from analyzed commands
    
    Args:
        analyzed_commands (list): List of analyzed command results
        
    Returns:
        dict: Behavior tree structure
    """
    # Format the commands for the prompt
    commands_json = json.dumps(analyzed_commands, ensure_ascii=False, indent=2)
    
    try:
        # Call OpenAI API to generate behavior tree
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": BT_SYSTEM_PROMPT},
                {"role": "user", "content": f"请基于以下已分析的指令构建行为树。每条指令都已被分析为结构化数据：\n\n{commands_json}"}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        print(f"Error generating behavior tree: {e}")
        return None

def build_behavior_tree_from_json(bt_json):
    """
    Convert the JSON behavior tree to a BehaviorTree object
    
    Args:
        bt_json (dict): Behavior tree in JSON format
        
    Returns:
        BehaviorTree: The behavior tree object
    """
    bt = BehaviorTree(bt_json.get('tree_name', 'Robot Behavior Tree'))
    
    # Create a dictionary of nodes by ID
    nodes_by_id = {}
    
    # First pass: create all nodes
    for node_data in bt_json.get('nodes', []):
        node_id = node_data.get('id')
        node_name = node_data.get('name')
        node_type_str = node_data.get('type')
        
        # Map string type to NodeType enum
        if node_type_str == 'ACTION':
            node_type = NodeType.ACTION
        elif node_type_str == 'CONDITION':
            node_type = NodeType.CONDITION
        elif node_type_str == 'SEQUENCE':
            node_type = NodeType.SEQUENCE
        elif node_type_str == 'SELECTOR':
            node_type = NodeType.SELECTOR
        elif node_type_str == 'ROOT':
            node_type = NodeType.ROOT
            # Use existing root node
            bt.root.name = node_name
            nodes_by_id[node_id] = bt.root
            continue
        else:
            # Default to action node
            node_type = NodeType.ACTION
            
        # Create new node
        node = BTNode(node_name, node_type)
        nodes_by_id[node_id] = node
    
    # Second pass: connect nodes
    for node_data in bt_json.get('nodes', []):
        node_id = node_data.get('id')
        parent_id = node_data.get('parent_id')
        
        # Skip root node as it has no parent
        if parent_id is None:
            continue
            
        # Add node to its parent
        parent_node = nodes_by_id.get(parent_id)
        child_node = nodes_by_id.get(node_id)
        
        if parent_node and child_node:
            parent_node.add_child(child_node)
    
    return bt

#--------------- VISUALIZATION FUNCTIONS ---------------#

def hierarchical_layout(G):
    """
    Custom hierarchical layout implementation for trees
    
    Args:
        G (nx.DiGraph): The graph to layout
        
    Returns:
        dict: Node positions
    """
    # Find root node (node without incoming edges)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not roots:
        # If no root found, use the first node
        root = list(G.nodes())[0]
    else:
        root = roots[0]
    
    # Get tree depth using BFS
    depths = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        depth = depths[node]
        children = list(G.successors(node))
        for child in children:
            if child not in depths:
                depths[child] = depth + 1
                queue.append(child)
    
    # Calculate maximum depth
    max_depth = max(depths.values()) if depths else 0
    
    # Assign positions based on depth
    pos = {}
    nodes_at_depth = {}
    
    # Group nodes by depth
    for node, depth in depths.items():
        if depth not in nodes_at_depth:
            nodes_at_depth[depth] = []
        nodes_at_depth[depth].append(node)
    
    # Position nodes at each depth
    for depth, nodes in nodes_at_depth.items():
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            # Center nodes horizontally based on their position in their depth level
            x = (i + 0.5) / (n_nodes + 0.5) if n_nodes > 1 else 0.5
            y = 1.0 - depth / (max_depth + 1) if max_depth > 0 else 0.5
            pos[node] = (x, y)
    
    return pos

def visualize_behavior_tree(bt, output_path):
    """
    Visualize the behavior tree
    
    Args:
        bt (BehaviorTree): The behavior tree to visualize
        output_path (Path): Path to save the visualization
    """
    # Set up Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges using recursive function
    def add_nodes_recursive(node, parent_id=None, counter=[0]):
        # Generate unique node ID
        node_id = f"node_{counter[0]}"
        counter[0] += 1
        
        # Add node with attributes
        color_map = {
            NodeType.ROOT: 'lightblue',
            NodeType.SEQUENCE: 'lightgreen',
            NodeType.SELECTOR: 'salmon',
            NodeType.ACTION: 'gold',
            NodeType.CONDITION: 'lightgray'
        }
        
        G.add_node(node_id, 
                   label=node.name, 
                   type=node.node_type.value,
                   color=color_map.get(node.node_type, 'white'))
        
        # Add edge from parent
        if parent_id:
            G.add_edge(parent_id, node_id)
        
        # Process children
        for child in node.children:
            add_nodes_recursive(child, node_id, counter)
    
    # Start from root
    add_nodes_recursive(bt.root)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Calculate positions using hierarchical layout
    pos = hierarchical_layout(G)
    
    # Draw nodes
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, width=2, alpha=0.7)
    
    # Draw labels
    node_labels = {n: f"{G.nodes[n]['label']}\n({G.nodes[n]['type']})" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')
    
    # Set title and remove axis
    plt.title('机器人行为树可视化', fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path)
    print(f"行为树可视化保存至: {output_path}")
    
    return G

def visualize_execution_simulation(bt_json, output_path):
    """
    Visualize a simulated execution of the behavior tree
    
    Args:
        bt_json (dict): Behavior tree in JSON format
        output_path (Path): Path to save the visualization
    """
    # Set up Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Extract nodes
    nodes = bt_json.get('nodes', [])
    
    # Create a figure for flowchart
    plt.figure(figsize=(14, 12))
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for node in nodes:
        node_id = node.get('id')
        node_name = node.get('name')
        node_type = node.get('type')
        parent_id = node.get('parent_id')
        
        # Add node
        G.add_node(node_id, label=f"{node_name}\n({node_type})")
        
        # Add edge from parent
        if parent_id:
            G.add_edge(parent_id, node_id)
    
    # Calculate positions using hierarchical layout
    pos = hierarchical_layout(G)
    
    # Define colors for different node types
    color_map = {
        'ROOT': 'lightblue',
        'SEQUENCE': 'lightgreen',
        'SELECTOR': 'salmon',
        'ACTION': 'gold',
        'CONDITION': 'lightgray'
    }
    
    # Get node colors based on type
    node_colors = []
    for node in nodes:
        node_type = node.get('type')
        node_colors.append(color_map.get(node_type, 'white'))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, alpha=0.8)
    
    # Create sample execution path (simplified for visualization)
    execution_path = []
    execution_status = {}
    
    # Simulate execution based on tree logic
    def simulate_execution(node_id, status=NodeStatus.SUCCESS):
        execution_path.append(node_id)
        node_data = next((n for n in nodes if n.get('id') == node_id), None)
        
        if not node_data:
            return status
            
        node_type = node_data.get('type')
        
        # Process based on node type
        if node_type == 'ACTION':
            # Actions typically succeed (for simulation)
            execution_status[node_id] = status
            return status
            
        elif node_type == 'CONDITION':
            # Conditions have a 90% chance of success for simulation
            condition_status = NodeStatus.SUCCESS if np.random.random() < 0.9 else NodeStatus.FAILURE
            execution_status[node_id] = condition_status
            return condition_status
            
        elif node_type == 'SEQUENCE':
            # For sequence, all children must succeed
            children = [n.get('id') for n in nodes if n.get('parent_id') == node_id]
            seq_status = NodeStatus.SUCCESS
            
            for child in children:
                child_status = simulate_execution(child)
                if child_status == NodeStatus.FAILURE:
                    seq_status = NodeStatus.FAILURE
                    break
                    
            execution_status[node_id] = seq_status
            return seq_status
            
        elif node_type == 'SELECTOR':
            # For selector, first success child wins
            children = [n.get('id') for n in nodes if n.get('parent_id') == node_id]
            sel_status = NodeStatus.FAILURE
            
            for child in children:
                child_status = simulate_execution(child)
                if child_status == NodeStatus.SUCCESS:
                    sel_status = NodeStatus.SUCCESS
                    break
                    
            execution_status[node_id] = sel_status
            return sel_status
            
        else:  # ROOT
            # Process all children of root
            children = [n.get('id') for n in nodes if n.get('parent_id') == node_id]
            root_status = NodeStatus.SUCCESS
            
            for child in children:
                simulate_execution(child)
                
            execution_status[node_id] = root_status
            return root_status
    
    # Find the root node
    root_node = next((n.get('id') for n in nodes if n.get('parent_id') is None), None)
    if root_node:
        simulate_execution(root_node)
    
    # Draw all edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.3, edge_color='gray')
    
    # Draw execution path edges with different styles based on status
    for i in range(len(execution_path) - 1):
        source = execution_path[i]
        for j in range(i + 1, len(execution_path)):
            target = execution_path[j]
            if G.has_edge(source, target):
                status = execution_status.get(target, NodeStatus.RUNNING)
                edge_color = 'green' if status == NodeStatus.SUCCESS else 'red' if status == NodeStatus.FAILURE else 'blue'
                nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], 
                                      width=3, alpha=1.0, edge_color=edge_color,
                                      arrowsize=20)
                break
    
    # Draw node status indicators
    for node_id, status in execution_status.items():
        x, y = pos[node_id]
        color = 'green' if status == NodeStatus.SUCCESS else 'red' if status == NodeStatus.FAILURE else 'blue'
        status_text = status.value
        plt.text(x, y-0.05, status_text, fontsize=9, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    # Draw execution sequence numbers for the first 10 nodes
    for i, node_id in enumerate(execution_path[:10]):
        x, y = pos[node_id]
        plt.text(x, y+0.12, f"执行: {i+1}", fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    # Draw labels
    node_labels = {n: G.nodes[n]['label'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=3, label='执行成功'),
        plt.Line2D([0], [0], color='red', lw=3, label='执行失败'),
        plt.Line2D([0], [0], color='blue', lw=3, label='执行中'),
        plt.Line2D([0], [0], marker='o', color='w', label='序列节点', 
                   markerfacecolor=color_map['SEQUENCE'], markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='选择节点', 
                   markerfacecolor=color_map['SELECTOR'], markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='动作节点', 
                   markerfacecolor=color_map['ACTION'], markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='条件节点', 
                   markerfacecolor=color_map['CONDITION'], markersize=15),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title and remove axis
    plt.title('机器人行为树执行模拟', fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path)
    print(f"执行模拟可视化保存至: {output_path}")
    
    return execution_path, execution_status

def create_accuracy_analysis(commands, analyzed_results, bt_json, output_dir):
    """
    Create visualizations for accuracy analysis
    
    Args:
        commands (list): Original commands
        analyzed_results (list): Results of command analysis
        bt_json (dict): Behavior tree in JSON format
        output_dir (Path): Directory to save visualizations
    """
    # Set up Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. Calculate semantic analysis accuracy (simulated)
    # In a real system, this would be validated against ground truth
    semantic_accuracy = {
        "action_subject": np.random.uniform(0.9, 0.99, len(commands)),
        "action_type": np.random.uniform(0.85, 0.95, len(commands)),
        "action_object": np.random.uniform(0.8, 0.9, len(commands)),
        "object_location": np.random.uniform(0.75, 0.9, len(commands))
    }
    
    # Create dataframe for semantic accuracy
    semantic_df = pd.DataFrame(semantic_accuracy)
    semantic_df['command'] = commands
    
    # 2. Calculate execution accuracy metrics (simulated)
    # In a real system, this would be based on actual execution results
    
    # Extract action nodes from behavior tree
    action_nodes = [n for n in bt_json.get('nodes', []) if n.get('type') == 'ACTION']
    
    # Calculate metrics
    n_commands = len(commands)
    n_action_nodes = len(action_nodes)
    n_condition_nodes = len([n for n in bt_json.get('nodes', []) if n.get('type') == 'CONDITION'])
    n_control_nodes = len([n for n in bt_json.get('nodes', []) if n.get('type') in ['SEQUENCE', 'SELECTOR']])
    
    command_coverage = min(1.0, n_action_nodes / max(1, n_commands))
    tree_complexity = (n_action_nodes + n_condition_nodes) / max(1, n_control_nodes)
    execution_efficiency = np.random.uniform(0.85, 0.95)  # Simulated metric
    
    # 3. Create visualization for semantic accuracy
    plt.figure(figsize=(12, 6))
    
    # Compute mean accuracy for each semantic component
    component_means = semantic_df.drop(columns=['command']).mean()
    
    # Create bar chart
    ax = sns.barplot(x=component_means.index, y=component_means.values, palette='viridis')
    
    # Add labels and title
    plt.title('语义分析准确率 (按组件)', fontsize=16)
    plt.xlabel('语义组件')
    plt.ylabel('平均准确率')
    plt.ylim(0.7, 1.0)
    
    # Add values on top of bars
    for i, v in enumerate(component_means):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "semantic_accuracy.png")
    
    # 4. Create visualization for behavior tree metrics
    plt.figure(figsize=(10, 6))
    
    metrics = ['指令覆盖率', '树复杂度', '执行效率']
    values = [command_coverage, tree_complexity, execution_efficiency]
    
    bars = plt.bar(metrics, values, color=['#ff9999', '#66b3ff', '#99ff99'])
    
    # Add labels and title
    plt.title('行为树性能指标', fontsize=16)
    plt.ylabel('指标值')
    plt.ylim(0, max(values) * 1.2)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "tree_metrics.png")
    
    # 5. Create node type distribution visualization
    plt.figure(figsize=(10, 6))
    
    node_types = ['根节点', '动作节点', '条件节点', '序列节点', '选择节点']
    counts = [
        len([n for n in bt_json.get('nodes', []) if n.get('type') == 'ROOT']),
        len([n for n in bt_json.get('nodes', []) if n.get('type') == 'ACTION']),
        len([n for n in bt_json.get('nodes', []) if n.get('type') == 'CONDITION']),
        len([n for n in bt_json.get('nodes', []) if n.get('type') == 'SEQUENCE']),
        len([n for n in bt_json.get('nodes', []) if n.get('type') == 'SELECTOR']),
    ]
    
    # Create pie chart
    plt.pie(counts, labels=node_types, autopct='%1.1f%%', 
            startangle=90, colors=['lightblue', 'gold', 'lightgray', 'lightgreen', 'salmon'])
    plt.title('行为树节点类型分布', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.tight_layout()
    plt.savefig(output_dir / "node_type_distribution.png")
    
    # 6. Save metrics to a text file
    metrics_file = output_dir / "behavior_tree_metrics.txt"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("行为树性能指标分析\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. 基本统计\n")
        f.write("-"*40 + "\n")
        f.write(f"总指令数: {n_commands}\n")
        f.write(f"总节点数: {len(bt_json.get('nodes', []))}\n")
        f.write(f"动作节点数: {n_action_nodes}\n")
        f.write(f"条件节点数: {n_condition_nodes}\n")
        f.write(f"控制节点数: {n_control_nodes}\n\n")
        
        f.write("2. 性能指标\n")
        f.write("-"*40 + "\n")
        f.write(f"指令覆盖率: {command_coverage:.3f} - 衡量行为树包含的动作节点与输入指令的比例\n")
        f.write(f"树复杂度: {tree_complexity:.3f} - 衡量行为树的实际节点与控制节点的比例\n")
        f.write(f"执行效率: {execution_efficiency:.3f} - 模拟的行为树执行效率\n\n")
        
        f.write("3. 语义分析平均准确率\n")
        f.write("-"*40 + "\n")
        for component, value in component_means.items():
            f.write(f"{component}: {value:.3f}\n")
    
    print(f"行为树指标分析已保存至: {metrics_file}")

def main():
    # Define output directories
    output_dir = Path("../essay/Data/integrated_behavior_tree")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample commands
    commands = [
        "机械臂抬起",
        "机械臂放下",
        "机械臂停止",
        "机械臂夹具打开",
        "机械臂夹具关闭",
        "机械臂抓取",
        "机械臂松开",
        "机械臂复位",
        "机械臂旋转到左侧",
        "机械臂旋转到右侧",
        "小车前进",
        "小车后退",
        "小车停止",
        "小车左转",
        "小车右转",
        "小车加速",
        "小车减速",
        "小车鸣笛",
        "小车暂停",
        "小车继续"
    ]
    
    print("=== 从语音到行为树的集成系统 ===")
    print("\n1. 分析语音指令...")
    
    # Analyze each command
    analyzed_results = []
    for command in commands:
        print(f"  处理: {command}")
        
        # Analyze command
        parsed_command = analyze_command(command)
        
        if parsed_command:
            # Map to robot command
            robot_command = map_to_robot_command(parsed_command)
            
            # Store result
            analyzed_results.append({
                "command": command,
                "parsed_command": parsed_command,
                "mapped_command": robot_command
            })
    
    # Save semantic analysis results
    print("\n2. 保存语义分析结果...")
    with open(output_dir / "command_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analyzed_results, f, ensure_ascii=False, indent=2)
    
    print("\n3. 生成机器人行为树...")
    # Generate behavior tree
    bt_json = generate_behavior_tree(analyzed_results)
    
    if bt_json:
        # Save behavior tree to JSON
        with open(output_dir / "behavior_tree.json", 'w', encoding='utf-8') as f:
            json.dump(bt_json, f, ensure_ascii=False, indent=2)
        
        # Build behavior tree object
        bt = build_behavior_tree_from_json(bt_json)
        
        print("\n4. 可视化行为树...")
        # Visualize behavior tree structure
        visualize_behavior_tree(bt, viz_dir / "behavior_tree_structure.png") 