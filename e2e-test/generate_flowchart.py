import graphviz

def generate_system_overview_flowchart():
    """Generates the system overview flowchart as a PNG image."""
    dot = graphviz.Digraph('System_Overview', comment='Perceive-Plan-Act Loop')
    dot.attr(rankdir='TD', label='Figure 1: System Overview - Perceive–Plan–Act Loop', fontsize='20')

    # Define node styles for better appearance if desired
    node_attrs = {'fontname': 'Helvetica', 'fontsize': '10', 'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'lightblue'}
    diamond_attrs = {**node_attrs, 'shape': 'diamond', 'fillcolor': 'lightyellow'}

    dot.node('A', 'User Speaks Command', **node_attrs)
    dot.node('B', 'Whisper Transcription', **diamond_attrs)
    dot.node('C', 'GPT-4o Planner', **diamond_attrs)
    dot.node('D', 'Executor Layer', **diamond_attrs)
    dot.node('E', 'Perception/Control Layer', **node_attrs) # Changed from Mocked Perception/Control
    dot.node('F', 'LLM Text Output (DONE:/ERROR:)', **node_attrs)
    dot.node('G', 'User Notified', **node_attrs)

    # Define edge styles
    edge_attrs = {'fontname': 'Helvetica', 'fontsize': '9'}

    dot.edge('A', 'B', **edge_attrs)
    dot.edge('B', 'C', **edge_attrs)
    dot.edge('C', 'D', label='Tool Call (JSON)', **edge_attrs)
    dot.edge('D', 'E', label='Function Call', **edge_attrs) # Changed from Mock Function Call
    dot.edge('E', 'D', label='Result', **edge_attrs)
    dot.edge('D', 'C', label='Tool Result (JSON)', **edge_attrs)
    dot.edge('C', 'C', label='Need More Steps?', **edge_attrs)
    dot.edge('C', 'F', **edge_attrs)
    dot.edge('F', 'G', **edge_attrs)

    try:
        dot.render('system_overview', format='png', cleanup=True)
        print("Flowchart 'system_overview.png' generated successfully.")
    except graphviz.backend.execute.ExecutableNotFound:
        print("ERROR: Graphviz executable not found. Please ensure Graphviz is installed and in your system PATH.")
    except Exception as e:
        print(f"An error occurred during flowchart generation: {e}")

if __name__ == '__main__':
    generate_system_overview_flowchart() 