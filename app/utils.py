import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io, base64

def compute_aac(df):
    """
    Compute Absolute Advantage Coefficient (AAC)
    df: dataframe with ['Source', 'Target', 'edge'] where edge = relation frequency
    """
    df = df.copy()
    df['edge'] = df['edge'].astype(float)
    count = df['edge'].sum()
    df['weight'] = df['edge'] / count

    # Compute node strength (sum of weights)
    node_strength = df.groupby('Source')['weight'].sum().add(df.groupby('Target')['weight'].sum(), fill_value=0)
    node_strength = node_strength.reset_index()
    node_strength.columns = ['node', 'AAC']

    return df, node_strength


def draw_network(df):
    """Draw simple weighted network and return as base64 PNG"""
    G = nx.from_pandas_edgelist(df, 'Source', 'Target', edge_attr='edge', create_using=nx.Graph())

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 5))
    edges = nx.draw_networkx_edges(G, pos, alpha=0.4)
    nodes = nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
    labels = nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
