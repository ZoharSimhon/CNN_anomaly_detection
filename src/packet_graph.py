import networkx as nx
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

from config import MAX_PACKETS_PER_FLOW, PIC_GRAPH_DIR

def create_traffic_graph(flow_packets):
    """
    Creates a graph from a list of packets based on the TIG definition.
    - Nodes: packets with feature = packet_length * direction
    - Intra-burst edges: between consecutive packets in the same direction
    - Inter-burst edges: between first and last nodes of consecutive bursts
    """
    G = nx.DiGraph()
    n = len(flow_packets)

    # Step 1: Add nodes
    for idx, signed_len, timestamp in flow_packets:
        G.add_node(idx, length=signed_len, timestamp=timestamp)

    # Step 2: Identify bursts
    bursts = []
    start = 0
    current_dir = flow_packets[0][1] > 0 # True for downstream, False for upstream

    for i in range(1, n):
        _, pkt_len, _ = flow_packets[i]
        direction = pkt_len > 0
        if direction != current_dir:
            bursts.append((start, i - 1)) # end current burst
            start = i # start of new burst
            current_dir = direction
    bursts.append((start, n - 1)) # end last burst

    # Step 3: Intra-burst edges (sequential within burst)
    for start_idx, end_idx in bursts:
        for u in range(start_idx, end_idx):
            t1 = G.nodes[u]['timestamp']
            t2 = G.nodes[u + 1]['timestamp']
            delta = (t2 - t1).total_seconds()
            # aadd edges between consecutive nodes in burst
            G.add_edge(u, u + 1, type='intra', weight=delta)

    # Step 4: Inter-burst edges (between first and last nodes of consecutive bursts)
    for i in range(1, len(bursts)):
        prev_start, prev_end = bursts[i - 1]
        curr_start, curr_end = bursts[i]

        # Add edge from first node of previous burst to first node of current burst
        t1 = G.nodes[prev_start]['timestamp']
        t2 = G.nodes[curr_start]['timestamp']
        G.add_edge(prev_start, curr_start, type='inter', weight=(t2 - t1).total_seconds())

        # Add edge from last node of previous burst to last node of current burst
        t1 = G.nodes[prev_end]['timestamp']
        t2 = G.nodes[curr_end]['timestamp']
        G.add_edge(prev_end, curr_end, type='inter', weight=(t2 - t1).total_seconds())

    return G

def plot_and_save_graph(G, flow_key):
    """
    Plots the graph G and saves it as an image.
    - flow_key: used for filename uniqueness.
    """
    # Ensure output directory exists
    os.makedirs(PIC_GRAPH_DIR, exist_ok=True) 

    # Create filename using flow key
    flow_id_str = "_".join(map(str, flow_key))
    filepath = os.path.join(PIC_GRAPH_DIR, f"{flow_id_str}.png")
    
    # Layout and plot
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # consistent layout

    # Draw nodes
    node_colors = ['#76c893' if G.nodes[n]['length'] > 0 else '#f28482' for n in G.nodes]
    print([G.nodes[n]['length'] for n in G.nodes])
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)

    # Draw edges
    edge_colors = ['#3e64ff' if G[u][v]['type'] == 'intra' else '#ff922b' for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True)

    # Draw labels
    labels = {n: f"{d['length']}" for n,d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    # Draw edge weights
    edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}s" for u, v in G.edges if 'weight' in G[u][v]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title(f"Traffic Interaction Graph: {flow_key}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Graph saved to: {filepath}")

def graph_to_rgb_image(G):
    """
    Converts a packet graph into a fixed-size RGB image.
    
    Each pixel (u, v) is:
        - R: signed_length(u)
        - G: edge weight (time delta) between u and v
        - B: signed_length(v)
    If no edge exists, the pixel is (0, 0, 0).

    Args:
        G: NetworkX DiGraph with packet nodes and intra/inter edges.
        max_packets: Fixed number of packets per flow (image size).

    Returns:
        image: NumPy array of shape (max_packets, max_packets, 3), dtype=uint8
    """
    image = np.zeros((MAX_PACKETS_PER_FLOW, MAX_PACKETS_PER_FLOW, 3), dtype=np.float32)
    
        # Collect all signed lengths and weights to normalize
    signed_lengths = [G.nodes[n]['length'] for n in G.nodes]
    edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]

    # Handle edge cases
    if not signed_lengths or not edge_weights:
        return image.astype(np.uint8)

    len_min, len_max = min(signed_lengths), max(signed_lengths)
    w_min, w_max = min(edge_weights), max(edge_weights)

    def normalize(val, vmin, vmax):
        if vmax == vmin:
            return 127.5  # Mid-range value if constant
        return (val - vmin) / (vmax - vmin) * 255

    # Fill image
    for u, v, data in G.edges(data=True):
        if u < MAX_PACKETS_PER_FLOW and v < MAX_PACKETS_PER_FLOW:
            u_len = G.nodes[u]['length']
            v_len = G.nodes[v]['length']
            weight = data['weight']

            r = normalize(u_len, len_min, len_max)
            g = normalize(weight, w_min, w_max)
            b = normalize(v_len, len_min, len_max)

            image[u, v] = (r, g, b)

    return image.astype(np.uint8)