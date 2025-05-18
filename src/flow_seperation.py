import pyshark
from collections import defaultdict
import numpy as np
from PIL import Image
import os

from packet_graph import create_traffic_graph, plot_and_save_graph, graph_to_rgb_image
from config import PCAP_PATH, MAX_PACKETS_PER_FLOW, TENSORS_DIR

def get_flow_key(pkt):
    try:
        src_ip = pkt.ip.src
        dst_ip = pkt.ip.dst 
        src_port = pkt[pkt.transport_layer].srcport
        dst_port = pkt[pkt.transport_layer].dstport
        protocol = pkt.transport_layer
        if int(src_port) > int(dst_port):
            return (src_ip, src_port, dst_ip, dst_port, protocol)
        else:
            return (dst_ip, dst_port, src_ip, src_port, protocol)
    except AttributeError as e:
        print(e)  # skip malformed packets
        return None 

# Dictionary to store flows and graphs
flows = defaultdict(list)
graphs = {}

# Read PCAP packet-by-packet
cap = pyshark.FileCapture(PCAP_PATH)

for packet in cap:
    flow_key = get_flow_key(packet)
    if not flow_key:
        continue  # skip non-IP packets

    print("iteration: ", len(flows[flow_key]), " flow_key: ", flow_key)
    if len(flows[flow_key]) < MAX_PACKETS_PER_FLOW:
        flows[flow_key].append(packet)
        
    # Once flow reaches MAX_PACKETS_PER_FLOW → build graph
    elif len(flows[flow_key]) == MAX_PACKETS_PER_FLOW and flow_key not in graphs:
        print(f"Creating graph for flow: {flow_key}")
        graph = create_traffic_graph(flows[flow_key], client_ip=flows[flow_key][0].ip.src)
        graphs[flow_key] = graph
        plot_and_save_graph(graph, flow_key)
        
        # Convert to fixed-format RGB image
        image_tensor = graph_to_rgb_image(graph)
        print("Image shape:", image_tensor.shape)  

        # Save as .npy for CNN use
        os.makedirs(TENSORS_DIR, exist_ok=True)
        np.save(f"../output/tensors/{'_'.join(map(str, flow_key))}.npy", image_tensor)
        
        # Optional: save as an actual image too
        img = Image.fromarray(image_tensor)
        img.save(os.path.join(TENSORS_DIR, f"{'_'.join(map(str, flow_key))}.png"))  
              
        # OPTIONAL: Remove flow from memory if not needed anymore
        # del flows[flow_key]
        
cap.close()
