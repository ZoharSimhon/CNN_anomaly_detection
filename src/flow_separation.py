import pyshark
from collections import defaultdict
import numpy as np
from PIL import Image
import os

from packet_graph import create_traffic_graph, plot_and_save_graph, graph_to_rgb_image
from config import (
    MAX_PACKETS_PER_FLOW, 
    PCAP_PATH,
    TRAIN_BENIGN_DIR,
    TRAIN_MALICIOUS_DIR,
    TEST_BENIGN_DIR,
    TEST_MALICIOUS_DIR,
    ATTACKER_IP,
    VICTIM_IP,
)

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
        return None # skip malformed packets

def get_output_dir(mode, flow_key):
    if mode == 'train':
        label = flow_key[0] in [ATTACKER_IP, VICTIM_IP] and flow_key[2] in [ATTACKER_IP, VICTIM_IP]
        if label:
            return TRAIN_MALICIOUS_DIR
        else:
            return TRAIN_BENIGN_DIR
    if mode == 'test':
        label = flow_key[0] in [ATTACKER_IP, VICTIM_IP] and flow_key[2] in [ATTACKER_IP, VICTIM_IP]
        if label:
            return TEST_MALICIOUS_DIR
        else:
            return TEST_BENIGN_DIR
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
# def get_output_dir(mode, flow_key):
#     if mode == 'train':
#         return TRAIN_BENIGN_DIR
#     if mode == 'test':
#         label = flow_key[0] in [ATTACKER_IP, VICTIM_IP] and flow_key[2] in [ATTACKER_IP, VICTIM_IP]
#         if label:
#             return TEST_MALICIOUS_DIR
#         else:
#             return TEST_BENIGN_DIR
#     else:
#         raise ValueError(f"Unknown mode: {mode}")

def process_flows(mode='train'):
    """
    Process the PCAP file, extract flows, and create traffic graphs.
    - Each flow is limited to MAX_PACKETS_PER_FLOW packets.
    - Graphs are saved as images and tensors for CNN training.
    """

    # Dictionary to store flows and graphs
    flows = defaultdict(list)

    # Read PCAP packet-by-packet
    cap = pyshark.FileCapture(PCAP_PATH)

    for i, packet in enumerate(cap):
        if i > 50000:
            break
        if i % 10000 == 0 and i > 0:
            print(f'processed {i} packets')
                
        flow_key = get_flow_key(packet)
        if not flow_key:
            continue  # skip non-IP packets

        if len(flows[flow_key]) < MAX_PACKETS_PER_FLOW:
            flows[flow_key].append(packet)
            
        # Once flow reaches MAX_PACKETS_PER_FLOW → build graph
        # elif len(flows[flow_key]) == MAX_PACKETS_PER_FLOW and flow_key not in graphs:
        #     print(f"Creating graph for flow: {flow_key}")
        #     graph = create_traffic_graph(flows[flow_key], client_ip=flows[flow_key][0].ip.src)
        #     graphs.append(graph)
        #     plot_and_save_graph(graph, flow_key)
            
        #     # Convert to fixed-format RGB image
        #     image_tensor = graph_to_rgb_image(graph)

        #     # Save as .npy for CNN use
        #     output_dir = get_output_dir(mode, flow_key)
        #     np.save(f"{output_dir}/{'_'.join(map(str, flow_key))}.npy", image_tensor)
            
        #     # OPTIONAL: Save as an actual image too
        #     img = Image.fromarray(image_tensor)
        #     img.save(os.path.join(f"{output_dir}", f"{'_'.join(map(str, flow_key))}.png"))  
                
        #     # OPTIONAL: Remove flow from memory if not needed anymore
        #     del flows[flow_key]
            
    cap.close()
    
    # Process remaining flows that didn't reach MAX_PACKETS_PER_FLOW
    # print("Processing remaining flows...")
    # for flow_key in flows:
    #     if len(flows[flow_key]) < MAX_PACKETS_PER_FLOW:
    #         print(f"Creating graph for flow: {flow_key}")
    #         graph = create_traffic_graph(flows[flow_key], client_ip=flows[flow_key][0].ip.src)
    #         plot_and_save_graph(graph, flow_key)
            
    #         # Convert to fixed-format RGB image
    #         image_tensor = graph_to_rgb_image(graph)

    #         # Save as .npy for CNN use
    #         output_dir = get_output_dir(mode, flow_key)
    #         np.save(f"{output_dir}/{'_'.join(map(str, flow_key))}.npy", image_tensor)
    
    # Process all the flows
    print("Processing flows...")
    for flow_key in flows:
        if len(flows[flow_key]) < MAX_PACKETS_PER_FLOW:
            print(f"Creating graph for flow: {flow_key}")
            graph = create_traffic_graph(flows[flow_key], client_ip=flows[flow_key][0].ip.src)
            # plot_and_save_graph(graph, flow_key)
            
            # Convert to fixed-format RGB image
            image_tensor = graph_to_rgb_image(graph)

            # Save as .npy for CNN use
            output_dir = get_output_dir(mode, flow_key)
            np.save(f"{output_dir}/{'_'.join(map(str, flow_key))}.npy", image_tensor)
