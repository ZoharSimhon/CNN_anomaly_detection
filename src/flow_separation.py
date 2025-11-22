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
    TRAIN_OE_DIR,
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
    malicious_ips = ATTACKER_IP + VICTIM_IP
    if mode == 'train':
        label = flow_key[0] in malicious_ips and flow_key[2] in malicious_ips
        if label:
            return TRAIN_OE_DIR
        else:
            return TRAIN_BENIGN_DIR
    
    elif mode == 'test':
        label = flow_key[0] in malicious_ips and flow_key[2] in malicious_ips
        if label:
            return TEST_MALICIOUS_DIR
        else:
            return TEST_BENIGN_DIR
        
    else:
        raise ValueError(f"Unknown mode: {mode}")

def extract_packet_info(packet, client_ip=None, idx_packet=0):
    """
    Extracts signed length and timestamp from Pyshark packet.
    Args:
        - packet: a single packet from a flow
        - client_ip: optional, used to determine direction
        - idx_packet: index of the packet in the flow
    Returns:
        Tuple[int, int, datetime.datetime]: A tuple (index, signed_length, timestamp)
    """
    try:
        pkt_time = packet.sniff_time
        pkt_len = int(packet.length)
        direction = -1 if packet.ip.src == client_ip else 1
        signed_length = pkt_len * direction
        return (idx_packet, signed_length, pkt_time)
    except AttributeError as e:
        print(e)  # skip malformed packets
    return None

def process_flows(mode='train', num_packets=-1):
    """
    Process the PCAP file, extract flows, and create traffic graphs.
    - Each flow is limited to MAX_PACKETS_PER_FLOW packets.
    - Graphs are saved as images and tensors for CNN training.
    """

    # Dictionary to store flows
    flows = defaultdict(list)
    
    # Read PCAP packet-by-packet
    cap = pyshark.FileCapture(PCAP_PATH)

    for i, packet in enumerate(cap):
        if i == num_packets:
            break
        if i % 10000 == 0 and i > 0:
            print(f'processed {i} packets')
                
        flow_key = get_flow_key(packet)
        if not flow_key:
            continue  # skip non-IP packets
        
        # Skip if flow already reached EOF
        if flows[flow_key] == 'EOF':
            continue
        
        # If flow is already in memory, append packet
        if len(flows[flow_key]) < MAX_PACKETS_PER_FLOW:
            packet_info = extract_packet_info(packet, client_ip=flow_key[0], idx_packet=len(flows[flow_key]))
            flows[flow_key].append(packet_info)

        # Once flow reaches MAX_PACKETS_PER_FLOW → build graph
        if len(flows[flow_key]) == MAX_PACKETS_PER_FLOW:
            # print(f"Creating graph for flow: {flow_key}")
            graph = create_traffic_graph(flows[flow_key])
            # plot_and_save_graph(graph, flow_key)
            
            # Convert to fixed-format RGB image
            image_tensor = graph_to_rgb_image(graph)

            # Save as .npy for CNN use
            output_dir = get_output_dir(mode, flow_key)
            np.save(f"{output_dir}/{'_'.join(map(str, flow_key))}.npy", image_tensor)
            
            # OPTIONAL: Save as an actual image too
            # img = Image.fromarray(image_tensor)
            # img.save(os.path.join(f"{output_dir}", f"{'_'.join(map(str, flow_key))}.png"))  
            
            # Reset flow to EOF state    
            flows[flow_key] = 'EOF' 
                       
    cap.close()
    
    # Process remaining flows that didn't reach MAX_PACKETS_PER_FLOW
    print("Processing remaining flows...")
    for flow_key in flows:
        if flows[flow_key] != 'EOF':
            print(f"Creating graph for flow: {flow_key}")
            graph = create_traffic_graph(flows[flow_key])

            # plot_and_save_graph(graph, flow_key)
            
            # Convert to fixed-format RGB image
            image_tensor = graph_to_rgb_image(graph)

            # Save as .npy for CNN use
            output_dir = get_output_dir(mode, flow_key)
            np.save(f"{output_dir}/{'_'.join(map(str, flow_key))}.npy", image_tensor)
            
            # Reset flow to EOF state    
            flows[flow_key] = 'EOF' 
