# Directory structure
PCAP_PATH = '../data/cic-mal-anal/Ransomware/Pletor/08_31_2017-ra-pletor-alibaba-a2603254188da3d67e4da5452e0304a9.pcap' # CHANGE
PIC_GRAPH_DIR = '../output/cic-mal-anal/graphs/'
MODEL_DIR = '../output/cic-mal-anal/model.pth'
TENSORS_DIR = '../output/cic-mal-anal/tensors' # Base path for npy files

# Subfolders
TRAIN_DIR = f'{TENSORS_DIR}/train'
TEST_BENIGN_DIR = f'{TENSORS_DIR}/test/ransomware/Pletor/benign' # CHANGE
TEST_MALICIOUS_DIR = f'{TENSORS_DIR}/test/ransomware/Pletor/malicious' # CHANGE

# Settings
MAX_PACKETS_PER_FLOW = 32  # P
IMAGE_SIZE = MAX_PACKETS_PER_FLOW  # Square image

# Model settings
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# OOD settings
T = 1.0
OOD_THRESHOLD = 0.1 # CHANGE

# Labels
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

# Malicious IPs
ATTACKER_IP = ['']
VICTIM_IP =  ['']