# Directory structure
PCAP_PATH = '../data/bruteforce-xss-2-23_02_2018.pcap'
PIC_GRAPH_DIR = '../output/graphs/'
MODEL_DIR = '../output/model1.pth'
TENSORS_DIR = '../output/tensors1' # Base path for npy files

# Subfolders
TRAIN_DIR = f'{TENSORS_DIR}/train'
TEST_BENIGN_DIR = f'{TENSORS_DIR}/test/benign'
TEST_MALICIOUS_DIR = f'{TENSORS_DIR}/test/malicious'

# Settings
MAX_PACKETS_PER_FLOW = 32  # P
IMAGE_SIZE = MAX_PACKETS_PER_FLOW  # Square image

# Model settings
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# OOD settings
T = 1.0
OOD_THRESHOLD = 0.5

# Labels
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

# Malicious IPs
ATTACKER_IP = '18.218.115.60'
VICTIM_IP = '172.31.69.28'