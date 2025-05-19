# Directory structure
PCAP_PATH = '../data/benign_train.pcap'
PIC_GRAPH_DIR = '../output/graphs/'
MODEL_DIR = '../output/model.pth'
TENSORS_DIR = '../output/tensors/' # Base path for npy files

# Subfolders
TRAIN_BENIGN_DIR = f"{TENSORS_DIR}/train"
TEST_BENIGN_DIR = f"{TENSORS_DIR}/test/benign"
TEST_MALICIOUS_DIR = f"{TENSORS_DIR}/test/malicious"

# Settings
MAX_PACKETS_PER_FLOW = 32  # P
IMAGE_SIZE = MAX_PACKETS_PER_FLOW  # Square image

# Model settings
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Labels
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1

# Malicious IPs
ATTACKER_IP = '18.218.115.60'
VICTIM_IP = '172.31.69.28'