# Directory structure
PCAP_PATH = '../data/cic2018/bruteforce-xss-1-22_02_2018.pcap' # CHANGE
PIC_GRAPH_DIR = '../output/cic2018/semi-supervised/graphs/'
MODEL_DIR = '../output/cic2018/semi-supervised/model.pth'
TENSORS_DIR = '../output/cic2018/semi-supervised/tensors' # Base path for npy files

# Subfolders
TRAIN_BENIGN_DIR = f'{TENSORS_DIR}/train/benign'
TRAIN_OE_DIR = f'{TENSORS_DIR}/train/malicious'
TEST_BENIGN_DIR = f'{TENSORS_DIR}/test/bruteforceXSS1/benign' # CHANGE
TEST_MALICIOUS_DIR = f'{TENSORS_DIR}/test/bruteforceXSS1/malicious' # CHANGE
# Settings
MAX_PACKETS_PER_FLOW = 32  # P
IMAGE_SIZE = MAX_PACKETS_PER_FLOW  # Square image

# Model settings
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # Lower LR is better for Energy Fine-tuning

# Energy-Based OOD Settings (Liu et al. 2020)
T = 1.0
# Paper Eq. 6 Margins:
# We want Benign Energy < -5
# We want Attack Energy > -1
M_IN = -5.0 
M_OUT = -1.0 
OE_LAMBDA = 0.1  # Weight for the energy loss term (usually 0.1)
OOD_THRESHOLD = -3.0  # Energy threshold for classifying as Malicious - must be between M_IN and M_OUT.


# Labels
BENIGN_LABEL = 0
MALICIOUS_LABEL = 1 # This is for testing; during training we use this to identify OE data

# Malicious IPs
ATTACKER_IP = ['18.218.115.60'] # CHANGE
VICTIM_IP =  ['172.31.69.28'] # CHANGE