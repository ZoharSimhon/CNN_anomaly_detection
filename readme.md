# I-GOOD: Image by Graph using Out-Of-Distribution

This project implements **I-GOOD**, a pipeline for anomaly detection in network traffic using Convolutional Neural Networks (CNNs) and Out-Of-Distribution (OOD) detection. It processes PCAP files, extracts flows, converts them into graph representations, and trains a CNN to distinguish between benign and malicious traffic.


## Features

- **Flow Extraction:** Parses PCAP files and separates network flows.
- **Graph Construction:** Builds Traffic Interaction Graphs (TIG) for each flow, where:
  - **Nodes:** Packets, with features: signed length (packet length × direction) and timestamp.
  - **Edges:** 
    - *Intra-burst:* Between consecutive packets in the same direction, weighted by time delta.
    - *Inter-burst:* Between bursts of different directions, weighted by time delta.
- **Graph-to-Image Conversion:** Converts each graph into a fixed-size RGB image for CNN input.
- **Dataset Preparation:** Organizes images into train/test splits for benign and malicious classes.
- **CNN Training:** Trains a simple CNN to classify flows as benign or malicious, handling class imbalance.
- **Energy Score for OOD:** Computes an energy score from the CNN output to detect Out-Of-Distribution (OOD) flows. The energy score helps to identify anomalous traffic that does not conform to the benign class distribution.
- **Evaluation:** Supports testing and evaluation on new flows.

## Directory Structure

```
CNN_anomaly_detection/
├── src/
│   ├── main.py
│   ├── flow_separation.py
│   ├── flow_separation_csv.py
│   ├── packet_graph.py
│   ├── train.py
│   ├── cnn_model.py
│   ├── dataset.py
│   ├── config.py
├── data/
│   ├── input.pcap
│   ├── train/
│   ├── test_benign/
│   ├── test_malicious/
├── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include: `pyshark`, `numpy`, `torch`, `networkx`, `matplotlib`, `Pillow`.

1. **Install TShark:**  
   Download and install [Wireshark/TShark](https://www.wireshark.org/download.html).  
   Make sure `tshark` is in your system PATH.

1. **Configure parameters:**  
   Edit `src/config.py` to set paths, constants, and IP addresses for attacker/victim.

## Usage

### 1. Train Mode

Extract benign flows, convert them to graph images, and train the model:
```bash
python src/main.py --train --preprocess
```

### 2. Test Mode

Extract mixed flows (benign and malicious), convert them to graph images, and test/evaluate the model:
```bash
python src/main.py --test --preprocess
```

## Model Details

- **Input:** RGB images representing traffic graphs.
- **Architecture:** Simple CNN (see `cnn_model.py`).
- **Loss Function:** Binary cross-entropy with logits, using energy score to handle imbalance.
