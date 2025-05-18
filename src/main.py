import os
import argparse

from flow_separation import process_flows
from train import train_model
from test import test_model
from config import TENSORS_DIR, TRAIN_BENIGN_DIR, TEST_BENIGN_DIR, TEST_MALICIOUS_DIR   


def ensure_dirs():
    os.makedirs(TENSORS_DIR, exist_ok=True)
    os.makedirs(TRAIN_BENIGN_DIR, exist_ok=True)
    os.makedirs(TEST_BENIGN_DIR, exist_ok=True)
    os.makedirs(TEST_MALICIOUS_DIR, exist_ok=True)

def main(args):
    print("🔧 Step 1: Ensuring required folders exist...")
    ensure_dirs()

    if args.preprocess:
        print("📦 Step 2: Processing flows and converting to image tensors...")
        process_flows()

    if args.train:
        print("🧠 Step 3: Training model on benign traffic...")
        train_model()

    if args.test:
        print("🔍 Step 4: Testing model on mixed traffic...")
        test_model()

    if not any([args.preprocess, args.train, args.test]):
        print("⚠️ No action specified. Use --preprocess, --train, or --test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Graph CNN Pipeline")
    parser.add_argument('--preprocess', action='store_true', help="Run flow processing and graph-to-image conversion")
    parser.add_argument('--train', action='store_true', help="Train CNN model on benign traffic")
    parser.add_argument('--test', action='store_true', help="Test model on mixed traffic")

    args = parser.parse_args()
    main(args)
