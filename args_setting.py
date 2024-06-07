import argparse

parser = argparse.ArgumentParser(description='args_setting')
# Train args
parser.add_argument('--DEVICE', type=str, default='cuda:6')
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=256)

parser.add_argument('--task', type=str, default='CT-MRI')  # CT-MRI, PET-MRI, SPECT-MRI

args = parser.parse_args()
