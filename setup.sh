# Download InceptionV3 checkpoint for FID and IS metrics
wget "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"

# Download CIFAR10 dataset
mkdir CIFAR10
cd CIFAR10
wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
tar -xf cifar-10-python.tar.gz

# Create directories
mkdir 50000_samples
mkdir samples
mkdir samples_ema