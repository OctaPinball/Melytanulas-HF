#Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

#Install Python3 and pip if not already installed
RUN apt-get update && apt-get install -y python3-pip

#Upgrade pip
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
RUN pip3 install --upgrade pip
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt

#Install PyTorch (already included in the base image, but you can add other required packages)
#RUN pip install torch torchvision torchaudio  # Not needed since PyTorch is already in the base image
#Set up a working directory
WORKDIR /workspace

#Default command to check if CUDA is available in Python
CMD ["python", "-u", "unet_pytorch.py"]