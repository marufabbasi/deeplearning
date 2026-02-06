"""
A minimal MNIST training tutorial that works on:
- 1 GPU (CUDA) if available
- otherwise CPU

This will work for CPU or GPU- but try to use a GPU if you have one.

What you'll learn:
- how to load MNIST with torchvision
- how DataLoader works
- how to define a simple neural net
- training + evaluation loops
- saving/loading a model

We have defined a set of common rules for training neural networks. 
Some of them are from Andrej Karpathy's tweet: https://x.com/karpathy/status/1013244313327681536?lang=en

TODO: You probably want to be able to write entire code from memory. *** VERY IMPORTANT ***.
You really should *NOT* try to memorize it, but do it so many times that you can write it from memory.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ----------------------------
# 1) Device: GPU if available
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# RULE: Device placement: model + data must be on same device.

print("Using device:", device)

# ----------------------------
# 2) Hyperparameters
# ----------------------------
batch_size = 64
# Number of samples per batch.
# RULE: Affects memory, gradient noise, and BatchNorm stability (if used).

learning_rate = 0.1
# Step size for optimizer updates.
# RULE: LR is #1 reason training diverges or stalls; 0.1 is high for SGD but can work here.
# TODO: research learning rate for various loss functions.
# Example optimizers: optim.SGD, optim.Adam, optim.RMSprop, optim.Adagrad, optim.AdamW, optim.Adadelta

epochs = 3
# Number of full passes over training data.
# RULE: More epochs generally improve accuracy until overfitting.
# TODO: research epochs for various real world scenarios.


# ----------------------------
# 3) Data transforms
# ----------------------------
# MNIST images are grayscale 28x28.
# transforms.ToTensor() converts PIL image [0..255] to float tensor [0..1]
transform = transforms.ToTensor()
# Converts input image to torch.float32 tensor scaled to [0,1], shape [1,28,28].
# RULE: Data scaling: stable training requires consistent scaling across train/test.
# RULE: Train/test preprocessing must match for consistent results.
# TODO: research data transform functions available in torchvision.transforms.
# Example transforms: transforms.ToTensor, transforms.Normalize, transforms.Resize, 
# transforms.RandomHorizontalFlip, transforms.RandomRotation

# ----------------------------
# 4) Download/load datasets
# RULE: Avoid data leakage: train and test must be different splits.
# ----------------------------
train_dataset = datasets.MNIST(
    root="./data", # Local path to store/download dataset files.
    train=True, # Selects training split.
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False, # Selects test split (held-out evaluation).
    # RULE: No tuning on test if you want unbiased measurement.
    download=True,
    transform=transform, # Same transform as train.
)


# ----------------------------
# 5) DataLoaders
# ----------------------------
# DataLoader does:
# - batching
# - shuffling (train)
# - parallel loading (num_workers) (keep 0 for maximum portability)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# Creates iterator that yields (images, labels) batches from training set.
# RULE: shuffle=True for train prevents learning from ordering artifacts.

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# Creates iterator for test set.
# RULE: shuffle=False for eval gives deterministic iteration; doesn't affect metrics but avoids confusion.

# TODO: research how to clean up data before traininging.
# TODO: research how to load various types of datasets with DataLoader.
# TODO: research num_workers for maximum portability.
#

# ----------------------------
# 6) Define a simple model
# ----------------------------
# We'll use a tiny MLP:
#   784 -> 256 -> 10
# Explanation:
# - Flatten 28x28 into 784 vector
# - Hidden layer gives capacity
# - Output layer gives 10 logits (one per digit)
class MLP(nn.Module):
    # Defines a neural network class (inherits from nn.Module).
    # RULE: Model must be an nn.Module for parameters to register and optimize.

    def __init__(self):
        super().__init__()
        # Initializes nn.Module base class internals (parameter tracking).
        # RULE: If you forget super().__init__(), parameters might not register correctly.

        self.net = nn.Sequential(
            # Sequential container: runs layers in order.

            nn.Linear(28 * 28, 256),
            # First linear layer maps 784 -> 256.
            # RULE: Shape correctness: input must be [B,784] after flatten.

            nn.ReLU(),
            # Nonlinearity; without it, stacked linear layers collapse into one linear layer.
            # RULE: Capacity: you need nonlinearities to learn non-linear decision boundaries.
            # TODO: research how to define a different types of activation functions - ReLU, GELU, Tanh, Sigmoid

            nn.Linear(256, 10),
            # Final linear layer outputs 10 numbers per sample.
            # RULE: Logits, not probabilities: do NOT apply softmax here if using CrossEntropyLoss.
            # TODO: research how to define a different types of loss functions - CrossEntropyLoss, MSELoss, L1Loss, L2Loss
        )

    def forward(self, x):
        # x shape: [B, 1, 28, 28]
        # forward() defines computation from inputs to outputs.

        x = x.view(x.size(0), -1)  # -> [B, 784]
        # Flattens each image into a vector.
        # RULE: Shape sanity: if you mess up flattening, model won't learn.
        # TODO: research how to define a different types of flattening functions - view, reshape, flatten

        return self.net(x)         # -> [B, 10] logits
        # Produces raw scores (logits) per class.
        # RULE: Loss compatibility: CrossEntropyLoss expects raw logits.        

# TODO: research how to define a Convolutional Neural Network (CNN)
# TODO: research how to define a Recurrent Neural Network (RNN)
# TODO: research how to define a Long Short-Term Memory (LSTM) network.
# TODO: research how to define a Gated Recurrent Unit (GRU) network.
# TODO: research how to define a Transformer network.

model = MLP().to(device)
# Instantiates the model and moves parameters to GPU/CPU.
# RULE: Device placement must match data device; otherwise runtime error or slow CPU fallback.


# ----------------------------
# 7) Loss and optimizer
# ----------------------------
# CrossEntropyLoss expects:
# - logits: [B, 10] (raw scores, NOT softmax)
# - labels: [B] (int class index 0..9)
criterion = nn.CrossEntropyLoss()
# Defines loss function for multi-class classification.
# RULE: "logits not softmax" and "labels are class indices" are common failure points.
# TODO: research how to define a different types of loss functions - CrossEntropyLoss, MSELoss, L1Loss, L2Loss

optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# Defines SGD optimizer operating on all model parameters.
# RULE: Optimizer must receive model.parameters() or nothing will update.
# TODO: research how to define a different types of optimizers - SGD, Adam, RMSprop, Adagrad, AdamW, Adadelta

# ----------------------------
# 8) Train for one epoch
# TODO: make sure you can write this loop without looking at the code- from your memory. VERY IMPORTANT.
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx):
    model.train()  # enables training behaviors (e.g., dropout, batchnorm updates)
    # Sets training mode.
    # RULE (Karpathy #2): Must toggle train() during training.

    running_loss = 0.0
    # Accumulates total loss (scaled by batch size) over the epoch.

    correct = 0
    total = 0
    # Tracking accuracy across epoch.

    for batch_idx, (images, labels) in enumerate(loader):
        # Each iteration yields one batch of images and labels from DataLoader.
        # RULE: Always verify shapes/dtypes here when debugging.

        # Move data to device (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)
        # Moves tensors to GPU/CPU.
        # RULE: Device mismatch is a top runtime error source.

        # 1) Forward pass
        logits = model(images)
        # Computes model outputs (logits) from images.
        # RULE: Forward should produce correct shape [B, num_classes].

        # 2) Compute loss
        loss = criterion(logits, labels)
        # Computes scalar loss for this batch.
        # RULE: Loss expects logits+labels with correct shapes/dtypes.

        # 3) Backward pass
        optimizer.zero_grad(set_to_none=True)
        # Clears accumulated gradients from previous step.
        # RULE (Karpathy #3): forgetting this accumulates grads and breaks training.
        # NOTE: set_to_none=True is faster and reduces memory; grads become None until computed.

        loss.backward()
        # Computes gradients for all parameters via autograd.
        # RULE: If loss is NaN/Inf, gradients will also be bad.

        # 4) Update parameters
        optimizer.step()
        # Applies gradient update to parameters.
        # RULE: If you forget step(), model never learns.

        # Stats
        running_loss += loss.item() * images.size(0)
        # loss.item() is Python float; multiply by batch size to sum total epoch loss.
        # RULE: Correct averaging: sum(batch_loss * batch_size) / total_samples.

        # Accuracy on this batch
        preds = logits.argmax(dim=1)  # [B]
        # Converts logits to predicted class index (max logit).
        # RULE: Metrics: argmax over class dimension.

        correct += (preds == labels).sum().item()
        # Counts correct predictions in this batch.
        # RULE: Ensure labels are class indices (not one-hot).

        total += labels.size(0)
        # Adds batch size to total sample count.

        # Print occasionally
        if batch_idx % 200 == 0:
            print(f"Epoch {epoch_idx} | Batch {batch_idx} | Loss {loss.item():.4f}")
            # Progress logging for debugging learning behavior.
            # RULE: If loss doesn't drop early, suspect LR, data, loss wiring, or model.

    avg_loss = running_loss / total
    # Computes mean loss per sample for epoch.
    # RULE: Proper averaging prevents misleading loss numbers.

    acc = correct / total
    # Epoch accuracy.
    return avg_loss, acc

# ----------------------------
# 9) Evaluate
# ----------------------------
@torch.no_grad()
# Disables gradient tracking inside function for speed + memory.
# RULE: Eval should not build computation graphs; prevents accidental backward.

def evaluate(model, loader, criterion, device):
    model.eval()  # disables training-only behaviors
    # Sets evaluation mode.
    # RULE (Karpathy #2): Must toggle eval() for correct dropout/batchnorm behavior.

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # Iterate over evaluation batches.

        images = images.to(device)
        labels = labels.to(device)
        # Move to same device as model.
        # RULE: Device consistency.

        logits = model(images)
        # Forward pass only (no gradients due to no_grad()).

        loss = criterion(logits, labels)
        # Compute eval loss.

        running_loss += loss.item() * images.size(0)
        # Sum loss over samples.

        preds = logits.argmax(dim=1)
        # Predicted classes.

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


# ----------------------------
# 10) Training loop
# ----------------------------
for epoch in range(1, epochs + 1):
    # Repeats training+evaluation for 'epochs' passes.

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
    # Trains one full epoch; returns training loss/acc.
    # RULE: Ensure model.train() is called inside training function.

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    # Evaluates on test set; returns loss/acc.
    # RULE: Ensure model.eval() + no_grad in evaluation.

    print(f"\nEpoch {epoch} summary")
    print(f"  Train: loss={train_loss:.4f} acc={train_acc*100:.2f}%")
    print(f"  Test : loss={test_loss:.4f}  acc={test_acc*100:.2f}%\n")
    # Prints epoch summary so you can see training progressing.
    # RULE: If train improves but test doesn't, you may be overfitting or have eval mismatch.


# ----------------------------
# 11) Save model
# ----------------------------
save_path = "mnist_mlp.pth"
# File path to store model weights.

torch.save(model.state_dict(), save_path)
# Saves only model parameters (recommended for PyTorch).
# RULE: Save/load: state_dict is portable across runs if model code matches.

print(f"Saved model weights to: {save_path}")
# TODO: look at the file and its size on disk and try to understand why it
# has that size

# ----------------------------
# 12) Load model (demo)
# ----------------------------
loaded_model = MLP().to(device)
# Creates a fresh model instance with same architecture.
# RULE: Must match architecture exactly to load weights.


loaded_model.load_state_dict(torch.load(save_path, map_location=device))
# Loads saved weights; map_location ensures CPU/GPU compatibility.
# RULE: Device portability: load GPU-saved weights on CPU safely.
# TODO: research how to estimate models memory footprint from GPU memory usage.
# TODO: try to change the model architecture and see if you can load the weights or what error you get.

loaded_model.eval()
# Put loaded model in eval mode for inference.
# RULE: User val mode for inference.

# Quick sanity check: evaluate loaded model
loaded_test_loss, loaded_test_acc = evaluate(loaded_model, test_loader, criterion, device)
# Re-runs evaluation to confirm loaded model behaves similarly.
# RULE: Sanity checks catch save/load mismatch or accidental training-mode inference.

print(f"Loaded model test acc: {loaded_test_acc*100:.2f}%")
