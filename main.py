# Medical Image Classification
# This code trains a neural network to classify medical images

# First, install the required packages
!python -c "import monai" || pip install -q "monai-weekly[pillow, tqdm]"
!python -c "import matplotlib" || pip install -q matplotlib
%matplotlib inline

# Import all the libraries 
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report

# MONAI is a library specifically for medical AI
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

# Check MONAI configuration
print_config()

# Step 1: Set up data directory
print("Setting up data directory...")
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(f"Data will be stored in: {root_dir}")

# Step 2: Download the medical dataset (MedNIST)
print("Downloading medical image dataset...")
resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
md5 = "0bc7306e7427e00ad1c5526a6677552d"

compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
data_dir = os.path.join(root_dir, "MedNIST")

# Only download if don't already have the data
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir, md5)
    print("Dataset downloaded successfully!")
else:
    print("Dataset already exists!")

# Make sure to get consistent results every time
set_determinism(seed=0)

# Step 3: Explore the dataset
print("Exploring the dataset...")
class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
num_classes = len(class_names)

# Get all image files for each class
image_files = []
for i in range(num_classes):
    class_path = os.path.join(data_dir, class_names[i])
    class_images = [os.path.join(class_path, x) for x in os.listdir(class_path)]
    image_files.append(class_images)

# Count how many images we have in each class
num_each_class = [len(image_files[i]) for i in range(num_classes)]

# Create a flat list of all images and their corresponding labels
all_image_files = []
all_labels = []
for i in range(num_classes):
    all_image_files.extend(image_files[i])
    all_labels.extend([i] * num_each_class[i])

total_images = len(all_labels)

# Get image dimensions by opening the first image
sample_image = PIL.Image.open(all_image_files[0])
image_width, image_height = sample_image.size

print(f"Total number of images: {total_images}")
print(f"Image size: {image_width} x {image_height} pixels")
print(f"Classes found: {class_names}")
print(f"Images per class: {num_each_class}")

# Step 4: Show some sample images
print("Showing sample images...")
plt.figure(figsize=(10, 10))
for i in range(9):  # Show 9 random images
    random_index = np.random.randint(total_images)
    image = PIL.Image.open(all_image_files[random_index])
    image_array = np.array(image)
    
    plt.subplot(3, 3, i + 1)
    plt.title(class_names[all_labels[random_index]])
    plt.imshow(image_array, cmap="gray", vmin=0, vmax=255)
    plt.axis('off')  # Hide axes for cleaner look

plt.tight_layout()
plt.show()

# Step 5: Split data into training, validation, and test sets
print("Splitting data into train/validation/test sets...")
validation_percent = 0.1  # 10% for validation
test_percent = 0.1        # 10% for testing
# Remaining 80% will be for training

total_length = len(all_image_files)
indices = np.arange(total_length)
np.random.shuffle(indices)  # Randomly shuffle the indices

# Calculate split points
test_split = int(test_percent * total_length)
val_split = int(validation_percent * total_length) + test_split

# Split the indices
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

# Create separate lists for each set
train_images = [all_image_files[i] for i in train_indices]
train_labels = [all_labels[i] for i in train_indices]
val_images = [all_image_files[i] for i in val_indices]
val_labels = [all_labels[i] for i in val_indices]
test_images = [all_image_files[i] for i in test_indices]
test_labels = [all_labels[i] for i in test_indices]

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")

# Step 6: Set up data transformations
print("Setting up data transformations...")

# For training data, apply augmentations to make the model more robust
train_transforms = Compose([
    LoadImage(image_only=True),           # Load the image
    EnsureChannelFirst(),                 # Make sure channels are in the right order
    ScaleIntensity(),                     # Normalize pixel values to 0-1 range
    RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),  # Randomly rotate images
    RandFlip(spatial_axis=0, prob=0.5),   # Randomly flip images horizontally
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),  # Randomly zoom in/out
])

# For validation and test data, only do basic preprocessing (no augmentation)
val_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity()
])

# Transformations for predictions and labels
y_pred_trans = Compose([Activations(softmax=True)])  # Convert outputs to probabilities
y_trans = Compose([AsDiscrete(to_onehot=num_classes)])  # Convert labels to one-hot encoding

# Step 7: Create a custom dataset class
class MedicalImageDataset(torch.utils.data.Dataset):
    """A simple dataset class for medical images"""
    
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load and transform the image, return with its label
        image = self.transforms(self.image_files[index])
        label = self.labels[index]
        return image, label

# Create datasets
print("Creating datasets...")
train_dataset = MedicalImageDataset(train_images, train_labels, train_transforms)
val_dataset = MedicalImageDataset(val_images, val_labels, val_transforms)
test_dataset = MedicalImageDataset(test_images, test_labels, val_transforms)

# Create data loaders (these feed data to the model in batches)
batch_size = 300  # Process 300 images at a time
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

# Step 8: Set up the model and training
print("Setting up the neural network model...")

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the model (DenseNet121 is a proven architecture for image classification)
model = DenseNet121(
    spatial_dims=2,           # 2D images
    in_channels=1,            # Grayscale images (1 channel)
    out_channels=num_classes  # Number of classes to predict
).to(device)

# Set up training components
loss_function = torch.nn.CrossEntropyLoss()  # Good for classification
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Adam optimizer with learning rate
auc_metric = ROCAUCMetric()  # Metric to evaluate performance

# Training settings
max_epochs = 4  # How many times to go through the entire dataset
val_interval = 1  # Validate after every epoch

# Variables to track the best model
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter()  # For logging training progress

# Step 9: Training loop
print("Starting training...")
print("This might take a while depending on your hardware!")

for epoch in range(max_epochs):
    print("-" * 50)
    print(f"Epoch {epoch + 1}/{max_epochs}")
    
    # Training phase
    model.train()  # Put model in training mode
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        
        # Get images and labels from the batch
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass: get predictions
        outputs = model(inputs)
        
        # Calculate loss
        loss = loss_function(outputs, labels)
        
        # Backward pass: calculate gradients
        loss.backward()
        
        # Update model weights
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Print progress
        total_steps = len(train_dataset) // batch_size
        if step % 10 == 0 or step == total_steps:  # Print every 10 steps
            print(f"Step {step}/{total_steps}, Loss: {loss.item():.4f}")
        
        # Log to tensorboard
        epoch_len = len(train_dataset) // batch_size
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    
    # Calculate average loss for the epoch
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    # Validation phase
    if (epoch + 1) % val_interval == 0:
        print("Evaluating on validation set...")
        model.eval()  # Put model in evaluation mode
        
        with torch.no_grad():  # Don't calculate gradients during validation
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            
            # Go through all validation batches
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                
                # Get predictions
                predictions = model(val_images)
                
                # Collect all predictions and labels
                y_pred = torch.cat([y_pred, predictions], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            
            # Calculate metrics
            y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
            
            # Calculate AUC
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            
            # Calculate accuracy
            correct_predictions = torch.eq(y_pred.argmax(dim=1), y)
            accuracy = correct_predictions.sum().item() / len(correct_predictions)
            
            metric_values.append(auc_result)
            
            # Save model if it's the best so far
            if auc_result > best_metric:
                best_metric = auc_result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(root_dir, "best_model.pth"))
                print("✓ Saved new best model!")
            
            print(f"Current AUC: {auc_result:.4f}")
            print(f"Current Accuracy: {accuracy:.4f}")
            print(f"Best AUC so far: {best_metric:.4f} (epoch {best_metric_epoch})")
            
            writer.add_scalar("val_accuracy", accuracy, epoch + 1)
            
            # Clean up memory
            del y_pred_act, y_onehot

print(f"\nTraining completed!")
print(f"Best AUC: {best_metric:.4f} achieved at epoch {best_metric_epoch}")
writer.close()

# Step 10: Plot training results
print("Creating training plots...")
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.title("Training Loss Over Time")
epochs_x = [i + 1 for i in range(len(epoch_loss_values))]
plt.plot(epochs_x, epoch_loss_values, 'b-', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Plot validation AUC
plt.subplot(1, 2, 2)
plt.title("Validation AUC Over Time")
val_epochs_x = [val_interval * (i + 1) for i in range(len(metric_values))]
plt.plot(val_epochs_x, metric_values, 'r-', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 11: Test the best model
print("Testing the best model on test set...")

# Load the best model
model.load_state_dict(torch.load(os.path.join(root_dir, "best_model.pth"), weights_only=True))
model.eval()

# Make predictions on test set
y_true = []
y_pred = []

with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        
        # Get predictions
        predictions = model(test_images).argmax(dim=1)  # Get the class with highest probability
        
        # Store true labels and predictions
        for i in range(len(predictions)):
            y_true.append(test_labels[i].item())
            y_pred.append(predictions[i].item())

# Print detailed results
print("\n" + "="*60)
print("FINAL TEST RESULTS")
print("="*60)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Clean up temporary directory if created one
if directory is None:
    shutil.rmtree(root_dir)
    print("Cleaned up temporary files.")

print("\nAll done!")
