import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import copy

#HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("HEIC support enabled")
except ImportError:
    print("Warning: pillow_heif not installed. HEIC files will not be supported.")
    print("Install with: pip install pillow-heif")

data_dir = r'C:\Users\Or\lash_rod_classification\data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

print(f"Train directory: {train_dir}")
print(f"Validation directory: {val_dir}")
print(f"Test directory: {test_dir}")

# Enhanced data transformations 
data_transforms = {
    'train': transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Custom Dataset class
class LashRodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['lift_rod_eyes', 'round_rod_eyes']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"Warning: Directory {cls_dir} does not exist")
                continue
                
            for img_name in os.listdir(cls_dir):
                # Include HEIC and HEIF formats
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.heic', '.heif')):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
        
        # Print dataset statistics
        if len(self.samples) > 0:
            class_counts = {}
            for _, class_idx in self.samples:
                class_name = self.classes[class_idx]
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                
            print(f"Dataset at {root_dir} has {len(self.samples)} images:")
            for class_name, count in class_counts.items():
                print(f"  - {class_name}: {count} images")
        else:
            print(f"Warning: No images found in {root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # Open image file and convert to RGB
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder if image loading fails
            placeholder = torch.zeros(3, 224, 224)
            return placeholder, label

# Create datasets
image_datasets = {
    'train': LashRodDataset(train_dir, data_transforms['train']),
    'val': LashRodDataset(val_dir, data_transforms['val']),
    'test': LashRodDataset(test_dir, data_transforms['test'])
}

# Check if datasets are valid
for phase in ['train', 'val', 'test']:
    if len(image_datasets[phase]) == 0:
        print(f"Warning: {phase} dataset is empty!")

# Count file types
def count_file_types(directory):
    extensions = {}
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file.lower())[1]
            if ext:
                if ext not in extensions:
                    extensions[ext] = 0
                extensions[ext] += 1
    return extensions

print("\nFile types in data directory:")
file_types = count_file_types(data_dir)
for ext, count in file_types.items():
    print(f"  {ext}: {count} files")

batch_size = 8  
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes if len(image_datasets['train']) > 0 else ['lift_rod_eyes', 'round_rod_eyes']

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model 
def initialize_model():
    # Load pretrained ResNet-50
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # check!!!!!!!!!!
    for param in model.layer3.parameters():
        param.requires_grad = True

    # Unfreeze the last layer block
    for param in model.layer4.parameters():
        param.requires_grad = True

    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)  # 2 classes: lift rod and round rod
    )
    
    return model

# Training function with early stopping
def train_model(model, criterion, optimizer, scheduler, num_epochs=18):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 5  # Early stopping patience
    
    # Track losses and accuracies
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                batch_size = inputs.size(0)
                running_samples += batch_size

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
            
            # Calculate proper epoch statistics based on actual processed samples
            if running_samples > 0:
                epoch_loss = running_loss / running_samples
                epoch_acc = running_corrects.double() / running_samples
            else:
                epoch_loss = float('inf')
                epoch_acc = 0.0
                
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # LR scheduler step
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                    else:
                        scheduler.step()
                
                # Early stopping check
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stop_counter = 0
                elif epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    early_stop_counter = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    early_stop_counter += 1

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Early stopping check
        #if early_stop_counter >= patience:
           # print(f"Early stopping triggered at epoch {epoch+1}")
           # break

        print()

    # Print final results
    time_elapsed = "training completed"
    print(f'\n{time_elapsed}')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Evaluation function
def evaluate_model(model, phase='test'):
    model.eval()
    
    # If test set is empty, use validation set
    if dataset_sizes[phase] == 0:
        print(f"Warning: {phase} dataset is empty! Using validation set instead.")
        phase = 'val'
    
    # Still empty? Use training set
    if dataset_sizes[phase] == 0:
        print(f"Warning: {phase} dataset is empty! Using training set instead.")
        phase = 'train'
    
    # If all datasets are empty, return without evaluation
    if dataset_sizes[phase] == 0:
        print("Error: All datasets are empty. Cannot evaluate model.")
        return None, None
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate and print metrics if we got predictions
    if all_preds and all_labels:
        try:
            cm = confusion_matrix(all_labels, all_preds)
            print("Confusion Matrix:")
            print(cm)
            
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=class_names))
        except Exception as e:
            print(f"Error generating evaluation metrics: {e}")
    else:
        print("No predictions were made. Check your evaluation dataset.")
    
    return all_preds, all_labels

# Plot training history
def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    try:
        plt.savefig(os.path.join(data_dir, 'training_history.png'))
        print(f"Training plot saved to {os.path.join(data_dir, 'training_history.png')}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        
    plt.close()  

# Function to make predictions on new images
def predict_image(model, image_path, transform):
    model.eval()
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)
        
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        return class_names[preds[0].item()], probs[0][preds[0]].item()
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return None, 0.0

# Main execution
def main():
    # Initialize model
    model = initialize_model()
    model = model.to(device)
    
    # Calculate class weights for imbalanced dataset
    if dataset_sizes['train'] > 0:
        class_counts = np.zeros(len(class_names))
        for _, label in image_datasets['train'].samples:
            class_counts[label] += 1
        
        # Avoid division by zero
        class_counts = np.where(class_counts == 0, 1, class_counts)
        
        # Calculate weights (inverse frequency)
        class_weights = torch.FloatTensor(1.0 / class_counts)
        class_weights = class_weights / class_weights.sum() * len(class_names)
        print(f"Class weights: {class_weights}")
        
        # Move weights to device
        class_weights = class_weights.to(device)
        
        # Define loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=0.001, weight_decay=0.0001)
    
    # Learning rate scheduler - simple step scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    print("Starting training...")
    model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=25)
    
    # Save the model
    model_save_path = os.path.join(data_dir, 'lash_rod_classifier.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Evaluate the model
    print("Evaluating model on test set...")
    evaluate_model(model, 'test')
    
    # Plot training history
    print("Generating training history plot...")
    plot_training_history(history)
    
    # Output a couple of sample predictions
    print("\nSample predictions:")
    if len(image_datasets['test'].samples) > 0:
        test_img_path, _ = image_datasets['test'].samples[0]
        pred_class, confidence = predict_image(model, test_img_path, data_transforms['test'])
        print(f"Image: {test_img_path}")
        print(f"Prediction: {pred_class} with {confidence*100:.1f}% confidence")
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()