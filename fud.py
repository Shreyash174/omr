import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pygame
import time
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
DATA_DIR = "omr_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ======================
# 1. Enhanced Data Preparation
# ======================
def prepare_training_data():
    """Organize note images into subdirectories by note name and type"""
    print("\nPreparing training data structure...")
    
    # Basic notes in multiple octaves
    notes = ["C", "D", "E", "F", "G", "A", "B"]
    octaves = [3, 4, 5]
    
    # Note durations (optional - for more advanced recognition)
    durations = ["quarter", "half", "whole", "eighth"]
    
    # Create directories for each note class
    note_classes = []
    for note in notes:
        for octave in octaves:
            note_name = f"{note}{octave}"
            note_dir = os.path.join(DATA_DIR, note_name)
            os.makedirs(note_dir, exist_ok=True)
            note_classes.append(note_name)
    
    # Optionally create directories for different note durations
    if False:  # Set to True if you want to add duration recognition
        note_classes = []
        for note in notes:
            for octave in octaves:
                for duration in durations:
                    note_name = f"{note}{octave}_{duration}"
                    note_dir = os.path.join(DATA_DIR, note_name)
                    os.makedirs(note_dir, exist_ok=True)
                    note_classes.append(note_name)
    
    print(f"Created {len(note_classes)} note class directories")
    for note_class in note_classes:
        print(f"  - {os.path.join(DATA_DIR, note_class)}")
    
    return note_classes

# ======================
# 2. Sheet Music Segmentation
# ======================
def segment_sheet_music(sheet_image_path, output_dir=None, visualize=False):
    """Extract individual note images from a full sheet music image"""
    print(f"\nSegmenting sheet music: {sheet_image_path}")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Read the sheet music image
    sheet = cv2.imread(sheet_image_path)
    if sheet is None:
        raise ValueError(f"Could not read image {sheet_image_path}")
    
    # Convert to grayscale if it's not already
    if len(sheet.shape) == 3:
        sheet_gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    else:
        sheet_gray = sheet.copy()
    
    # Keep a color copy for visualization
    sheet_color = cv2.cvtColor(sheet_gray, cv2.COLOR_GRAY2BGR)
    
    # Preprocessing - binarize the image
    _, binary = cv2.threshold(sheet_gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Detect staff lines
    horizontal = np.copy(binary)
    horizontal_size = horizontal.shape[1] // 30
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
    
    # Detect the staff line positions (useful for note pitch classification)
    staff_rows = []
    staff_row_sum = np.sum(horizontal, axis=1)
    for i, row_sum in enumerate(staff_row_sum):
        if row_sum > horizontal.shape[1] * 0.5:  # If more than 50% of the row is white
            staff_rows.append(i)
    
    # Group staff rows into staff systems
    staff_systems = []
    current_system = []
    for i in range(len(staff_rows)):
        if i == 0:
            current_system.append(staff_rows[i])
        elif staff_rows[i] - staff_rows[i-1] < 20:  # Threshold for staff line spacing
            current_system.append(staff_rows[i])
        else:
            if current_system:
                staff_systems.append(current_system)
            current_system = [staff_rows[i]]
    
    if current_system:
        staff_systems.append(current_system)
    
    # Find contours of possible notes (after removing staff lines for better isolation)
    no_staff = binary.copy()
    no_staff[horizontal > 0] = 0
    
    # Dilate to connect note components
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(no_staff, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and extract note images
    note_images = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size (adjust these thresholds based on your sheet music)
        if 10 < w < 70 and 20 < h < 120:
            # Add padding around the note
            padding = 5
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(binary.shape[1] - x_pad, w + 2*padding)
            h_pad = min(binary.shape[0] - y_pad, h + 2*padding)
            
            # Extract the note image
            note_img = binary[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            # Determine which staff system this note belongs to
            staff_idx = -1
            for idx, staff in enumerate(staff_systems):
                staff_top = min(staff)
                staff_bottom = max(staff)
                note_center_y = y + h/2
                
                if staff_top - 30 <= note_center_y <= staff_bottom + 30:
                    staff_idx = idx
                    break
            
            # Store the note image along with position and staff info
            note_info = {
                'image': note_img,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'staff_idx': staff_idx
            }
            note_images.append(note_info)
            
            # Draw bounding box for visualization
            if visualize:
                cv2.rectangle(sheet_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(sheet_color, f"{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Sort notes by x position within each staff system
    def sort_key(note):
        return (note['staff_idx'], note['x'])
    
    note_images.sort(key=sort_key)
    
    # Save individual note images if output directory is provided
    if output_dir:
        for i, note_info in enumerate(note_images):
            output_path = os.path.join(output_dir, f"note_{i+1:03d}.png")
            cv2.imwrite(output_path, note_info['image'])
            print(f"Saved note image: {output_path}")
    
    # Save visualization if requested
    if visualize:
        # Draw staff system boundaries
        for i, staff in enumerate(staff_systems):
            staff_top = min(staff) - 30
            staff_bottom = max(staff) + 30
            cv2.line(sheet_color, (0, staff_top), (sheet_color.shape[1], staff_top), (255, 0, 0), 1)
            cv2.line(sheet_color, (0, staff_bottom), (sheet_color.shape[1], staff_bottom), (255, 0, 0), 1)
            cv2.putText(sheet_color, f"Staff {i+1}", (10, staff_top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Save visualization
        viz_path = os.path.join(output_dir if output_dir else ".", "segmentation_result.png") if output_dir else "segmentation_result.png"
        cv2.imwrite(viz_path, sheet_color)
        print(f"Saved visualization: {viz_path}")
    
    print(f"Detected {len(note_images)} notes across {len(staff_systems)} staff systems")
    return note_images

# ======================
# 3. Enhanced CNN Model
# ======================
class OMR_CNN(nn.Module):
    def __init__(self, num_classes=21):  # Default for 7 notes × 3 octaves
        super(OMR_CNN, self).__init__()
        self.features = nn.Sequential(
            # First Conv Block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Second Conv Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Third Conv Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Fourth Conv Block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ======================
# 4. Advanced ResNet-style Model
# ======================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class OMR_ResNet(nn.Module):
    def __init__(self, num_classes=21):
        super(OMR_ResNet, self).__init__()
        
        # Initial layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block with potential downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# ======================
# 5. Dataset Loader
# ======================
class NoteDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform
        
        # Find all note class directories
        self.note_classes = sorted([d for d in os.listdir(data_dir) 
                                  if os.path.isdir(os.path.join(data_dir, d))])
        
        # Load images from each class
        for class_idx, note_name in enumerate(self.note_classes):
            note_dir = os.path.join(data_dir, note_name)
            for img_file in os.listdir(note_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(note_dir, img_file)
                    self.data.append((img_path, class_idx))
        
        print(f"\nLoaded {len(self.data)} images across {len(self.note_classes)} note classes")
        print("Note classes:", self.note_classes)
        
        if len(self.data) == 0:
            print("WARNING: No training images found! Add images to the note class directories.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        # Read and preprocess the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image {img_path}")
        
        # Resize and binarize
        img = cv2.resize(img, (128, 128))
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Apply transforms if any
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img, label

# ======================
# 6. Training Function
# ======================
def train_model(model_type="cnn"):
    # Determine which model to use
    if model_type.lower() == "resnet":
        print("\nUsing ResNet-style model architecture")
        model_class = OMR_ResNet
    else:
        print("\nUsing CNN model architecture")
        model_class = OMR_CNN
    
    # Data augmentation transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load dataset
    dataset = NoteDataset(DATA_DIR, transform=transform)
    if len(dataset) == 0:
        raise ValueError("No training images found. Please add images to the note class directories.")
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = model_class(num_classes=len(dataset.note_classes))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    
    # Training loop
    print("\nStarting training...")
    best_acc = 0.0
    epochs = 100
    patience = 15
    patience_counter = 0
    
    # For plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in pbar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", 
                              "acc": f"{100. * correct / total:.2f}%"})
        
        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in pbar:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", 
                                 "acc": f"{100. * val_correct / val_total:.2f}%"})
        
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)
        
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(DATA_DIR, f"omr_model_{model_type}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved: {model_path} (acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break
        
        # Early stopping if accuracy is too low after 20 epochs
        if epoch > 20 and val_acc < 50:
            print("Early stopping due to low validation accuracy")
            break
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f"training_curves_{model_type}.png"))
    plt.close()
    
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")
    
    # Load the best model for return
    best_model_path = os.path.join(DATA_DIR, f"omr_model_{model_type}_best.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
        model.eval()
    
    return model, dataset.note_classes

# ======================
# 7. Note Detection
# ======================
def detect_note(image, model, note_classes, threshold=0.5):
    """
    Detect note from image with confidence scores
    
    Args:
        image: Can be an image path or a numpy array
        model: Trained OMR model
        note_classes: List of note class names
        threshold: Confidence threshold for detection
    
    Returns:
        detected_note: Name of the detected note
        confidence: Detection confidence (0-1)
        all_probs: All class probabilities
    """
    # Handle different types of image input
    if isinstance(image, str):
        # Image path
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image {image}")
    elif isinstance(image, np.ndarray):
        # Numpy array
        img = image.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Image must be a path or numpy array")
    
    # Resize and binarize
    img = cv2.resize(img, (128, 128))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Prepare for model input
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)
    
    # Get results
    detected_note = note_classes[predicted.item()]
    confidence = conf.item()
    all_probs = probabilities[0].cpu().numpy()
    
    # Print probabilities
    print("\nDetection probabilities:")
    for i, note_name in enumerate(note_classes):
        print(f"{note_name}: {all_probs[i]:.4f}")
    
    return detected_note, confidence, all_probs

# ======================
# 8. Audio Playback
# ======================
def play_detected_sequence(note_sequence):
    """
    Play a sequence of detected notes
    
    Args:
        note_sequence: List of (note_name, confidence) tuples
    """
    print("\nPlaying detected sequence...")
    
    try:
        from music21 import stream, note, instrument, midi
        
        s = stream.Stream()
        s.append(instrument.Piano())
        
        for detected_note, confidence in note_sequence:
            # Parse the note name (e.g., "C4" or "C4_quarter")
            parts = detected_note.split('_')
            pitch = parts[0]
            
            # Set duration based on note type (if included)
            if len(parts) > 1:
                duration_map = {
                    "whole": 4.0,
                    "half": 2.0,
                    "quarter": 1.0,
                    "eighth": 0.5,
                    "sixteenth": 0.25
                }
                duration = duration_map.get(parts[1], 1.0)
            else:
                duration = 1.0  # Default to quarter note
            
            # Create note
            n = note.Note(pitch)
            n.duration.quarterLength = duration
            s.append(n)
        
        # Play the sequence
        try:
            sp = midi.realtime.StreamPlayer(s)
            sp.play()
            return s  # Return the stream for potential saving
        except:
            print("Could not play MIDI. Saving to MIDI file instead.")
            mf = s.write('midi')
            print(f"Saved to MIDI file: {mf}")
            return s
        
    except ImportError:
        print("Could not import music21. Please install it with: pip install music21")
        print("Playback will be simulated:")
        
        for detected_note, confidence in note_sequence:
            print(f"Playing note: {detected_note} (confidence: {confidence:.4f})")
            time.sleep(0.5)  # Simulate playback timing
        
        return None

# ======================
# 9. Full Sheet Processing
# ======================
def process_sheet_music(sheet_image_path, model, note_classes, output_dir=None, visualize=True):
    """
    Process a full sheet music image and play the detected notes
    
    Args:
        sheet_image_path: Path to the sheet music image
        model: Trained model
        note_classes: List of note class names
        output_dir: Directory to save segmented notes and visualization
        visualize: Whether to create visualization images
    
    Returns:
        detected_sequence: List of (note_name, confidence) tuples
    """
    print(f"\nProcessing sheet music: {sheet_image_path}")
    
    # Create temporary directory for segmented notes
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join(os.path.dirname(sheet_image_path), "processed_notes")
        os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Segment the sheet music into individual note images
    note_images = segment_sheet_music(sheet_image_path, output_dir, visualize)
    
    # Step 2: Classify each note image
    detected_sequence = []
    note_thumbnails = []
    
    print("\nClassifying notes...")
    for i, note_info in enumerate(note_images):
        note_img = note_info['image']
        
        # Detect the note
        detected_note, confidence, _ = detect_note(note_img, model, note_classes)
        detected_sequence.append((detected_note, confidence))
        
        # Save the result for visualization
        if visualize:
            # Create a colored thumbnail with note name
            thumbnail = cv2.cvtColor(note_img, cv2.COLOR_GRAY2BGR)
            color = (0, 255, 0) if confidence >= 0.7 else (0, 165, 255) if confidence >= 0.5 else (0, 0, 255)
            cv2.putText(thumbnail, detected_note, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            note_thumbnails.append((thumbnail, note_info['x']))
    
    # Step 3: Display the detected sequence
    print("\nDetected note sequence:")
    for i, (note, conf) in enumerate(detected_sequence):
        print(f"Note {i+1}: {note} (confidence: {conf:.4f})")
    
    # Step 4: Create visualization with all detected notes
    if visualize and note_thumbnails:
        # Sort thumbnails by x position
        note_thumbnails.sort(key=lambda x: x[1])
        
        # Create a row of thumbnails
        max_height = max(thumb.shape[0] for thumb, _ in note_thumbnails)
        total_width = sum(thumb.shape[1] for thumb, _ in note_thumbnails) + 10 * (len(note_thumbnails) - 1)
        
        # Visualization image
        vis_img = np.ones((max_height + 40, total_width, 3), dtype=np.uint8) * 255
        
        # Add thumbnails
        x_offset = 0
        for thumb, _ in note_thumbnails:
            h, w = thumb.shape[:2]
            vis_img[20:20+h, x_offset:x_offset+w] = thumb
            x_offset += w + 10
        
        # Add title
        cv2.putText(vis_img, "Detected Notes Sequence", (10, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Save visualization
        vis_path = os.path.join(output_dir, "detected_sequence.png")
        cv2.imwrite(vis_path, vis_img)
        print(f"Saved sequence visualization: {vis_path}")
    
    # Step 5: Play the sequence
    if detected_sequence:
        play_detected_sequence(detected_sequence)
    
    return detected_sequence

# ======================
# 10. Data Augmentation Utilities
# ======================
def augment_training_data(input_dir, output_dir=None, num_augmentations=5):
    """
    Augment existing training data to create more samples
    
    Args:
        input_dir: Directory containing note class subdirectories
        output_dir: Directory to save augmented images (defaults to input_dir)
        num_augmentations: Number of augmented images to create per original image
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all note class directories
    note_classes = sorted([d for d in os.listdir(input_dir) 
                          if os.path.isdir(os.path.join(input_dir, d))])
    
    print(f"\nAugmenting training data for {len(note_classes)} note classes...")
    
    # Setup augmentation transforms
    transforms_list = [
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.8),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ]
    
    # Process each note class
    total_original = 0
    total_augmented = 0
    
    for note_class in note_classes:
        input_class_dir = os.path.join(input_dir, note_class)
        output_class_dir = os.path.join(output_dir, note_class)
        
        # Create output class directory if it doesn't exist
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Find all images in the class directory
        image_files = [f for f in os.listdir(input_class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        total_original += len(image_files)
        
        for img_file in image_files:
            img_path = os.path.join(input_class_dir, img_file)
            
            # Read image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # Convert to PIL Image for transformations
            pil_img = Image.fromarray(img)
            
            # Create augmented images
            for i in range(num_augmentations):
                # Apply random augmentations
                augmented_img = pil_img
                for transform in transforms_list:
                    if np.random.random() > 0.5:  # 50% chance to apply each transform
                        augmented_img = transform(augmented_img)
                
                # Convert back to numpy array
                augmented_array = np.array(augmented_img)
                
                # Save augmented image
                base_name = os.path.splitext(img_file)[0]
                aug_path = os.path.join(output_class_dir, f"{base_name}_aug_{i+1}.png")
                cv2.imwrite(aug_path, augmented_array)
                total_augmented += 1
    
    print(f"Augmentation complete! Created {total_augmented} augmented images from {total_original} original images.")

# ======================
# 11. Interactive Visualization
# ======================
def create_visualizations(dataset, model, note_classes):
    """
    Create visualizations of the model's understanding of note classes
    
    Args:
        dataset: Dataset instance
        model: Trained model
        note_classes: List of note class names
    """
    print("\nCreating model visualizations...")
    
    # Create output directory for visualizations
    vis_dir = os.path.join(DATA_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create a confusion matrix visualization
    # Select a subset of validation samples
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create and plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=note_classes,
                yticklabels=note_classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"))
    plt.close()
    
    # Create sample prediction visualizations
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()
    
    # Get some sample images and their predictions
    samples_shown = 0
    for inputs, labels in val_loader:
        if samples_shown >= 15:
            break
            
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        for i in range(min(inputs.shape[0], 15 - samples_shown)):
            img = inputs[i].squeeze().cpu().numpy()
            label = labels[i].item()
            pred = preds[i].item()
            
            ax = axes[samples_shown]
            ax.imshow(img, cmap='gray')
            
            title_color = 'green' if label == pred else 'red'
            ax.set_title(f"True: {note_classes[label]}\nPred: {note_classes[pred]}", 
                         color=title_color, fontsize=9)
            ax.axis('off')
            
            samples_shown += 1
            if samples_shown >= 15:
                break
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "sample_predictions.png"))
    plt.close()
    
    print(f"Visualizations saved to {vis_dir}")

# ======================
# 12. Main Execution
# ======================
def main():
    try:
        pygame.init()
        print("\n===== Optical Music Recognition (OMR) System =====")
        print("This system detects and plays musical notes from sheet music.")
        
        # Step 1: Prepare training data structure
        note_classes = prepare_training_data()
        
        # Main menu loop
        while True:
            print("\n===== OMR System Menu =====")
            print("1. Train model")
            print("2. Process sheet music")
            print("3. Detect single note")
            print("4. Augment training data")
            print("5. Create visualizations")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                # Train model
                model_type = input("Choose model type (cnn/resnet): ").strip().lower()
                if model_type not in ['cnn', 'resnet']:
                    model_type = 'cnn'  # Default to CNN
                
                model, note_classes = train_model(model_type)
                
            elif choice == '2':
                # Process sheet music
                model_path = os.path.join(DATA_DIR, "omr_model_cnn_best.pth")
                resnet_path = os.path.join(DATA_DIR, "omr_model_resnet_best.pth")
                
                # Check which model to use
                if os.path.exists(resnet_path):
                    print("Using ResNet model")
                    model = OMR_ResNet(num_classes=len(note_classes))
                    model.load_state_dict(torch.load(resnet_path, map_location=torch.device('cpu')))
                elif os.path.exists(model_path):
                    print("Using CNN model")
                    model = OMR_CNN(num_classes=len(note_classes))
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                else:
                    print("No trained model found. Please train the model first.")
                    continue
                
                model.eval()
                
                # Get sheet music path
                sheet_music_path = input("Enter path to sheet music image: ").strip()
                if not os.path.exists(sheet_music_path):
                    print(f"Error: File not found at {sheet_music_path}")
                    continue
                
                # Process the sheet music
                process_sheet_music(sheet_music_path, model, note_classes)
                
            elif choice == '3':
                # Detect single note
                model_path = os.path.join(DATA_DIR, "omr_model_cnn_best.pth")
                resnet_path = os.path.join(DATA_DIR, "omr_model_resnet_best.pth")
                
                # Check which model to use
                if os.path.exists(resnet_path):
                    print("Using ResNet model")
                    model = OMR_ResNet(num_classes=len(note_classes))
                    model.load_state_dict(torch.load(resnet_path, map_location=torch.device('cpu')))
                elif os.path.exists(model_path):
                    print("Using CNN model")
                    model = OMR_CNN(num_classes=len(note_classes))
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                else:
                    print("No trained model found. Please train the model first.")
                    continue
                
                model.eval()
                
                # Get note image path
                note_path = input("Enter path to note image: ").strip()
                if not os.path.exists(note_path):
                    print(f"Error: File not found at {note_path}")
                    continue
                
                # Detect the note
                detected_note, confidence, _ = detect_note(note_path, model, note_classes)
                print(f"\nDetected note: {detected_note} (confidence: {confidence:.4f})")
                
                # Play the note
                play = input("Play the detected note? (y/n): ").lower()
                if play == 'y':
                    play_detected_sequence([(detected_note, confidence)])
                
            elif choice == '4':
                # Augment training data
                augment_training_data(DATA_DIR, num_augmentations=5)
                
            elif choice == '5':
                # Create visualizations
                model_path = os.path.join(DATA_DIR, "omr_model_cnn_best.pth")
                resnet_path = os.path.join(DATA_DIR, "omr_model_resnet_best.pth")
                
                # Check which model to use
                if os.path.exists(resnet_path):
                    print("Using ResNet model for visualizations")
                    model = OMR_ResNet(num_classes=len(note_classes))
                    model.load_state_dict(torch.load(resnet_path, map_location=torch.device('cpu')))
                elif os.path.exists(model_path):
                    print("Using CNN model for visualizations")
                    model = OMR_CNN(num_classes=len(note_classes))
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                else:
                    print("No trained model found. Please train the model first.")
                    continue
                
                model.eval()
                
                # Create dataset
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                dataset = NoteDataset(DATA_DIR, transform=transform)
                
                create_visualizations(dataset, model, note_classes)
                
            elif choice == '6':
                # Exit
                print("\nExiting OMR System. Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter a number between 1 and 6.")
    
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()