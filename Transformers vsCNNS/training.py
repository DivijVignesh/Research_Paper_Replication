import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm
import os
from tqdm import tqdm
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories for saving models
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Data preprocessing and augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Upscale from 32x32 to 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Model configuration
model_configs = {
    'vgg19': {
        'model_name': 'vgg19',
        'unfreeze_from': 'features.32',  # block5_conv1 equivalent
        'batch_size': 512
    },
    'resnet152': {
        'model_name': 'resnet152',
        'unfreeze_from': 'layer4.0',  # conv5_block1 equivalent
        'batch_size': 512
    },
    'densenet201': {
        'model_name': 'densenet201',
        'unfreeze_from': 'features.denseblock4.denselayer1',  # conv5_block1_0_bn equivalent
        'batch_size': 512
    },
    'efficientnetv2_l': {
        'model_name': 'efficientnetv2_l',
        'unfreeze_from': 'blocks.6',  # Last InvertedResidual block
        'batch_size': 512
    },
    'nfnet_f6': {
        'model_name': 'nfnet_f6',
        'unfreeze_from': 'stages.3',  # Last NormFreeBlock
        'batch_size': 512
    },
    'coat_lite_small': {
        'model_name': 'coat_lite_small',
        'unfreeze_from': 'serial_blocks4',  # Last SerialBlock
        'batch_size': 512
    },
    'cait_s24_224': {
        'model_name': 'cait_s24_224',
        'unfreeze_from': 'blocks_token_only.23',  # Last block of token only module
        'batch_size': 512
    },
    'deit_base_distilled_patch16_224': {
        'model_name': 'deit_base_distilled_patch16_224',
        'unfreeze_from': 'blocks.11',  # 12th Transformer Block (0-indexed)
        'batch_size': 512
    },
    'swin_base_patch4_window7_224': {
        'model_name': 'swin_base_patch4_window7_224',
        'unfreeze_from': 'layers.3',  # Last BasicLayer
        'batch_size': 256  # Reduced batch size due to GPU limitations
    },
    'vit_large_patch32_224': {
        'model_name': 'vit_large_patch32_224',
        'unfreeze_from': 'blocks.23',  # Transformer/encoder block 23
        'batch_size': 512
    }
}

def freeze_model_except_last_block(model, model_name, unfreeze_from):
    """Freeze all layers except the specified block and classifier"""
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the specified block
    found_unfreeze_point = False
    for name, module in model.named_modules():
        if unfreeze_from in name:
            found_unfreeze_point = True
            for param in module.parameters():
                param.requires_grad = True
    
    # Always unfreeze the classifier/head
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
    elif hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters for {model_name}: {trainable_params:,}")
    
    return model

def create_model(model_name):
    """Create and modify model for CIFAR-10 classification"""
    try:
        # Load pretrained model
        model = timm.create_model(model_name, pretrained=True, num_classes=10)
        print(f"Successfully loaded {model_name}")
        return model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def train_model(model, model_name, train_loader, val_loader, epochs=20):
    """Train the model with specified hyperparameters"""
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=3, 
                                min_lr=0.0000001, verbose=True)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            # Update progress bar
            train_acc = 100. * correct_train / total_train
            train_pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{train_acc:.2f}%'
            })
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for batch_idx, (inputs, targets) in enumerate(val_pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
                
                val_acc = 100. * correct_val / total_val
                val_pbar.set_postfix({
                    'Loss': f'{val_loss/(batch_idx+1):.3f}',
                    'Acc': f'{val_acc:.2f}%'
                })
        
        # Calculate metrics
        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_acc = 100. * correct_val / total_val
        epoch_train_acc = 100. * correct_train / total_train
        
        train_losses.append(epoch_train_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, Val Acc: {epoch_val_acc:.2f}%')
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_acc)
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': epoch_val_acc,
                'train_acc': epoch_train_acc,
                'train_loss': epoch_train_loss
            }, f'saved_models/{model_name}_best.pth')
            print(f'New best validation accuracy: {best_val_acc:.2f}%')
    
    return best_val_acc, train_losses, val_accuracies

def main():
    """Main training loop for all models"""
    results = {}
    
    for config_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing {config['model_name']}")
        print(f"{'='*60}")
        
        # Create data loaders with appropriate batch size
        train_loader = DataLoader(trainset, batch_size=config['batch_size'], 
                                shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(testset, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=4, pin_memory=True)
        
        # Create model
        model = create_model(config['model_name'])
        if model is None:
            print(f"Skipping {config['model_name']} due to loading error")
            continue
        
        # Freeze model except last block
        model = freeze_model_except_last_block(model, config['model_name'], 
                                             config['unfreeze_from'])
        
        # Train model
        try:
            best_acc, train_losses, val_accs = train_model(
                model, config['model_name'], train_loader, val_loader
            )
            
            results[config['model_name']] = {
                'best_validation_accuracy': best_acc,
                'final_train_losses': train_losses,
                'validation_accuracies': val_accs,
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            print(f"\n{config['model_name']} completed!")
            print(f"Best validation accuracy: {best_acc:.2f}%")
            
        except Exception as e:
            print(f"Error training {config['model_name']}: {e}")
            continue
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Save results summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    results_summary = []
    for model_name, result in results.items():
        print(f"{model_name}: {result['best_validation_accuracy']:.2f}% "
              f"(Trainable params: {result['trainable_params']:,})")
        results_summary.append({
            'model': model_name,
            'accuracy': result['best_validation_accuracy'],
            'trainable_params': result['trainable_params']
        })
    
    # Save detailed results
    torch.save(results, 'results/detailed_results.pth')
    
    # Sort by accuracy
    results_summary.sort(key=lambda x: x['accuracy'], reverse=True)
    print(f"\nTop performing models:")
    for i, result in enumerate(results_summary[:5]):
        print(f"{i+1}. {result['model']}: {result['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
