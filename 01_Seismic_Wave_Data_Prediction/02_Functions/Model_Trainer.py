#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model training for earthquake magnitude prediction.
This module handles the training process for the neural network models.
"""

import os
import torch
from torch import nn, optim
from torch.amp import GradScaler 
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path
from collections import Counter
from math import pow
from torch.utils.data import DataLoader
#import torch, numpy as np, torch.serialization as _ts





from SeisLM_train import SeisLMForMagnitudePrediction, ScatterEarthquakePredictor
from seisLM.model.foundation.pretrained_models import LitMultiDimWav2Vec2


def epoch_train_accuracy(model, loader, device, use_tabular_features: bool):
    print('Calculating true epoch accuracy...')
    model.eval()                                # turn off dropout / BN updates
    correct, total = 0, 0
    with torch.no_grad():                       # no gradients, no autograd graph
        for batch in loader:
            if use_tabular_features:
                x, tabs, _, y = batch           # keep the same unpacking logic
                x, tabs, y = x.to(device), tabs.to(device), y.to(device)
                logits = model(x, tabs)
            else:
                x, _, y = batch                 # tabs placeholder is "_"
                x, y = x.to(device), y.to(device)
                logits = model(x)

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.numel()

    model.train()                               # IMPORTANT: switch back
    return correct / total if total else 0.0


def count_params(model):
    trainable, frozen = 0, 0
    for p in model.parameters():
        num = p.numel()
        if p.requires_grad:
            trainable += num
        else:
            frozen += num
    return trainable, frozen




class ModelTrainer:
    """
    Class for training earthquake prediction models.
    Now supports incorporating tabular features.
    """
    
    def __init__(self, config, output_dir, device=None):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
            output_dir: Directory to save outputs
            device: Device to run training on ('cuda' or 'cpu')
        """
        self.config = config
        
        self.output_dir = output_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = config['model']['model_type']
        self.aggregation_type = config['model']['aggregation_type']
        self.num_classes = config['model']['num_classes']
        
        # Flag to control use of tabular features
        self.use_tabular_features = config['data'].get('use_tabular_features', False)
        
        # Get tabular features dimension if using them
        if self.use_tabular_features:
            # Try to determine the dimension from config or use a default
            self.tabular_features_dim = config['data'].get('tabular_features_dim', 0)
            
            # If dimension is not provided, we'll try to determine it at model initialization
            if self.tabular_features_dim == 0:
                print("Warning: Tabular features dimension not specified. Will attempt to determine from dataset.")
        else:
            self.tabular_features_dim = -1
        
        # Create checkpoints directory
        self.checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Set environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
    def initialize_model(self):
        """
        Initialize model based on configuration.
        
        Returns:
            Initialized model
        """
        if self.model_type == 'seislm':
            print("Loading pretrained SeisLM model...")
            # Load the pre-trained seisLM model
            #_ts.add_safe_globals([np.core.multiarray.scalar])
            
            ckpt_path = Path(self.config['paths']['pretrained_seislm_model'])
            # 1) raw torch.load with full pickle allowed
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            # 2) instantiate the module and load weights
            pretrained = LitMultiDimWav2Vec2(**checkpoint['hyper_parameters'])
            pretrained.load_state_dict(checkpoint['state_dict'], strict=False)
            pretrained_model = pretrained.model    
            
            #pretrained_model = LitMultiDimWav2Vec2.load_from_checkpoint(
            #    self.config['paths']['pretrained_seislm_model']
            #).model
            
            if self.config['model']['freeze_backbone']:
                print("Freezing seislm model parameters...")
                # Freeze all parameters in the pretrained model
                for param in pretrained_model.parameters():
                    param.requires_grad = False
                
            print(f"Initializing SeisLM model with {self.aggregation_type} aggregation...")
            toto_cfg = self.config['model'].get('toto', {})
            
            model = SeisLMForMagnitudePrediction(
                pretrained_model, 
                aggregation_type=self.aggregation_type,
                num_classes=self.num_classes,
                use_tabular_features=self.use_tabular_features,
                tabular_features_dim=self.tabular_features_dim,
                device=self.device,
                toto_cfg=toto_cfg  
            )
            totals = {"trainable": 0, "frozen": 0}

            """for p in model.toto.parameters():              # <─ one variable
                if p.requires_grad:
                    totals["trainable"] += p.numel()
                else:
                    totals["frozen"] += p.numel()

            assert totals["trainable"] == 0, (
                f"{totals['trainable']:,} Toto parameters are still trainable!"
            )
            print("✓ Toto parameters   trainable:", totals['trainable'],
                  "   frozen:", totals['frozen'])"""

        else:
            print(f"Initializing Scattering model with {self.aggregation_type} aggregation...")
            model = ScatterEarthquakePredictor(
                aggregation_type=self.aggregation_type,
                num_classes=self.num_classes,
                use_tabular_features=self.use_tabular_features,
                tabular_features_dim=self.tabular_features_dim,
                device=self.device
            )
        
        # Use DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training!")
            model = nn.DataParallel(model)
        
        # Move model to device
        model = model.to(self.device)
        
        return model
    
    def load_model(self, checkpoint_path: str, warn_threshold: float = 0.10) -> torch.nn.Module:
        """
        Load a model from `checkpoint_path` and verify that its parameters differ
        from a freshly initialised network (i.e. the model really trained).

        Args
        ----
        checkpoint_path : str
            Path to the checkpoint file.
        warn_threshold : float, optional
            Fraction of tensors that may remain identical to the fresh initialisation
            before a warning is emitted. 0.10 = 10 %.

        Returns
        -------
        torch.nn.Module
            The model with weights loaded from the checkpoint.
        """
        # ------------------------------------------------------------------ #
        # 1. Create a reference copy of an *untrained* model for comparison   #
        # ------------------------------------------------------------------ #
        ref_model = self.initialize_model()                  # fresh weights
        ref_state  = {k: v.clone() for k, v in ref_model.state_dict().items()}

        # ------------------------------------------------------------------ #
        # 2. Load the checkpoint into a new model instance                    #
        # ------------------------------------------------------------------ #
        model = self.initialize_model()
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # ------------------------------------------------------------------ #
        # 3. Sanity-check: how many tensors are **exactly** the same?         #
        # ------------------------------------------------------------------ #
        """identical = 0
        for name, tensor in model.state_dict().items():
            if torch.equal(tensor.cpu(), ref_state[name].cpu()):
                identical += 1

        total      = len(ref_state)
        frac_same  = identical / total

        if identical == 0:
            print("✓ All parameters differ from fresh initialisation – looks trained.")
        elif frac_same <= warn_threshold:
            print(f"⚠️  {identical} / {total} tensors unchanged ({frac_same:.1%}). "
                  "Most parameters differ, so the checkpoint is probably trained.")
        else:
            raise RuntimeError(
                f"{identical} / {total} tensors ({frac_same:.1%}) are *identical* "
                "to fresh initialisation – checkpoint may be untrained or corrupted."
            )"""

        # ------------------------------------------------------------------ #
        # 4. Informative log and return                                       #
        # ------------------------------------------------------------------ #
        print(f"Loaded model from epoch {checkpoint['epoch']} "
              f"with loss {checkpoint['loss']:.4f} and accuracy {checkpoint['accuracy']:.4f}")

        model.to(self.device)
        return model

    
    def build_balanced_criterion(self, train_dataset, beta: float = 0.999):
        """
        train_dataset  : the *Dataset* object, not the DataLoader
        """
        print("Building balanced criterion …")

        # ------------------------------------------------------------------
        # 1) Count labels without touching waveform tensors
        scan_loader = DataLoader(
            train_dataset,
            batch_size=1024,
            shuffle=False,
            num_workers=0,                 # avoid multiprocessing traps
            collate_fn=lambda batch: [item[-1] for item in batch]  # <── ONLY LABELS
        )

        label_counts = Counter()
        for batch_labels in scan_loader:   # batch_labels is a list of tensors/ints
            label_counts.update(int(lbl) for lbl in batch_labels)

        print("Label counts:", label_counts)

        # ------------------------------------------------------------------
        # 2) Compute effective-number weights
        num_classes = max(label_counts) + 1
        counts = torch.tensor(
            [label_counts.get(c, 0) for c in range(num_classes)],
            dtype=torch.float,
            device=self.device,
        )

        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights /= weights.sum() * num_classes

        print("Class weights:", weights.tolist())

        # ------------------------------------------------------------------
        return nn.CrossEntropyLoss(weight=weights)

    
    def train(self, data_loader):
        """
        Train the model.
        
        Args:
            data_loader: DataLoader for training data
            
        Returns:
            Tuple of (trained_model, best_checkpoint_path)
        """
        # Initialize model
        model = self.initialize_model()
        t, f = count_params(model)
        print(f"Trainable params: {t:,}\nFrozen params:    {f:,}")
        
        # Setup loss function, optimizer, and scheduler
        #criterion = nn.CrossEntropyLoss()
        criterion = self.build_balanced_criterion(data_loader.dataset,          # pass the Dataset, not the loader
                                                    beta=self.config['training'].get('class_balance_beta', 0.999),
                    )
        
        
        trainable = (p for p in model.parameters() if p.requires_grad)
        optimizer = optim.AdamW(
            trainable, 
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        scaler = GradScaler() # For mixed precision
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2    )
        
        # Training Loop
        print("Starting training...")
        best_loss = float('inf')
        best_checkpoint_path = None
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"EPOCH {epoch+1}/{self.config['training']['num_epochs']} " + "-" * 50)
            model.train()
            epoch_loss = 0.0
            running_loss = 0.0
            batch_count = 0
            correct_predictions = 0
            total_samples = 0
            
            for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}")):
                try:
                    # Handle different batch structures based on tabular features usage
                    if self.use_tabular_features:
                        # Unpack structured batch data
                        windows_tensor, tabular_features, _, label = batch_data
                        windows_tensor = windows_tensor.to(self.device)
                        tabular_features = tabular_features.to(self.device)
                        label = label.to(self.device)
                    else:
                        # Standard batch structure
                        windows_tensor, tabular_features, label  = batch_data
                        windows_tensor = windows_tensor.to(self.device)
                        label = label.to(self.device)
                        tabular_features = None
                    
                    # Print shapes for debugging - only first batch of first epoch
                    if epoch == 0 and batch_idx == 0:
                        print(f"Input tensor shape: {windows_tensor.shape}")
                        if self.use_tabular_features:
                            print(f"Tabular features shape: {tabular_features.shape}")
                        print(f"Label shape: {label.shape}")
                    
                    optimizer.zero_grad()
                    
                    try:
                        # Mixed precision forward pass
                        with torch.amp.autocast(device_type=self.device.type if self.device.type != 'cpu' else 'cpu'):
                            # Forward pass with or without tabular features
                            if self.use_tabular_features:
                                output = model(windows_tensor, tabular_features)
                            else:
                                output = model(windows_tensor)
                                
                            loss = criterion(output, label)
                        
                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        
                        # Clip gradients to avoid explosion
                        if self.config['training']['grad_clip'] > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training']['grad_clip'])
                        
                        # Update weights
                        scaler.step(optimizer)
                        scaler.update()
                        
                        # Calculate classification metrics
                        _, predicted = torch.max(output.data, 1)  # Get predicted class indices
                        batch_correct = (predicted == label).sum().item()  # Count correct predictions
                        batch_total = label.size(0)  # Total samples in batch
                        
                        # Update accuracy stats
                        correct_predictions += batch_correct
                        total_samples += batch_total
                        
                        # Calculate cross-entropy loss
                        batch_loss = loss.item()
                        
                        # Update running stats
                        running_loss += batch_loss
                        epoch_loss += batch_loss
                        batch_count += 1
                        
                        # Log progress
                        if (batch_idx + 1) % self.config['training']['log_interval'] == 0:
                            avg_loss = running_loss / self.config['training']['log_interval']
                            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
                            print(f"Batch [{batch_idx+1}/{len(data_loader)}], "
                                f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                            running_loss = 0.0
                            
                            # Memory monitoring
                            if self.config['training']['memory_monitoring'] and torch.cuda.is_available():
                                current_mem = torch.cuda.memory_allocated(self.device) / (1024**3)
                                max_mem = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                                print(f"GPU Memory: Current {current_mem:.2f} GB, Max {max_mem:.2f} GB")
                    
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print("Out of memory error!")
                            if torch.cuda.is_available():
                                print(f"Memory allocated just before error: "
                                    f"{torch.cuda.memory_allocated()/1e9:.2f} GB")
                            torch.cuda.empty_cache()
                            raise e
                
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
            
            # Calculate average loss and accuracy for the epoch
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
            epoch_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            
            """# ── NEW: compute post-update accuracy ─────────────────────────────────────────
            true_train_acc = epoch_train_accuracy(
                model=model,
                loader=data_loader,             # same loader is fine; turn shuffle=False when creating it
                device=self.device,
                use_tabular_features=self.use_tabular_features,
            )"""
            print(f"Epoch [{epoch+1}/{self.config['training']['num_epochs']}] completed.")
            print(f"Average Loss (train batches): {avg_epoch_loss:.4f}")
            print(f"Accuracy  (running batches):  {epoch_accuracy:.4f}")
            #print(f"Accuracy  (full pass post-upd): {true_train_acc:.4f}")   # <- this is the trustworthy one
            
            # Update learning rate scheduler
            scheduler.step(avg_epoch_loss)
            
            # Memory profiling
            if self.config['training']['memory_monitoring'] and torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                print(f"Peak GPU memory this epoch: {peak_memory:.2f} GB")
                # Reset peak memory stats for the next epoch
                torch.cuda.reset_peak_memory_stats()
            
            # Save model checkpoint if it's the best so far
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                checkpoint_path = os.path.join(
                    self.checkpoints_dir, 
                    f'best_model_{self.model_type}_{self.aggregation_type}.pth'
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'accuracy': epoch_accuracy,
                    'config': self.config
                }, checkpoint_path)
                best_checkpoint_path = checkpoint_path
                print(f"New best model saved at: {checkpoint_path}")
            
            # Regular checkpoint save
            if (epoch + 1) % self.config['training']['checkpoint_interval'] == 0:
                checkpoint_path = os.path.join(
                    self.checkpoints_dir, 
                    f'checkpoint_epoch{epoch+1}.pth'
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'accuracy': epoch_accuracy,
                    'config': self.config
                }, checkpoint_path)
                print(f"Checkpoint saved at: {checkpoint_path}")
        
        print("Training completed!")
        print(f"Best loss: {best_loss:.4f}")
        
        return model, best_checkpoint_path