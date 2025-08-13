import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from seisLM.toto_head import TotoHead          # NEW import
 

class AttnPool(nn.Module):
    def __init__(self, d_in, d_hidden=128):
        super().__init__()
        self.W1 = nn.Linear(d_in, d_hidden)
        self.W2 = nn.Linear(d_hidden, 1)

    def forward(self, H):                 # H: [B, T, d_in]
        a = torch.tanh(self.W1(H))        # [B, T, d_hidden]
        α = torch.softmax(self.W2(a), 1)  # [B, T, 1]
        z = (α * H).sum(1)                # [B, d_in]
        return z
    
    
class SeisLMForMagnitudePrediction(nn.Module):
    """
    Model that processes windowed seismic data through SeisLM and aggregates
    window embeddings to predict earthquake magnitude.
    
    Now supports combining embeddings with tabular features.
    """
    def __init__(self, pretrained_model, aggregation_type="lstm", num_classes=4, 
                 use_tabular_features=False, tabular_features_dim=0, device=None, toto_cfg=None):
        super(SeisLMForMagnitudePrediction, self).__init__()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move the pretrained model to the chosen device
        self.seisLM = pretrained_model.to(self.device)
        self.seisLM.eval()
        for p in self.seisLM.parameters():
            p.requires_grad = False
        
        
        self.aggregation_type = aggregation_type
        self.use_tabular_features = use_tabular_features
        self.tabular_features_dim = tabular_features_dim
        self.num_classes = num_classes 
        
        # Frozen embedding dimension from seisLM
        self.embedding_dim = 256
        
        # Aggregation methods
        if aggregation_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
                bidirectional=True,
        
            ).to(self.device)
            self.pool = AttnPool(d_in=256).to(self.device)         # 128×2 → 256
            self.fc_input_dim = 256

            
            
        elif aggregation_type == "toto":
            self.toto = TotoHead(
            d_in=self.embedding_dim,                    # 256
            num_classes=self.num_classes,
            regime=toto_cfg.get("regime", "head"),
            pretrained_name=toto_cfg["pretrained_name"]).to(self.device)
            self.fc_input_dim = 0      # TotoHead returns final logits
    
    
        
        elif aggregation_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc_input_dim = self.embedding_dim
        
        elif aggregation_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.embedding_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            ).to(self.device)
            self.fc_input_dim = self.embedding_dim
        
        else:  # mean or max pooling
            self.fc_input_dim = self.embedding_dim
        
        # Tabular feature processing if enabled
        if self.use_tabular_features:
            # Projection layer for tabular features
            self.tabular_projection = nn.Sequential(
                nn.Linear(self.tabular_features_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 64),
                nn.ReLU()
            ).to(self.device)
            
            # Combined input dimension for final layers
            self.fc_input_dim += 64  # Add tabular feature dimension
        
        # Change the final layer to output class logits
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)  # Output num_classes instead of 1
        ).to(self.device)
    
    """@torch.no_grad()    
    def _process_window(self, window_cpu):
       
        #Process a single window through SeisLM.
        #- window_cpu: a CPU tensor of shape [batch_size, channels, window_length].
        
        # Move just this window to GPU (self.device)
        window_gpu = window_cpu.to(self.device, non_blocking=True)
        
        # If you're only using GPU, you can also do:
        # with torch.amp.autocast(device_type='cuda'):
        with torch.amp.autocast(self.device.type):
            # Print shapes for debugging if needed
            #print(f"_process_window received shape (CPU): {window_cpu.shape}")
            #print(f"  Transferred shape (GPU): {window_gpu.shape}")
            
            # Run through SeisLM => [batch_size, embedding_dim, sequence_length]
            with torch.no_grad():
                model_out = self.seisLM(window_gpu)
                embeddings_gpu = model_out.projected_states        # [B, seq, 256]
            #print(f"embeddings_gpu shape: {embeddings_gpu.shape}")
            # e.g. torch.Size([batch_size, seq_len, hidden_dim])

            # If you want [batch_size, embedding_dim, seq_len], do:
            embeddings_gpu = embeddings_gpu.permute(0, 2, 1)
            #print(f"after permute: {embeddings_gpu.shape}")
            # e.g. torch.Size([batch_size, hidden_dim, seq_len])

            # Now you can safely mean over dim=-1 (the last dimension)
            window_embedding_gpu = torch.mean(embeddings_gpu, dim=-1)
            #print(f"window_embedding_gpu shape: {window_embedding_gpu.shape}")
        
        # Move the embedding back to CPU to avoid storing all in GPU
        window_embedding_cpu = window_embedding_gpu.detach().cpu()
        return window_embedding_cpu"""
        
    @torch.no_grad()               # ← keep!
    def _process_window(self, window_cpu):
        """
        window_cpu : [B, C, L=86 400]  →  embeds : [B, 256]  (CPU, detached)
        """

        window_gpu = window_cpu.to(self.device, non_blocking=True)

        # ── 1️⃣  down-sample 86 400 → 8 640  (10-s mean) ───────────────────
        #kernel_size = stride = 10  → avg over every 10 samples
        #window_gpu = F.avg_pool1d(window_gpu, kernel_size=10, stride=10)

        #      B  C   8 640
        # ── 2️⃣  frozen SeisLM inference (cheap now) ──────────────────────
        emb = self.seisLM(window_gpu).projected_states.mean(dim=1)  # [B,256]

        # ── 3️⃣  back to CPU, detached ───────────────────────────────────
        return emb.cpu()

    def forward(self, x_cpu,tabular_features=None,return_embedding: bool = False):
        """
        Forward pass for windowed data with optional tabular features
        
        Args:
            x_cpu: CPU tensor of shape [batch_size, num_windows, channels, window_length]
            tabular_features: Optional tensor of shape [batch_size, tabular_features_dim]
            
        Returns:
            Tensor of shape [batch_size, num_classes] with class logits
        """

        #print(f"forward() input shape (CPU): {x_cpu.shape}")
        batch_size, num_windows, channels, window_length = x_cpu.shape
        
        # Process each window individually on GPU, then return embedding to CPU
        #print("Prcoessing windows started")
        window_embeddings_cpu = []
        for i in range(num_windows):
            # [batch_size, channels, window_length]
            window_cpu = x_cpu[:, i, :, :]  
            embedding_cpu = self._process_window(window_cpu)
            # embedding_cpu => [batch_size, embedding_dim] on CPU
            window_embeddings_cpu.append(embedding_cpu)
        
        # Stack all CPU embeddings => [batch_size, num_windows, embedding_dim]
        stacked_embeddings_cpu = torch.stack(window_embeddings_cpu, dim=1)
        
        #print(f"stacked_embeddings shape (CPU): {stacked_embeddings_cpu.shape}")
        
        # Move embeddings to GPU for aggregationattention pooling lstm output
        #stacked_embeddings_gpu = stacked_embeddings_cpu.to(self.device)
        stacked_embeddings_gpu = stacked_embeddings_cpu.to(self.device, dtype=torch.float32)
        
        
        # Now apply aggregator on GPU
        if self.aggregation_type == "lstm":
            lstm_out, _ = self.lstm(stacked_embeddings_gpu)          # [B, T, 256]
            #print("attention pooling lstm output")
            aggregated_gpu    = self.pool(lstm_out)   
            #lstm_out_gpu, _ = self.lstm(stacked_embeddings_gpu)
            
            #print(f"  LSTM output shape (GPU): {lstm_out_gpu.shape}")
            # Use last time step => [batch_size, hidden_size*2]
            #aggregated_gpu = lstm_out_gpu[:, -1, :]
            
        elif self.aggregation_type == "toto":
            
            #print(">>> stacked_embeddings_gpu.shape:", stacked_embeddings_gpu.shape)
            
            logits, pooled = self.toto(stacked_embeddings_gpu, return_pooled=True)
            if self.use_tabular_features:
                tab = self.tabular_projection(tabular_features.to(self.device))
                logits = self.fc_layers(torch.cat([logits, tab], dim=1))
            if return_embedding:
                return logits, pooled.detach()
            return logits

        elif self.aggregation_type == "transformer":
            transformer_out_gpu = self.transformer(stacked_embeddings_gpu)
            #print(f"  Transformer output shape (GPU): {transformer_out_gpu.shape}")
            aggregated_gpu = torch.mean(transformer_out_gpu, dim=1)
        elif self.aggregation_type == "mean":
            aggregated_gpu = torch.mean(stacked_embeddings_gpu, dim=1)
        elif self.aggregation_type == "max":
            aggregated_gpu, _ = torch.max(stacked_embeddings_gpu, dim=1)
        elif self.aggregation_type == "attention":
            attention_scores_gpu = self.attention(stacked_embeddings_gpu)  # [batch_size, num_windows, 1]
            attention_weights_gpu = F.softmax(attention_scores_gpu, dim=1)
            aggregated_gpu = torch.sum(stacked_embeddings_gpu * attention_weights_gpu, dim=1)
        
        # Process tabular features if provided
        if self.use_tabular_features and tabular_features is not None:
            # Move tabular features to device
            tabular_features_gpu = tabular_features.to(self.device)
            
            # Process tabular features
            tabular_embedding = self.tabular_projection(tabular_features_gpu)
            
            # Combine with seismic embeddings
            combined_features = torch.cat([aggregated_gpu, tabular_embedding], dim=1)
            
            # Pass through final layers
            output_gpu = self.fc_layers(combined_features)
        else:
            # Use only seismic embeddings
            output_gpu = self.fc_layers(aggregated_gpu)
        
        if return_embedding:
            return output_gpu, aggregated_gpu.detach()
        return output_gpu
            
            


class ScatterEarthquakePredictor(nn.Module):
    """
    Alternative model using wavelet scattering transform for feature extraction.
    Now supports combining with tabular features.
    """
    def __init__(self, 
                 window_length=21600,  # Default for 1-day windows at 1Hz after decimation
                 J=6,
                 Q=8,
                 aggregation_type="lstm",
                 hidden_dim=128,
                 dropout=0.3,
                 channels=3,
                 use_tabular_features=False,
                 tabular_features_dim=0,
                 num_classes=4,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        
        # Import here to avoid issues if not installed
        from kymatio.torch import Scattering1D
        
        self.window_length = window_length
        self.channels = channels
        self.aggregation_type = aggregation_type
        self.use_tabular_features = use_tabular_features
        self.tabular_features_dim = tabular_features_dim
        
        # Scattering transform for feature extraction
        self.scattering = Scattering1D(J=J, shape=window_length, Q=Q).to(device)
        
        # Calculate scattering output dimension
        with torch.no_grad():
            dummy = torch.randn(1, window_length).to(device)
            scat_dummy = self.scattering(dummy)
            scat_out_dim = scat_dummy.shape[1]
        
        self.feature_dim = scat_out_dim * channels
        
        # Embedding projection layer
        self.projection = nn.Linear(self.feature_dim, hidden_dim)
        
        # Aggregation layers with same architecture as the SeisLM model
        if aggregation_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim//2,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
                bidirectional=True
            )
            self.fc_input_dim = hidden_dim
        
        elif aggregation_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim*2,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc_input_dim = hidden_dim
            
        elif aggregation_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            self.fc_input_dim = hidden_dim
            
        else:  # mean or max pooling
            self.fc_input_dim = hidden_dim
        
        # Tabular feature processing if enabled
        if self.use_tabular_features:
            # Projection layer for tabular features
            self.tabular_projection = nn.Sequential(
                nn.Linear(self.tabular_features_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            
            # Combined input dimension for final layers
            self.fc_input_dim += 64  # Add tabular feature dimension
        
        # Final prediction layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
    def _process_window(self, x):
        """Process a single window through scattering transform"""
        # x shape: [batch_size, channels, window_length]
        batch_size = x.shape[0]
        
        # Process each channel separately
        channel_features = []
        for c in range(self.channels):
            x_c = x[:, c, :]  # [batch_size, window_length]
            with torch.no_grad():  # No need to track gradients for scattering
                scat_out = self.scattering(x_c)  # [batch_size, scat_out_dim]
            channel_features.append(scat_out)
        
        # Combine features from all channels
        combined = torch.cat(channel_features, dim=1)  # [batch_size, feature_dim] 
        
        # Project to hidden dimension
        window_embedding = self.projection(combined)  # [batch_size, hidden_dim]
        
        return window_embedding
        
    def forward(self, x, tabular_features=None):
        """
        Forward pass for windowed data with optional tabular features
        
        Args:
            x: Input tensor of shape [batch_size, num_windows, channels, window_length]
            tabular_features: Optional tensor of shape [batch_size, tabular_features_dim]
            
        Returns:
            Tensor of shape [batch_size, num_classes] with class predictions
        """
        batch_size, num_windows, channels, window_length = x.shape
        
        # Process each window individually
        window_embeddings = []
        for i in range(num_windows):
            # Extract current window for all samples in batch
            window = x[:, i, :, :]  # [batch_size, channels, window_length]
            embedding = self._process_window(window)  # [batch_size, hidden_dim]
            window_embeddings.append(embedding)
        
        # Stack embeddings from all windows
        # [batch_size, num_windows, hidden_dim]
        stacked_embeddings = torch.stack(window_embeddings, dim=1)
        
        # Aggregate window embeddings based on selected method
        if self.aggregation_type == "lstm":
            lstm_out, _ = self.lstm(stacked_embeddings)
            # Use the output from the last time step
            aggregated = lstm_out[:, -1, :]
            
        elif self.aggregation_type == "transformer":
            transformer_out = self.transformer(stacked_embeddings)
            # Average over all time steps
            aggregated = torch.mean(transformer_out, dim=1)
            
        elif self.aggregation_type == "mean":
            # Simple mean pooling
            aggregated = torch.mean(stacked_embeddings, dim=1)
            
        elif self.aggregation_type == "max":
            # Simple max pooling
            aggregated, _ = torch.max(stacked_embeddings, dim=1)
            
        elif self.aggregation_type == "attention":
            # Attention-based pooling
            # Calculate attention scores
            attention_scores = self.attention(stacked_embeddings)  # [batch_size, num_windows, 1]
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Apply attention weights
            aggregated = torch.sum(stacked_embeddings * attention_weights, dim=1)
        
        # Process tabular features if provided
        if self.use_tabular_features and tabular_features is not None:
            # Process tabular features
            tabular_embedding = self.tabular_projection(tabular_features)
            
            # Combine with seismic embeddings
            combined_features = torch.cat([aggregated, tabular_embedding], dim=1)
            
            # Final prediction
            output = self.fc_layers(combined_features)
        else:
            # Use only seismic embeddings
            output = self.fc_layers(aggregated)
        
        return output  # [batch_size, num_classes]