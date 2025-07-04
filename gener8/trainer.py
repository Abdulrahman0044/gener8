import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Optional
import logging
import time

class GeneratorNN(nn.Module):
    """Neural network for generating synthetic data."""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(GeneratorNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class DiscriminatorNN(nn.Module):
    """Neural network for discriminating real vs. synthetic data."""
    def __init__(self, input_dim, hidden_dim=128):
        super(DiscriminatorNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class Trainer:
    """Trains a GAN-based model on the input data, handling missing values and showing progress."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model_generator = None
        self.model_discriminator = None
        self.columns = None
        self.categorical_columns = None
        self.data = None
        self.missing_proportions = None
        self.dtypes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def preprocess_data(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training by dropping rows with NaN and encoding.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (training data, validation data) as numpy arrays
        """
        # Identify numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in data")
            
        self.columns = numeric_cols
        self.categorical_columns = categorical_cols
        self.dtypes = data.dtypes.to_dict()
        self.missing_proportions = data.isna().mean().to_dict()
        
        # Create a deep copy and drop rows with NaN
        clean_data = data.copy(deep=True).dropna()
        if len(clean_data) == 0:
            raise ValueError("No complete cases in data")
        
        # Encode categorical columns
        categorical_encoded = []
        for col in categorical_cols:
            le = LabelEncoder()
            # Convert to string and encode, ensuring numeric output
            clean_data.loc[:, col] = clean_data.loc[:, col].astype(str)
            clean_data.loc[:, col] = le.fit_transform(clean_data.loc[:, col]).astype(np.int64)
            self.label_encoders[col] = le
            categorical_encoded.append(clean_data[col].values.reshape(-1, 1))
        
        # Scale numeric data
        numeric_scaled = self.scaler.fit_transform(clean_data[numeric_cols]).astype(np.float32)
        
        # Combine numeric and categorical data
        processed_data = numeric_scaled
        if categorical_encoded:
            categorical_array = np.hstack(categorical_encoded).astype(np.float32)
            processed_data = np.hstack([numeric_scaled, categorical_array])
        
        # Ensure processed_data is float32 for PyTorch
        processed_data = processed_data.astype(np.float32)
        
        # Split into training and validation sets
        train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)
        
        return train_data, val_data
    
    def train(self, 
              data: pd.DataFrame, 
              n_components: int = 3,
              epochs: int = 100,
              max_sample_size: int = 100000,
              batch_size: int = 64,
              gradient_accumulation_steps: int = 1,
              max_training_time: Optional[float] = None,
              max_epochs: Optional[int] = None,
              max_sequence_window: int = 1,
              enable_flexible_generation: bool = True,
              value_protection: bool = True,
              rare_category_replacement_method: str = "CONSTANT",
              differential_privacy: Optional[Dict] = None) -> None:
        """
        Train GAN model on complete cases with progress reporting.
        
        Args:
            data: Input DataFrame
            n_components: Number of components for latent dim
            epochs: Number of training epochs
            max_sample_size: Maximum samples for training
            batch_size: Batch size for training
            gradient_accumulation_steps: Steps for gradient accumulation
            max_training_time: Maximum training time in seconds
            max_epochs: Maximum epochs
            max_sequence_window: Window size for sequential data
            enable_flexible_generation: Allow dynamic sample size
            value_protection: Prevent exact value replication
            rare_category_replacement_method: Method for rare categories
            differential_privacy: Differential privacy parameters
        """
        try:
            self.data = data
            train_data, val_data = self.preprocess_data(data)
            
            # Limit sample size
            if train_data.shape[0] > max_sample_size:
                indices = np.random.choice(train_data.shape[0], max_sample_size, replace=False)
                train_data = train_data[indices]
            
            if train_data.shape[0] < batch_size:
                raise ValueError(f"Too few complete cases ({train_data.shape[0]}) for batch_size ({batch_size})")
            
            # Initialize models
            input_dim = train_data.shape[1]
            latent_dim = n_components * max_sequence_window
            self.model_generator = GeneratorNN(latent_dim, input_dim).to(self.device)
            self.model_discriminator = DiscriminatorNN(input_dim).to(self.device)
            
            # Optimizers
            g_optimizer = optim.Adam(self.model_generator.parameters(), lr=0.0002)
            d_optimizer = optim.Adam(self.model_discriminator.parameters(), lr=0.0002)
            
            # Loss function
            criterion = nn.BCELoss()
            
            # Handle rare categories
            if rare_category_replacement_method == "CONSTANT":
                for col in self.categorical_columns:
                    if col in self.label_encoders:
                        value_counts = data[col].value_counts()
                        rare_mask = value_counts < 5
                        if rare_mask.any():
                            data.loc[data[col].isin(value_counts[rare_mask].index), col] = "RARE"
            
            # Training loop
            start_time = time.time()
            max_epochs = max_epochs or epochs
            train_dataset = torch.tensor(train_data, dtype=torch.float32).to(self.device)
            val_dataset = torch.tensor(val_data, dtype=torch.float32).to(self.device)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
            
            for epoch in range(min(epochs, max_epochs)):
                self.model_generator.train()
                self.model_discriminator.train()
                g_loss_avg, d_loss_avg = 0.0, 0.0
                for i, real_data in enumerate(train_loader):
                    batch_size = real_data.size(0)
                    
                    # Train discriminator
                    d_optimizer.zero_grad()
                    real_labels = torch.ones(batch_size, 1).to(self.device)
                    fake_labels = torch.zeros(batch_size, 1).to(self.device)
                    
                    # Real data
                    d_real = self.model_discriminator(real_data)
                    d_loss_real = criterion(d_real, real_labels)
                    
                    # Fake data
                    noise = torch.randn(batch_size, latent_dim).to(self.device)
                    fake_data = self.model_generator(noise)
                    d_fake = self.model_discriminator(fake_data.detach())
                    d_loss_fake = criterion(d_fake, fake_labels)
                    
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    d_loss.backward()
                    d_optimizer.step()
                    
                    # Train generator
                    g_optimizer.zero_grad()
                    fake_output = self.model_discriminator(fake_data)
                    g_loss = criterion(fake_output, real_labels)
                    g_loss.backward()
                    g_optimizer.step()
                    
                    g_loss_avg += g_loss.item()
                    d_loss_avg += d_loss.item()
                    
                    # Gradient accumulation
                    if (i + 1) % gradient_accumulation_steps == 0:
                        g_optimizer.step()
                        d_optimizer.step()
                        g_optimizer.zero_grad()
                        d_optimizer.zero_grad()
                
                # Validation loss
                self.model_generator.eval()
                self.model_discriminator.eval()
                val_g_loss, val_d_loss = 0.0, 0.0
                with torch.no_grad():
                    for real_data in val_loader:
                        batch_size = real_data.size(0)
                        real_labels = torch.ones(batch_size, 1).to(self.device)
                        fake_labels = torch.zeros(batch_size, 1).to(self.device)
                        
                        # Discriminator validation loss
                        d_real = self.model_discriminator(real_data)
                        d_loss_real = criterion(d_real, real_labels)
                        noise = torch.randn(batch_size, latent_dim).to(self.device)
                        fake_data = self.model_generator(noise)
                        d_fake = self.model_discriminator(fake_data)
                        d_loss_fake = criterion(d_fake, fake_labels)
                        val_d_loss += (d_loss_real + d_loss_fake).item() / 2
                        
                        # Generator validation loss
                        fake_output = self.model_discriminator(fake_data)
                        val_g_loss += criterion(fake_output, real_labels).item()
                
                # Log progress
                self.logger.info(
                    f"Epoch {epoch+1}/{min(epochs, max_epochs)}, "
                    f"Train G Loss: {g_loss_avg/len(train_loader):.4f}, "
                    f"Train D Loss: {d_loss_avg/len(train_loader):.4f}, "
                    f"Val G Loss: {val_g_loss/len(val_loader):.4f}, "
                    f"Val D Loss: {val_d_loss/len(val_loader):.4f}"
                )
                
                # Check time limit
                if max_training_time and (time.time() - start_time) > max_training_time:
                    self.logger.info("Max training time reached")
                    break
            
            # Save model state
            torch.save(self.model_generator.state_dict(), "generator.pth")
            torch.save(self.model_discriminator.state_dict(), "discriminator.pth")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise