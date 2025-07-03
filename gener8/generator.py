import pandas as pd
import numpy as np
import torch
import logging
from typing import Tuple, Optional

class Generator:
    """Generates synthetic data using the trained neural network model."""
    
    def __init__(self, trainer: 'Trainer'):
        self.logger = logging.getLogger(__name__)
        self.trainer = trainer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def generate(self, n_samples: int, enable_flexible_generation: bool = True) -> pd.DataFrame:
        """
        Generate synthetic data with the same structure as the input.
        
        Args:
            n_samples: Number of samples to generate
            enable_flexible_generation: Allow dynamic sample size
            
        Returns:
            DataFrame containing synthetic data
        """
        if self.trainer.model_generator is None:
            raise ValueError("Model not trained. Run Trainer.train() first.")
                
        try:
            # Generate synthetic data
            latent_dim = self.trainer.model_generator.model[0].in_features
            if enable_flexible_generation:
                batch_size = min(n_samples, 1024)
                synthetic_samples = []
                remaining_samples = n_samples
                while remaining_samples > 0:
                    current_batch = min(batch_size, remaining_samples)
                    noise = torch.randn(current_batch, latent_dim).to(self.device)
                    with torch.no_grad():
                        batch_samples = self.trainer.model_generator(noise).cpu().numpy()
                    synthetic_samples.append(batch_samples)
                    remaining_samples -= current_batch
                synthetic_samples = np.concatenate(synthetic_samples, axis=0)
            else:
                noise = torch.randn(n_samples, latent_dim).to(self.device)
                with torch.no_grad():
                    synthetic_samples = self.trainer.model_generator(noise).cpu().numpy()
            
            # Split numeric and categorical features
            num_numeric_cols = len(self.trainer.columns)
            numeric_samples = synthetic_samples[:, :num_numeric_cols]
            categorical_samples = synthetic_samples[:, num_numeric_cols:] if synthetic_samples.shape[1] > num_numeric_cols else None
            
            # Inverse transform numeric data
            numeric_data = self.trainer.scaler.inverse_transform(numeric_samples)
            
            # Create DataFrame for numeric data
            synthetic_data = pd.DataFrame(numeric_data, columns=self.trainer.columns)
            
            # Decode categorical data
            if categorical_samples is not None and len(self.trainer.categorical_columns) > 0:
                for idx, col in enumerate(self.trainer.categorical_columns):
                    if col in self.trainer.label_encoders:
                        le = self.trainer.label_encoders[col]
                        cat_values = np.round(categorical_samples[:, idx]).astype(int)
                        cat_values = np.clip(cat_values, 0, len(le.classes_) - 1)
                        synthetic_data[col] = le.inverse_transform(cat_values)
            
            # Add missing values based on original proportions
            for col in synthetic_data.columns:
                missing_prop = self.trainer.missing_proportions.get(col, 0.0)
                if missing_prop > 0:
                    mask = np.random.random(n_samples) < missing_prop
                    synthetic_data.loc[mask, col] = np.nan
            
            # Ensure all original columns are present
            all_columns = list(self.trainer.data.columns) if self.trainer.data is not None else list(self.trainer.columns)
            synthetic_data = synthetic_data.reindex(columns=all_columns)
            
            # Restore original dtypes
            for col, dtype in self.trainer.dtypes.items():
                if col in synthetic_data.columns:
                    try:
                        synthetic_data[col] = synthetic_data[col].astype(dtype)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Could not convert column {col} to {dtype}: {str(e)}")
            
            return synthetic_data
                
        except Exception as e:
            self.logger.error(f"Error generating data: {str(e)}")
            raise