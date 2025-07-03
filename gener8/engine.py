import logging
from .data_connector import DataConnector
from .trainer import Trainer
from .generator import Generator
from typing import Union, Optional, Dict
import pandas as pd

class Gener8Engine:
    """Main engine for synthetic data generation with progress reporting."""
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.data_connector = DataConnector()
        self.trainer = Trainer()
        self.generator = Generator(self.trainer)
        self.logger.info("Gener8 engine initialized")
    
    def load_and_train(self, 
                       source: Union[str, pd.DataFrame], 
                       file_type: Optional[str] = None, 
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
        Load data and train the model with NN parameters.
        
        Args:
            source: Path to file or DataFrame
            file_type: Type of file (csv, json, excel)
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
        data = self.data_connector.load_data(source, file_type)
        self.trainer.train(
            data=data,
            n_components=n_components,
            epochs=epochs,
            max_sample_size=max_sample_size,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_training_time=max_training_time,
            max_epochs=max_epochs,
            max_sequence_window=max_sequence_window,
            enable_flexible_generation=enable_flexible_generation,
            value_protection=value_protection,
            rare_category_replacement_method=rare_category_replacement_method,
            differential_privacy=differential_privacy
        )
    
    def generate(self, n_samples: int, enable_flexible_generation: bool = True) -> pd.DataFrame:
        """
        Generate synthetic data.
        
        Args:
            n_samples: Number of samples to generate
            enable_flexible_generation: Allow dynamic sample size
            
        Returns:
            DataFrame containing synthetic data
        """
        return self.generator.generate(n_samples, enable_flexible_generation)

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Initialize engine
    engine = Gener8Engine()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'age': np.random.normal(30, 10, 1000),
        'income': np.random.normal(50000, 10000, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    sample_data.loc[np.random.choice(1000, 100), 'age'] = np.nan
    sample_data.loc[np.random.choice(1000, 50), 'category'] = np.nan
    
    # Load and train
    engine.load_and_train(
        sample_data,
        epochs=50,
        batch_size=32,
        max_sample_size=500,
        gradient_accumulation_steps=2
    )
    
    # Generate synthetic data
    synthetic_data = engine.generate(100)
    print(synthetic_data.head())