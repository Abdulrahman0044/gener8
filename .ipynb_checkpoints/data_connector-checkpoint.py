import pandas as pd
import numpy as np
import logging
from typing import Union, Optional

class DataConnector:
    """Handles data ingestion from various sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, source: Union[str, pd.DataFrame], 
                  file_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from file or DataFrame.
        
        Args:
            source: Path to file or DataFrame
            file_type: Type of file (csv, json, excel)
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            if isinstance(source, pd.DataFrame):
                return source.copy()
            
            if file_type is None and isinstance(source, str):
                file_type = source.split('.')[-1].lower()
            
            if file_type == 'csv':
                return pd.read_csv(source)
            elif file_type == 'json':
                return pd.read_json(source)
            elif file_type == 'excel':
                return pd.read_excel(source)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise