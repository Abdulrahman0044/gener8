# Gener8

**Gener8** is a Python-based synthetic data generation engine using neural networks. It loads data, trains a GAN-based model, and generates synthetic data that mimics the input's structure, including missing values.

---

## Features

- **Data Connector**: Load data from CSV, JSON, Excel, or pandas DataFrames.
- **Trainer**: Train a Gaussian Mixture Model to capture data distributions.
- **Generator**: Produce synthetic data with numerical and categorical columns.
- **Modular Design**: Each component is independent and reusable.
- **Pip Installable**: Easily install and integrate into your projects.

---

## Installation

To install Gener8, use pip:

```bash
pip install gener8-synth
```

Alternatively, to install from source:

```bash
# Clone the repository
git clone https://github.com/abdulrahman0044/gener8.git
cd gener8

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install .
```

---

## Usage

Below is an example of how to use Gener8 to generate synthetic data:

```python
import pandas as pd
import numpy as np
from gener8 import Gener8Engine

# Initialize the engine
engine = Gener8Engine()

# Create sample data
data = pd.DataFrame({
    'age': np.random.normal(30, 10, 1000),
    'income': np.random.normal(50000, 10000, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000)
})

# Load and train the model
engine.load_and_train(
    data,
    n_components=3,
    epochs=5,
    max_sample_size=10000,
    batch_size=32,
    gradient_accumulation_steps=2,
    max_training_time=3600,
    max_sequence_window=1,
    enable_flexible_generation=True,
    value_protection=True,
    rare_category_replacement_method="CONSTANT",
    differential_privacy=None)

# Generate synthetic data
synthetic_data = engine.generate(100)
print(synthetic_data.head())
```

---

## Requirements

- Python >= 3.8
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.1.0

---

## Project Structure

```
gener8/
├── gener8/
│   ├── __init__.py
│   ├── data_connector.py
│   ├── trainer.py
│   ├── generator.py
│   ├── engine.py
│   └── evaluation.py
├── setup.py
├── README.md
├── requirements.txt
├── LICENSE
├── MANIFEST.in
```

---

## Development

To contribute or modify Gener8:

```bash
# Clone the repository
git clone https://github.com/abdulrahman0044/gener8.git

# Install dependencies
pip install -r requirements.txt

# Make changes and test locally

# Build the package
python -m build

# Install locally
pip install dist/gener8-0.1.0-py3-none-any.whl
```

---

## License

This project is licensed under the **Apache 2.0 License**. See the `LICENSE` file for details.

---

## Contact

For questions or contributions, please open a pull request or a issue on the [GitHub repository](https://github.com/abdulrahman0044/gener8) or contact `abdulrahamanbabatunde12@gmail.com`.
