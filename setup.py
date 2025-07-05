from setuptools import setup, find_packages

setup(
    name="gener8-synth",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0,<2.0",
        "scikit-learn>=1.1.0",
        "torch>=1.13.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    author="Abdulrahman Abdulrahman",
    author_email="abdulrahamanbabatunde12@gmail.com",
    description="A synthetic data generation engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Abdulrahman0044/gener8",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)