from setuptools import setup, find_packages

setup(
    name="gener8",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
    ],
    author="Abdulrahman Abdulrahman",
    author_email="abdulrahamanbabatunde12@gmail.com",
    description="A synthetic data generation engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/abdulrahman0044/gener8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)