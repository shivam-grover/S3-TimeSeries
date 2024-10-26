from setuptools import setup, find_packages

# Reading the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='s3-timeseries',  # Package name on PyPI
    version='0.1.0',  # Initial release version
    author='Shivam Grover',  # Your name
    author_email='shivam.grover@queensu.com',  # Your email
    description='Official implementation of "Segment, Shuffle, and Stitch: A Simple Mechanism for Improving Time-Series Representations"',
    long_description=long_description,  # From the README.md
    long_description_content_type='text/markdown',  # Format of README.md
    url='https://github.com/shivam-grover/S3-TimeSeries',  # Link to the project on GitHub (if available)
    packages=find_packages(include=['S3', 'S3.*']),
    install_requires=[
        'torch>=1.8.0',  # Dependencies
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version
)
