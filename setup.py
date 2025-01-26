from setuptools import setup, find_packages

setup(
    name="simple_nn",  # Library name
    version="0.1.0",  # Initial version
    description="A simple K-Nearest Neighbors (KNN) machine learning framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/radhofan/Gapred",  # Replace with your repository
    packages=find_packages(),
    install_requires=["numpy"],  # Dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
