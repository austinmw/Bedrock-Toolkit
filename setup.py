""" Setup file for the bedrock_toolkit package. """
from setuptools import setup, find_packages

setup(
    name="bedrock_toolkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'loguru',
        'boto3',
        'pydantic',
        'rich',
        'streamlit',
        'requests',
        'Wikipedia-API',
        'tavily-python',
        'types-setuptools',
        'types-requests',
    ],
)
