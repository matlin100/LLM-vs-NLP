from setuptools import setup, find_packages

setup(
    name="llm_vs_nlp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "openai",
        "transformers",
        "torch",
        "numpy",
        "scikit-learn",
    ],
) 