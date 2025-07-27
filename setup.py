from setuptools import setup, find_packages

setup(
    name="glucose-ai-assistant",
    version="1.0.0",
    description="LangChain-powered blood glucose analysis system",
    author="Miraj Uddin Chowdhury",
    author_email="mirazchowdhury03@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-groq>=0.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "streamlit>=1.28.0",
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0"
        ]
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "glucose-ai=cli_interface:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)