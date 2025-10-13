from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="modular-rag-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive RAG system with file upload and web-accessible links",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/modular-rag-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=5.0.0"],
        "full": ["chromadb>=0.4.0", "sentence-transformers>=2.2.0", "faiss-cpu>=1.7.0"],
    },
    entry_points={
        "console_scripts": [
            "rag-launcher=rag_launcher:main",
            "rag-system=enhanced_rag_system:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
