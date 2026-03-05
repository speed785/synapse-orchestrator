from setuptools import setup, find_packages

with open("../README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="synapse-orchestrator",
    version="0.1.0",
    description="Parallel tool call orchestrator for AI agents — auto-detect dependencies, maximise concurrency.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Synapse Contributors",
    url="https://github.com/speed785/synapse-orchestrator",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Core has zero mandatory runtime deps — pure stdlib (asyncio).
    ],
    extras_require={
        "openai": ["openai>=1.0"],
        "anthropic": ["anthropic>=0.25"],
        "langchain": ["langchain>=0.1.0"],
        "all": ["openai>=1.0", "anthropic>=0.25", "langchain>=0.1.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "pytest-cov>=4.1",
            "openai>=1.0",
            "anthropic>=0.25",
            "langchain>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
    keywords="ai agents llm tool-calls parallel orchestration asyncio openai anthropic",
)
