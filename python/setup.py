from setuptools import setup  # pyright: ignore[reportMissingModuleSource]

_ = setup(
    extras_require={
        "llamaindex": ["llama-index-core>=0.10.0"],
        "crewai": ["crewai>=0.28.0"],
    }
)
