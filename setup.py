from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clipchat",
    version="0.1.0",
    author="Aditya Kumar",
    description="video rag cli tool for processing and querying video content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "development status :: 3 - alpha",
        "intended audience :: developers",
        "topic :: software development :: libraries :: python modules",
        "license :: osi approved :: mit license",
        "programming language :: python :: 3.8",
        "programming language :: python :: 3.9",
        "programming language :: python :: 3.10",
        "programming language :: python :: 3.11",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "clipchat=src.cli:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)