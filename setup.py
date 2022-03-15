from setuptools import setup, find_packages
setup(
    name="xconf",
    version="0.0.1",
    description="Turns TOML files and command-line arguments into dataclasses for config",
    author="Joseph D. Long",
    author_email="jdl@zesty.space",
    install_requires=[
        "toml>=0.10.2"
    ],
    packages=find_packages(),
    long_description=open('./README.md').read(),
    long_description_content_type='text/markdown',
)
