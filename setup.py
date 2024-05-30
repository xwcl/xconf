from setuptools import setup, find_packages

extras = {
    'dev': ['pytest'],
    'contrib': ['fsspec', 'ray'],
}
all_deps = set()
for _, deps in extras.items():
    for dep in deps:
        all_deps.add(dep)
extras['all'] = list(all_deps)

setup(
    name="xconf",
    version="0.0.2",
    description="Turns TOML files and command-line arguments into dataclasses for config",
    author="Joseph D. Long",
    author_email="jdl@zesty.space",
    install_requires=[
        "toml>=0.10.2",
    ],
    extras_require=extras,
    packages=find_packages(),
    long_description=open('./README.md').read(),
    long_description_content_type='text/markdown',
)
