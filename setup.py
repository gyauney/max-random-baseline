import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="max-random-baseline",
    version="0.1.1",
    author="Gregory Yauney",
    author_email="gyauney@cs.cornell.edu",
    description="A simple random baseline that accounts for evaluation set size and reuse",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gyauney/max-random-baseline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=['numpy']
)