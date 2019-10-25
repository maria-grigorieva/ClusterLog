import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ClusterLogs",
    version="0.0.2",
    author="Maria Grigorieva",
    author_email="magsend@gmail.com",
    description="Unsupervized Error Logs Clusterization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maria-grigorieva/ClusterLog.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)