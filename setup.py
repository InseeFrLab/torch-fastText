from setuptools import find_packages, setup

setup(
    name="torchFastText",
    version="0.0.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,  # Ensure additional files like README.md are included
    install_requires=["pytorch_lightning", "captum", "unidecode", "nltk", "scikit-learn"],
)
