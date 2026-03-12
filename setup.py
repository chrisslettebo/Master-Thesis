from setuptools import setup, find_packages

setup(
    name="masterthesis",
    version="0.1",
    package_dir={"": "Src"},          # ⭐ THIS IS THE KEY LINE
    packages=find_packages("Src"),    # ⭐ tell setuptools where packages live
)
