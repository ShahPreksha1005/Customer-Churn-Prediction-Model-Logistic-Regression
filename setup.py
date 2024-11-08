from setuptools import setup, find_packages

setup(
    name="Diabetes Prediction",
    version="1.0",
    author='Adhyayan',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "contourpy==1.2.0",
        "cycler==0.12.1",
        "fonttools==4.50.0",
        "importlib-resources==6.4.0",
        "kiwisolver==1.4.5",
        "matplotlib==3.8.3",
        "numpy==1.26.4",
        "packaging==24.0",
        "pandas==2.2.1",
        "pillow==10.2.0",
        "pyparsing==3.1.2",
        "python-dateutil==2.9.0.post0",
        "pytz==2024.1",
        "seaborn==0.13.2",
        "six==1.16.0",
        "tzdata==2024.1",
        "zipp==3.18.1"
    ],
)
