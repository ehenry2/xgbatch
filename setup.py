import setuptools


setuptools.setup(
    name="xgbatch",
    version="0.0.2",
    author="Eric Henry",
    description="High Performance Serving for XGBoost",
    url="https://github.com/ehenry2/xgbatch",
    packages=["xgbatch"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pyarrow>=3.0.0",
        "xgboost>=1.0.0",
        "numpy>=1.0.0",
    ],
    python_requires=">=3.6",
)
