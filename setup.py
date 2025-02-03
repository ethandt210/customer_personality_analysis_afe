from setuptools import setup, find_packages

setup(
    name="clustering_afe",
    version="1.1.9",
    packages=find_packages(),
    description="A Python library for automated feature engineering tailored to clustering in customer personality analysis within the retail industry.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ethan Timadius",
    author_email="ethan.timadius@gmail.com",
    url="https://github.com/ethandt210/customer_personality_analysis_afe",
    license="LICENSE.txt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "openai>=1.55.3", # Faced issue with newer version
        "httpx>=0.27.2", # Faced issue with newer version
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
    ],
    extras_require={
        "full": ["autofeat", "featuretools", "tabpfn[full]"],
    },
)