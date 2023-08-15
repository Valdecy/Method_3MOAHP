from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='method_3mo_ahp',
    version='1.1.2',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/Method_3MOAHP',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'plotly',
        'matplotlib',
        'scipy'
    ],
    zip_safe=True,
    description='A Method to Reduce AHP Inconsistency According a Set of Quality Measures',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
