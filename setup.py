from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='transqinterp',
    version='0.1.0',
    description='Mechanistic intepretatibility tools and experiments for the transitive-property task',
    long_description=readme,
    author='Sachal Malick',
    author_email='sachalmalick@gmail.com',
    url='https://github.com/sachalmalick/transq-interp',
    license=license,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'experiment = interpexperiments.experiment:main',
        ]
    },
    install_requires=[
        'numpy',
        'torch',
        'transformer_lens',
        'evaluate',
        'transformers'
    ]
)
