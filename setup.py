import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(
    name='psifr',
    version='0.4.3',
    description='Package for analysis of free recall data.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Neal Morton',
    author_email='mortonne@gmail.com',
    license='GPLv3',
    url='http://github.com/mortonne/psifr',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        'psifr': ['data/*.csv']
    },
    install_requires=[
        'numpy',
        'scipy',
        'pandas>=1.0.0',
        'matplotlib!=3.3.1',
        'seaborn>=0.9.1',
    ],
    extras_require={
        'docs': ['sphinx', 'pydata-sphinx-theme', 'ipython'],
        'test': ['pytest', 'codecov', 'pytest-cov'],
    },
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
    ]
)
