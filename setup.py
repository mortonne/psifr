import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(
    name='psifr',
    version='0.1.0',
    description='Package for analysis of free recall data.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Neal Morton',
    author_email='mortonne@gmail.com',
    license='GPLv3',
    url='http://github.com/mortonne/psifr',
    packages=setuptools.find_packages(),
    package_data={
        'psifr': ['data/*.csv']
    },
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
    ],
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
    ]
)
