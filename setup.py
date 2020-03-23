from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='psifr',
      version='0.1.0',
      description='Package for analysis of free recall data.',
      long_description=readme(),
      classifiers=[
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 3.8',
      ],
      url='http://github.com/mortonne/psifr',
      author='Neal Morton',
      author_email='mortonne@gmail.com',
      license='GPLv3',
      requires=['numpy', 'pandas', 'seaborn'],
      include_package_data=True,
      zip_safe=False)
