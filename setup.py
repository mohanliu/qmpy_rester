from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name="qmpy_rester",
      version='0.2.0',
      description="A python wrapper for OQMD API",
      url="https://github.com/mohanliu/qmpy_rester",
      author="Mohan Liu",
      author_email="mohan@u.northwestern.edu",
      license="LICENSE",
      packages=['qmpy_rester'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=["requests"],
      )
