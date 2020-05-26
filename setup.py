from setuptools import setup, find_packages

NAME = "dmtools"
PACKAGES = [NAME] + ["%s.%s" % (NAME, i) for i in find_packages(NAME)]

setup(
    name=NAME,
    version='0.0.1',
    author='leo lei',
    author_email='m18349125880@gmail.com',
    description='data mining tools.',
    packages=PACKAGES,

    install_requires=[
        'pandas == 1.0.0',
        'xgboost == 0.90',
        'seaborn == 0.10.0',
        'matplotlib == 3.1.2',
        'setuptools==45.1.0',
        'scipy == 1.4.1',
        'scikit_learn == 0.22.1',
    ],
)
