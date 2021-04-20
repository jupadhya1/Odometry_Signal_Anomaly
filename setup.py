from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='odometry',
    description='building a training pipeline having preprocessing and training',
    long_description=readme(),
    author='sunil',
    include_package_data=True,
    install_requires=required,
    url='https://alstom-smart-mobility@dev.azure.com/alstom-smart-mobility/Odometry/_git/odometry_classification',
    classifiers=[
        'Programming Language :: Python :: 3.6'
    ],
    version='0.1.0'
)