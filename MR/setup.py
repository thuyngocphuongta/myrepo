from setuptools import find_packages, setup
import os
import io

def pip(filename):
    """Parse pip reqs file and transform it to setuptools requirements."""
    requirements = []
    for line in io.open(os.path.join('requirements', '{0}.pip'.format(filename))):
        line = line.strip()
        if not line or '://' in line or line.startswith('#'):
            continue
        requirements.append(line)
    return(requirements)

install_requires = pip('install')

setup(
    name="PyBox",
    description='Python package for the Box project',
    author="Box",
    author_email="",
    packages=['PyBox'],
    package_dir={'PyBox': 'PyBox'},
    install_requires=install_requires,
    version='0.0.6',
)



