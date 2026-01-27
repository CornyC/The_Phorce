from setuptools import setup

setup(name='The_Phorce',
      version='0.1',
      description='Flexible Force Matching for Parametrization',
      author='Viktoria Korn',
      author_email='viktoria.korn@simtech.uni-stuttgart.de',
      license='MIT',
      packages=['The_Phorce'],
      zip_safe=True,
      python_requires=">=3.8",
      install_requires=["numpy>=1.23.5","openmm>=8.0.0","mdanalysis>=2.4.3","scipy>=1.9.3","ase>=3.22.1"])
