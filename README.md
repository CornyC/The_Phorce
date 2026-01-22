# The Phorce - Flexible Force Matching for Parametrization

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#how-to-use-it">How to use it</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contacts">Contacts</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About
The_Phorce is a package to parametrize molecules for force fields using force matching. Any parameter type (bonded: bond, angle, dihedral, improper or nonbonded: charge, sigma, epsilon, NBFIX) or combination thereof can be selected. It is possible to utilize net forces between two solutes and only desired atoms of the input molecules can be selected to enter the parametrization process. OpenMM is used as a Molecular Dynamics engine and therefore all formats supported by OpenMM are supported in The_Phorce.  Input coordinates can be modified using an MDAnalysis interface. For quantum-chemical reference data, either cp2k output files from a HPC cluster can be processed, cp2k can be called directly if installed on the machine, or ASE can be used. A wide choice of different optimizers are provided with SciPy being the standard. Bayesian optimization is based on [BOSS](https://sites.utu.fi/boss/) and a stochastic optimization on [PyCMA](https://github.com/CMA-ES/pycma). The form of the objective function can be changed easily by the user thanks to modularity. The_Phorce has been successfully used to generate new nonbonded parameters of phosphorylated serine for the CHARMM36m force field (see our [publication](doi.org/...)). QM datasets for phosphorylated amino acid analogs are provided here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17966358.svg)](https://doi.org/10.5281/zenodo.17966358)


<!-- INSTALLATION -->
## Installation
Requires at least Python 3.8. Uses numpy, SciPy, MDAnalysis, and OpenMM. Optional are ASE, BoSS, and PyCMA. 

1. Clone the repo
   ```sh
   git clone https://github.com/CornyC/The_Phorce.git
   ``` 
2. Go into the directory
   ```sh
   cd The_Phorce
   ``` 
3. Install
   ```sh
   python setup.py install
   ```
<!-- HOW TO USE IT -->
## How to use it

The_Phorce_UI.ipynb will guide you through the process of parametrization using force matching in form of a jupyter notebook.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACTS -->
## Contacts

Viktoria Korn - viktoria.korn@simtech.uni-stuttgart.de

Kristyna Pluhackova - kristyna.pluhackova@simtech.uni-stuttgart.de

Project Link: [https://github.com/CornyC/The_Phorce.git](https://github.com/CornyC/The_Phorce.git)
