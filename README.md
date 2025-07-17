# sterile-dm-lfa: Sterile neutrino dark matter with lepton flavor asymmetries

The code "sterile-dm-lfa" has been developed by Kensuke Akita and Maksym Ovchynnikov to trace the evolution of sterile neutrino dark matter in presence of primordial lepton flavor asymmetries. The underlying physics and thechnical details are described in the associated preprint []. If you use this code, please cite this reference.

## The code structure

The code contains the following main scripts in Python:

Solver_SterileDM.ipynb: is a sample script in the Jupyter Notebook that solves the system of the sterile neutrino production and output the sterile neutrino abundance, its momentum distribution in the current Universe and the evolution of lepton flavor asymmetries mixing with sterile neutrinos.

Sterile_Nu_Prameters.py: fixes sterile neutrino mass and mixing angle with active neutrinos, and initial values of lepton flavor asymmetries. In the current version, any flavor space in lepton flavor asymmetries can be taken. On the other hand, sterile neutrinos can mix with only one flavor active neutrinos. 

System_SterileNuDM.py: describes the evolution equations for sterile neutrinos, the plasma temperature, and lepton flavor asymmetry mixing with sterile neutrinos.

Asy_Redistribution.py: describes the evolution equations for the redistribution of each particle-antiparticle asymmetry.

Make_Rate_Table.py: is a script to make a table of the active neutrino interaction rate in temperature (T) and its momentum (y=p/T). This table is an input for System_SterileNuDM.py.

Setup_Grids.py: fixes the numbers and ranges of grids, such as the momentum of sterile neutrinos. In particular, the number of grids for the momentum of sterile neutrinos must be very large for numerical convergence. The default is 10001, but the larger value is required for very large lepton asymmetries.  

## Limitations

Details of the limitation of this code are described in the main text of [].

