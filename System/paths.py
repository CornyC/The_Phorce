from pathlib import Path

def check_path(path: str):
    """
    Check if the given path is a valid directory.

    Parameters
    ----------
    path : str
        Path to be checked.

    Raises
    ------
    AssertionError
        If the path is not a valid directory.
    """
    if path is not None:
        path = Path(path)
        assert path.is_dir(), f'Path {path} does not exist or is not a directory.'

#### instance to store input file paths and names ####

class Paths:
    """
    Stores paths and filenames to external input files. Parameters can also be set manually by the user.

    Parameters
    ----------
    working_dir : str (path)
        working directory
    project_name : str
        name of the project, e.g. 'arginine'

    mm_traj_file : str
        sampled trajectory filename w/ extension (any format that MDAnalysis can read), e.g. 'coords.xyz'
    mm_traj_file_path : str (path)
        path to trajectory containing sampled coordinates
    mm_crd_file : str
        coordinate file name
    mm_crd_file_path : str (path)
        path to coordinate file
    mm_top_file : str
        e.g. charmm psf file name
    mm_top_file_path : str (path)
        path to the psf file
    mm_str_file : str
        charmm parameter stream file name
    mm_str_file_path : str (path)
        path to parameter stream file
    -"-nosol-"-
        if System.system.Molecular_system.system_type == '1_solvent': The molecule w/o the sol-
        vent  
    -"-mol1-"-  
        if System.system.Molecular_system.system_type == '2_...': The first molecule (w/o sol-
        vent)
    -"-mol2-"-  
        if System.system.Molecular_system.system_type == '2_...': The 2nd molecule (w/o solvent)

    """

    def __init__(self):

        # general
        self.working_dir = None
        self.project_name = None

        # md-specific
        self.mm_traj_file = None
        self.mm_traj_file_path = None
        self.mm_crd_file = None
        self.mm_crd_file_path = None
        self.mm_top_file = None
        self.mm_top_file_path = None
        self.mm_str_file = None
        self.mm_str_file_path = None

        # optional
        self.mm_nosol_crd_file = None
        self.mm_nosol_crd_file_path = None
        self.mm_nosol_top_file = None
        self.mm_nosol_top_file_path = None
        self.mm_mol1_crd_file = None
        self.mm_mol1_crd_file_path = None
        self.mm_mol1_top_file = None
        self.mm_mol1_top_file_path = None
        self.mm_mol2_crd_file = None
        self.mm_mol2_crd_file_path = None
        self.mm_mol2_top_file = None
        self.mm_mol2_top_file_path = None

    def decide_on_reference_data(self, decision):
        if decision == 'read-in all':
            self.read_in_all_data()
        elif decision == 'compute all':
            self.compute_all_data()
        elif decision == 'hybrid':
            self.hybrid_approach()
        else:
            raise ValueError("Invalid decision. Please choose 'read-in all', 'compute all', or 'hybrid'.")

    def read_in_all_data(self):
        # logic check
        self.mm_traj_file_path = 'path/to/traj/'

    def compute_all_data(self):
        self.mm_crd_file_path = 'path/to/crd/'

    def hybrid_approach(self):
        # some other approach to compute, maybe?
        self.mm_nosol_crd_file_path = 'path/to/nosol_crd/'
        self.compute_all_data()

    def set(self):
        """
        sets the remaining class attributes
        """
        file_attributes = [
            'mm_top', 'mm_stream', 'mm_crd', 'mm_traj',
            'mm_nosol_crd', 'mm_nosol_top', 'mm_mol1_crd', 'mm_mol1_top',
            'mm_mol2_crd', 'mm_mol2_top'
        ]

        # iterate over file attributes, set paths, and check them
        for attribute in file_attributes:
            file_path_attribute = f'{attribute}_file_path'
            file_attribute = f'{attribute}_file'

            # combine paths and filenames
            setattr(self, attribute, getattr(self, file_path_attribute) + getattr(self, file_attribute))

            # check the path
            check_path(getattr(self, file_path_attribute))
            
        for filepath in [self.mm_top_file_path, self.mm_str_file_path, self.mm_crd_file_path, 
                         self.mm_traj_file_path, self.mm_nosol_crd_file_path, self.mm_nosol_top_file_path,
                         self.mm_mol1_crd_file_path, self.mm_mol1_top_file_path, 
                         self.mm_mol2_crd_file_path, self.mm_mol2_top_file_path]:
            if filepath != None:
                check_path(filepath)

        self.mm_top = self.mm_top_file_path + self.mm_top_file
        self.mm_stream = self.mm_str_file_path + self.mm_str_file
        self.mm_crd = self.mm_crd_file_path + self.mm_crd_file
        self.mm_traj = self.mm_traj_file_path + self.mm_traj_file
        self.mm_nosol_crd = self.mm_nosol_crd_file_path + self.mm_nosol_crd_file
        self.mm_nosol_top = self.mm_nosol_top_file_path + self.mm_nosol_top_file
        self.mm_mol1_crd = self.mm_mol1_crd_file_path + self.mm_mol1_crd_file
        self.mm_mol1_top = self.mm_mol1_top_file_path + self.mm_mol1_top_file
        self.mm_mol2_crd = self.mm_mol2_crd_file_path + self.mm_mol2_crd_file
        self.mm_mol2_top = self.mm_mol2_top_file_path + self.mm_mol2_top_file