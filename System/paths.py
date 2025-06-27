from pathlib import Path


def check_path(path: str):

    if path != None:

        assert Path(path).is_dir() is True, 'path {} does not exist'.format(path)

        if path[-1] != '/':
            path += '/'

        return path
    
def check_if_file_exists(path: str):

    if path != None:

        if Path(path).is_dir() == True:

            pass

        else:

            assert Path(path).is_file() is True, 'file {} does not exist'.format(path)

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

    def set(self):
        """
        sets the remaining class attributes
        """

        for key in vars(self):

            if vars(self)[key] == None:

                vars(self)[key] = ''

            if ('dir' in key or 'path' in key) == True:

                if len(vars(self)[key]) != 0:

                    vars(self)[key] = check_path(vars(self)[key])



        self.mm_top = self.mm_top_file_path + self.mm_top_file
        self.mm_stream = self.mm_str_file_path + self.mm_str_file
        self.mm_crd = self.mm_crd_file_path + self.mm_crd_file
        self.mm_traj = self.mm_traj_file_path + self.mm_traj_file
        self.mm_nosol_crd = self.mm_nosol_crd_file_path + '/' + self.mm_nosol_crd_file
        self.mm_nosol_top = self.mm_nosol_top_file_path + '/' + self.mm_nosol_top_file
        self.mm_mol1_crd = self.mm_mol1_crd_file_path + self.mm_mol1_crd_file
        self.mm_mol1_top = self.mm_mol1_top_file_path + self.mm_mol1_top_file
        self.mm_mol2_crd = self.mm_mol2_crd_file_path + self.mm_mol2_crd_file
        self.mm_mol2_top = self.mm_mol2_top_file_path + self.mm_mol2_top_file


        for key in vars(self):
            
            if len(vars(self)[key]) == 0:

                vars(self)[key] = None
            

