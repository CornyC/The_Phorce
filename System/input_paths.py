from pathlib import Path

def check_path(path: str):
    if path != None:

        assert Path(path).is_dir() is True, 'path {} does not exist'.format(path)

        if path[-1] != '/':
            path += '/'

#### instance to store input file paths and names ####

class Input_Paths:
    """
    Stores paths and filenames to external input files

    Parameters
    ----------
    mm_top_file : str
        e.g. charmm psf file name
    mm_top_file_path : str
        path to the psf file
    str_file : str
        charmm parameter stream file name
    str_file_path : str
        path to parameter stream file
    crd_pdb_file : str
        pdb file name
    crd_pdb_file_path : str
        path to pdb file
    traj_file : str
        sampled trajectory filename (any format that MDAnalysis can read)
    traj_file_path : str
        path to trajectory
    working_dir : str
        working directory
    """

    def __init__(self):

        self.mm_top_file = None
        self.mm_top_file_path = None
        self.str_file = None
        self.str_file_path = None
        self.crd_pdb_file = None
        self.crd_pdb_file_path = None
        self.traj_file = None
        self.traj_file_path = None
        self.working_dir = None


    def set(self):

        for filepath in [self.mm_top_file_path, self.str_file_path, self.crd_pdb_file_path, self.traj_file_path]:
            if filepath != None:
                check_path(filepath)

        self.mm_top = self.mm_top_file_path + self.mm_top_file
        self.stream = self.str_file_path + self.str_file
        self.crd_pdb = self.crd_pdb_file_path + self.crd_pdb_file
        self.traj = self.traj_file_path + self.traj_file