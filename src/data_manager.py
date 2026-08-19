import h5py
import numpy as np
import os

class BBODataManager:
    def __init__(self, h5_filename):
        """
        Initialize the DataManager with the path to the HDF5 file.
        """
        self.h5_filename = h5_filename
        self.dims = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}

    def initialize_from_npy(self, base_dir):
        """
        Reads existing .npy files and initializes the HDF5 file, enabling dynamic appending.
        Run this once to create the consolidated dataset.
        """
        with h5py.File(self.h5_filename, 'w') as h5f:
            for func_idx, d in self.dims.items():
                f_dir = os.path.join(base_dir, f"function_{func_idx}")
                
                if not os.path.exists(f_dir):
                    print(f"Warning: Skipping function {func_idx}, directory not found at {f_dir}.")
                    continue
                    
                X = np.load(os.path.join(f_dir, "initial_inputs.npy"))
                y = np.load(os.path.join(f_dir, "initial_outputs.npy"))
                
                # Create a group for this specific function
                group = h5f.create_group(f"function_{func_idx}")
                
                # Create dataset for X (feature vectors)
                # maxshape=(None, d) allows appending rows indefinitely while preserving feature dimension
                group.create_dataset("X", data=X, maxshape=(None, d), chunks=True)
                
                # Create dataset for y (outputs)
                group.create_dataset("y", data=y, maxshape=(None,), chunks=True)
                
        print(f"Successfully consolidated initialization data into {self.h5_filename}")

    def append_data(self, func_idx, new_x, new_y):
        """
        Appends new evaluated point(s) to the HDF5 file for a specific function.
        
        Args:
            func_idx (int): The function index (1-8)
            new_x (numpy.ndarray): The new input point(s) to append. Can be 1D or 2D.
            new_y (float or numpy.ndarray): The new output value(s) to append.
        """
        with h5py.File(self.h5_filename, 'a') as h5f:
            group = h5f[f"function_{func_idx}"]
            dset_X = group["X"]
            dset_y = group["y"]
            
            current_rows = dset_X.shape[0]
            
            # Ensure correct dimensions for stacking
            new_x = np.atleast_2d(new_x)
            new_y = np.atleast_1d(new_y)
            num_new_rows = new_x.shape[0]
            
            # Resize datasets to make room for new data
            dset_X.resize(current_rows + num_new_rows, axis=0)
            dset_y.resize(current_rows + num_new_rows, axis=0)
            
            # Append the new data at the end
            dset_X[current_rows:] = new_x
            dset_y[current_rows:] = new_y

    def load_data(self, func_idx):
        """
        Retrieves all inputs and outputs for a specific function.
        
        Returns:
            X (numpy.ndarray): Evaluated input points
            y (numpy.ndarray): Evaluated output values
        """
        with h5py.File(self.h5_filename, 'r') as h5f:
            X = h5f[f"function_{func_idx}"]["X"][:]
            y = h5f[f"function_{func_idx}"]["y"][:]
            return X, y

if __name__ == "__main__":
    # Setup test/initialization (Change paths as needed)
    manager = BBODataManager("bbo_consolidated_data.h5")
    base_dir = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/M12/Data/M12"
    
    # Example workflow:
    # 1. Initialize file (You only need to run this once!)
    # manager.initialize_from_npy(base_dir)
    
    # 2. Simulate reading data
    # X, y = manager.load_data(func_idx=3)
    
    # 3. Simulate appending data from a solver
    # dummy_x = np.random.rand(3)
    # dummy_y = 0.5
    # manager.append_data(func_idx=3, new_x=dummy_x, new_y=dummy_y)
