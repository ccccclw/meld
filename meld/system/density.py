from meld.system import scalers
from openmm import unit as u  # type: ignore
import numpy as np  # type: ignore
import copy 
import scipy.ndimage # type: ignore
from typing import Union # type: ignore

class DensityManager:
    def __init__(self):
        self.densities = []

    def add_density(self, filename: Union[str, list], blur_scaler: scalers.BlurScaler, threshold=None, scale_factor=None):
        try:
            import mrcfile  # type: ignore
        except ImportError:
            print("***")
            print("The mrcfile package must be installed to use density maps.")
            print("***")
            raise
        if scale_factor is None:
            scale_factor = [0.3] * blur_scaler._num_replicas
        elif type(scale_factor) in [float, int]:
            scale_factor = [scale_factor] * blur_scaler._num_replicas
        else:
            scale_factor = scale_factor

        if threshold is None:
            threshold = [0] * blur_scaler._num_replicas
        elif type(threshold) in [float, int]:
            threshold = [threshold] * blur_scaler._num_replicas
        else:
            threshold = threshold
        
        density_data = []
        origin = []
        voxel_size = []
        if type(filename) is list:
            for density_file in filename:
                density_data.append(mrcfile.open(density_file).data)
                origin.append(list(mrcfile.open(density_file).header["origin"].item()) * u.angstrom)
                voxel_size.append(list(mrcfile.open(density_file).voxel_size.item()) * u.angstrom)
        else:
            density_data = [mrcfile.open(filename).data]
            origin = [list(mrcfile.open(filename).header["origin"].item())] * blur_scaler._num_replicas * u.angstrom
            voxel_size = [list(mrcfile.open(filename).voxel_size.item())] * blur_scaler._num_replicas * u.angstrom
        
        density = DensityMap(density_data, origin, voxel_size, blur_scaler, scale_factor, threshold)
        self.densities.append(density)
        return density 


class DensityMap:
    def __init__(
        self, 
        density_data, 
        origin, 
        voxel_size, 
        blur_scaler,
        scale_factor,
        threshold
    ):  
        self.scale_factor = scale_factor
        self.threshold = threshold
        self.nx = [density.shape[2] for density in density_data] if len(density_data) > 1 else [density_data[0].shape[2]] * blur_scaler._num_replicas
        self.ny = [density.shape[1] for density in density_data] if len(density_data) > 1 else [density_data[0].shape[1]] * blur_scaler._num_replicas
        self.nz = [density.shape[0] for density in density_data] if len(density_data) > 1 else [density_data[0].shape[0]] * blur_scaler._num_replicas
        self.blur_scaler = blur_scaler

        if len(density_data) > 1:
            self.nx = [density.shape[2] for density in density_data]
            self.ny = [density.shape[1] for density in density_data]
            self.nz = [density.shape[0] for density in density_data]
            self.density_data = [self.map_potential(density_data[index], threshold[index], scale_factor[index]) for index in range(len(density_data))]
        else:
            self.nx = [density_data[0].shape[2]] * blur_scaler._num_replicas
            self.ny = [density_data[0].shape[1]] * blur_scaler._num_replicas
            self.nz = [density_data[0].shape[0]] * blur_scaler._num_replicas
            density_data_cp = copy.deepcopy(density_data[0])
            if blur_scaler._scaler_key_ == "constant_blur":
                tmp_pot = scipy.ndimage.gaussian_filter(density_data_cp,blur_scaler.blur)
                tmp_pot = np.matrix.flatten(self.map_potential(tmp_pot,threshold[0],scale_factor[0]))
                self.density_data = [tmp_pot.astype(np.float64)] * blur_scaler._num_replicas
            elif blur_scaler._scaler_key_ == "linear_blur":
                density_data = []
                for blur in np.linspace(blur_scaler._min_blur,blur_scaler._max_blur,blur_scaler._num_replicas):
                    tmp_pot = scipy.ndimage.gaussian_filter(density_data_cp,blur)
                    tmp_pot = np.matrix.flatten(self.map_potential(tmp_pot,threshold[0],scale_factor[0]))
                    density_data.append(tmp_pot.astype(np.float64))
                self.density_data = density_data
    
        self.origin = np.array(origin.value_in_unit(u.nanometer))
        self.voxel_size = np.array(voxel_size.value_in_unit(u.nanometer))
    
    def map_potential(self, map, threshold, scale_factor):
        map_cp = copy.deepcopy(map)
        map = scale_factor * ((map - threshold) / (map.max() - threshold))
        map_where = np.where(map <= 0)
        map_cp = scale_factor * (1 - (map_cp - threshold) / (map_cp.max() - threshold))
        map_cp[map_where[0], map_where[1], map_where[2]] = scale_factor
        return map_cp

