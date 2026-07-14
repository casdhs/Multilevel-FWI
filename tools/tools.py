import torch
import torch.nn as nn
import deepwave
import numpy as np
import scipy.ndimage
from typing import Tuple
from .utils import *
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from .regularization import*
import os

import psutil

def get_cpu_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2   

def get_gpu_memory():
    if torch.cuda.is_available():
        
        return torch.cuda.memory_allocated() / 1024 ** 2
    else:
        return 0

## the forward modeling function
class Physics_deepwave(nn.Module):
    def __init__(self, dh, dt, src,
                 src_loc, rec_loc, accuracy,pml_width):
        super(Physics_deepwave, self).__init__()
        self.dh = dh
        self.dt = dt
        self.src = src
        self.src_loc = src_loc
        self.rec_loc = rec_loc
        self.accuracy = accuracy
        self.pml_width = pml_width

    
    def forward(self, vp):
        
        out = deepwave.scalar(vp, self.dh, self.dt,
                      source_amplitudes=self.src,
                      source_locations=self.src_loc,
                      receiver_locations=self.rec_loc,
                      accuracy = self.accuracy,
                      pml_width = self.pml_width
                      )
        taux = out[-1]
        return taux.permute(0, 2, 1).unsqueeze(0)


def setup_acquisition_geometry_comstom(
    inpa: dict,
    model_shape: Tuple[int, int],
    N_SHOTS: int,
    N_SOURCE_PER_SHOT: int,
    DEVICE: torch.device,
    dh: float = None,
    src_margin: int = 13,
    src_depth: int = 2,
    rec_margin: int = 13,
    rec_depth: int = 17
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Configures source and receiver geometry for seismic acquisition. Only for comtom case
    
    Args:
        inpa: Dictionary containing modeling parameters with keys:
              - dh: Grid spacing in meters
              - dt: Time sampling interval
        model_shape: Tuple of (nz, nx) model dimensions in grid points
        N_SHOTS: Total number of shot gathers
        N_SOURCE_PER_SHOT: Sources per shot gather  
        DEVICE: Target device for tensors (cpu/cuda)
        dh: Optional override for grid spacing
        src_margin: Source margin from model edges (grid points)
        src_depth: Source depth below surface (grid points)
        rec_margin: Receiver margin from model edges (grid points)
        rec_depth: Receiver depth below surface (grid points)
        
    Returns:
        src_loc: Source locations tensor [N_SHOTS, N_SOURCE_PER_SHOT, 2] 
                 (last dim: [x,z] grid indices)
        rec_loc: Receiver locations tensor [N_SHOTS, N_RECEIVERS, 2]
    
    Geometry Design:
        - Sources: Distributed linearly across model width with margin
        - Receivers: Surface array with margin on each side
        - Coordinates: [x, z] with z positive downward
        - Units: Grid indices (converted from physical units)
    """
    dh = dh or inpa['dh']  # Use provided dh or fallback to inpa
    
    # 1. Calculate model dimensions in physical units
    offsetx = dh * model_shape[1]  # Total model width (m)
    depth = dh * model_shape[0]    # Total model depth (m)
    
    print(f"Model dimensions: {offsetx:.1f}m (width) x {depth:.1f}m (depth)")

    # 2. Initialize receiver geometry (surface array)
    rec_start = rec_margin * dh
    rec_end = offsetx - rec_margin * dh
    surface_loc_x = np.arange(
        start=rec_start,
        stop=rec_end,
        step=dh,
        dtype=np.float32
    )
    n_surface_rec = len(surface_loc_x)
    surface_loc_z = rec_depth * dh * np.ones(n_surface_rec, np.float32)
    rec_loc_temp = np.vstack((surface_loc_x, surface_loc_z)).T
    
    # 3. Initialize source geometry (shallow depth line)
    src_start = src_margin * dh
    src_end = offsetx - src_margin * dh
    src_loc_temp = np.vstack((
        np.linspace(
            start=src_start,
            stop=src_end,
            num=N_SHOTS,
            dtype=np.float32
        ),
        src_depth * dh * np.ones(N_SHOTS, np.float32)
    )).T
    
    # Adjust source depth (optional)
    src_loc_temp[:, 1] -= 2 * dh  # Move sources slightly upward

    # 4. Convert to grid indices and format tensors
    # Sources [N_SHOTS, N_SOURCE_PER_SHOT, 2]
    src_loc = torch.zeros(N_SHOTS, N_SOURCE_PER_SHOT, 2,
                         dtype=torch.int, device=DEVICE)
    src_loc[:, 0, :] = torch.from_numpy(np.flip(src_loc_temp) // dh)
    src_loc[:, :, 0] = src_depth  # Set x-indices (example configuration)
    
    # Receivers [N_SHOTS, N_RECEIVERS, 2]
    rec_loc = torch.zeros(N_SHOTS, n_surface_rec, 2,
                         dtype=torch.long, device=DEVICE)
    rec_loc[:, :, :] = torch.from_numpy(np.flip(rec_loc_temp) / dh)
    
    print(f"Acquisition geometry configured:")
    print(f"- Sources: {N_SHOTS}x{N_SOURCE_PER_SHOT} at ~{src_depth*dh}m depth")
    print(f"- Receivers: {n_surface_rec} channels at ~{rec_depth*dh}m depth")
    print(f"- Source locations shape: {tuple(src_loc.shape)}")
    print(f"- Receiver locations shape: {tuple(rec_loc.shape)}")
    
    return src_loc, rec_loc

def generate_source_wavelets(
    F_PEAK: float,
    NT: int,
    DT: float,
    N_SHOTS: int,
    N_SOURCE_PER_SHOT: int,
    DEVICE: torch.device,
    test_low_fre: str,
    cutoff_fre: float,
    corners: float
) -> torch.Tensor:
    """
    Generate and process source wavelets for seismic simulation.
    
    Args:
        F_PEAK: Peak frequency for Ricker wavelet (Hz)
        NT: Number of time samples
        DT: Time sampling interval (s)
        N_SHOTS: Number of shot gathers
        N_SOURCE_PER_SHOT: Sources per shot gather  
        DEVICE: Target device for tensors (cpu/cuda)
        test_low_fre: 'yes' to apply highpass filter, 'no' otherwise
        cutoff_fre: Filter cutoff frequency when test_low_fre='yes' (Hz)
        corners: Filter corners/slope when filtering
        
    Returns:
        src: Source wavelet tensor of shape [N_SHOTS, N_SOURCE_PER_SHOT, NT]
        
    The function:
    1. Generates Ricker wavelets with specified peak frequency
    2. Replicates for all shots and sources
    3. Optionally applies highpass filter
    4. Ensures proper tensor dtype and device placement
    """
    
    # Generate base Ricker wavelet
    src = deepwave.wavelets.ricker(
        F_PEAK,
        NT,
        DT,
        1.5/F_PEAK  # Standard 1.5 cycle delay
    )
    
    # Replicate for all shots and sources
    src = src.repeat(N_SHOTS, N_SOURCE_PER_SHOT, 1).to(DEVICE)
    
    # Apply optional highpass filtering
    if test_low_fre == 'yes':
        src = seismic_filter(
            data=src.cpu().numpy(),  # Filter operates on numpy arrays
            filter_type='highpass',
            freqmin=cutoff_fre,
            freqmax=None,  # No upper cutoff
            df=1/DT,       # Frequency step
            corners=corners
        )
        src = torch.tensor(src, dtype=torch.float32, device=DEVICE)
    else:
        src = src.to(torch.float32)  # Ensure consistent dtype
    
    # Validate output
    print(f"Generated {N_SHOTS}x{N_SOURCE_PER_SHOT} source wavelets:")
    print(f"- Peak frequency: {F_PEAK}Hz")
    print(f"- Duration: {NT*DT:.3f}s ({NT} samples)")
    if test_low_fre == 'yes':
        print(f"- Applied {corners}-corner highpass at {cutoff_fre}Hz")
    
    return src

def run_forward_simulation(
    Physics: callable,
    vp_true: torch.Tensor,
    inpa: dict,
    src: torch.Tensor,
    src_loc: torch.Tensor,
    rec_loc: torch.Tensor,
    test_low_noise: str,
    noise_level: float,
    device: torch.device = None
) -> torch.Tensor:
    """
    Run forward seismic simulation and apply specified data modifications.
    
    Args:
        Physics: Wave propagation physics class (e.g., Physics_deepwave)
        vp_true: True velocity model [nz, nx]
        inpa: Simulation parameters dictionary with:
              - dh: Grid spacing (m)
              - dt: Time step (s)
        src: Source wavelets [N_SHOTS, N_SOURCES, NT]
        src_loc: Source locations [N_SHOTS, N_SOURCES, 2]
        rec_loc: Receiver locations [N_SHOTS, N_RECEIVERS, 2]
        test_low_noise: 'yes' to add noise, 'no' otherwise
        noise_level: Standard deviation of Gaussian noise
        test_outers: 'yes' to zero out intervals, 'no' otherwise  
        interval: Number of consecutive samples to zero out
        device: Target device for computations
        
    Returns:
        d_obs: Synthetic seismic data [N_SHOTS, N_RECEIVERS, NT]
    
    Processing steps:
        1. Initialize physics solver with model parameters
        2. Run forward simulation
        3. Optionally add Gaussian noise
        4. Optionally zero out time intervals
    """
    
        
    accuracy = inpa['accuracy']
    pml_width = inpa['pml_width']
    
    
    # 1. Initialize physics solver
    physics = Physics(
        inpa['dh'],
        inpa['dt'],
        src=src.to(device),
        src_loc=src_loc.to(device),
        rec_loc=rec_loc.to(device),
        accuracy = accuracy,
    	pml_width = pml_width
    )
    
    # 2. Run forward simulation
    d_obs = physics(vp_true.squeeze()).squeeze()
    
    # 3. Add Gaussian noise if specified
    if test_low_noise.lower() == 'yes':
        d_obs = AddAWGN(d_obs, noise_level)
        #d_obs = add_gaussian_noise(d_obs, noise_level, device)
        print(f"Added Gaussian noise (snr={noise_level})")
        
    # Validate output dimensions
    print(f"Generated synthetic data shape: {tuple(d_obs.shape)} "
          f"(shots×receivers×time)")
    
    return d_obs






