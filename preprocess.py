import os

import xarray
import xarray as xr
import numpy as np
import h5py
import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
import fire
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import *
from torch import Size, Tensor
from tqdm import tqdm

def monthly_aggregate():
    input_file_pattern = "MARv3.14-ERA5-histo-{year}.nc"
    output_dir = "data/monthly_aggregated"
    os.makedirs(output_dir, exist_ok=True)
    for year in range(1940, 2023):
        input_file = input_file_pattern.format(year=year)
        output_file = os.path.join(output_dir, f"MARv3.14-ERA5-histo-{year}-monthly-all-vars.nc")
        try:
            ds = xr.open_dataset(input_file)
            print(f"Processing {input_file}...")
            monthly_data = xr.Dataset()
            for var in ds.data_vars:
                dims = ds[var].dims
                if "time" in dims and "y" in dims and "x" in dims:
                    if var in ["RF", "SF", "RO1", "RO2", "RO3", "RO4", "RO5", "RO6", "SMB", "ET"]:
                        monthly_data[var] = ds[var].resample(time="1M").sum()
                    else:
                        monthly_data[var] = ds[var].resample(time="1M").mean()
                elif "time" in dims:
                    if var in ["MM", "DD", "HH"]:
                        monthly_data[var] = ds[var].resample(time="1M").first()
                    else:
                        monthly_data[var] = ds[var].resample(time="1M").mean()
                elif "y" in dims and "x" in dims:
                    monthly_data[var] = ds[var]
            for coord in ds.coords:
                if coord not in monthly_data:
                    monthly_data[coord] = ds[coord]
            monthly_data.to_netcdf(output_file)
        except FileNotFoundError:
            print(f"File {input_file} not found. Skipping...")

    print("All files processed!")
# Filter files by year
def filter_by_year(files, start_year, end_year):
    filtered_files = []
    for file in files:
        year = int(file.stem.split('-')[3])  # Extract year from filename
        if start_year <= year <= end_year:
            filtered_files.append(file)
    return filtered_files


# Create new Dataset : new = old[var_keep]
def reduction_dataset(dataset, var_keep):
    list_vars = list(dataset.keys())
    vars_drop = [x for x in list_vars if x not in var_keep]
    return dataset.drop_vars(vars_drop)


# Convert inf values (unobserved values) to nan
def inf_to_nan(ds : xarray.Dataset, variables : list[str]):
    for var in variables:
        ds[var] = ds[var].where(~np.isinf(ds[var]), np.nan)
    return ds
# Conver _fillVall (unobserved values) to nan
def fillVal_to_nan(ds : xarray.Dataset, variables : list[str]):
    # Replace the fill value (e.g., 9.9692100e+36) with NaN
    for var in variables:
        fill_value = ds[var].attrs.get("_FillValue", 9.96921e36)  # Default if not explicitly set
        ds[var] = ds[var].where(ds[var] != fill_value, np.nan)  # Replace fill values with NaN
    return ds


# Returns xarray Dataset from monthly *.nc files, unobserved values are np.NaN from files between start_year and end_year
def load_xarray(
        input_folder: str = 'data/',
        start_year: int = 1940,
        end_year: int = 2022,
        var_keeps: list[str] = ['DD', 'DIST', 'HH', 'LAT', 'LON', 'MIN', 'MM', 'RF', 'U10m', 'T2m', 'YYYY'],
        process_undefined : Callable[[xarray.Dataset, list[str]], xarray.Dataset] = fillVal_to_nan
):
    Folder = Path(input_folder)
    assert Folder.exists()
    assert Folder.is_dir()
    files = list(Folder.glob("*.nc"))
    assert 1940 <= start_year <= end_year <= 2022
    filtered_files = filter_by_year(files, start_year, end_year)
    print("[LOAD XARRAY]")
    print(f"Filtered file : {filtered_files}")
    # Combine the different datasets
    print(f"Combination Dataset")
    datasets = [reduction_dataset(xr.open_dataset(file), var_keeps) for file in filtered_files]
    print(f"Preprocess Undefined regions")
    combined_ds = process_undefined(xr.concat(datasets, dim="time").sortby("time"), var_keeps)
    print("[\LOAD XARRAY]")
    return combined_ds


# largely inspired from https://github.com/schmidtjonathan/Climate2Weather/blob/main/data/pipeline.py
def normalize_ds(var_data: np.array, var : str, normalization_mode: str = None, h5file_path: str = 'data/norm_params.h5'  ):
    if h5file_path is None:
        raise ValueError("h5file_path must be specified")

    assert not (np.isinf(var_data).any()), f"{var} numpy array contains inf values !"

    if normalization_mode is None:
        var_data = var_data
        norm_dic = {}
    elif normalization_mode == 'minmax_01':  # Values [0,1]
        var_min = np.nanmin(var_data)
        var_max = np.nanmax(var_data)
        var_data = (var_data - var_min) / (var_max - var_min)
        norm_dic = {'var_min' : var_min, 'var_max' : var_max}
    elif normalization_mode == 'minmax_11':  # Values [-1,1]
        var_min = np.nanmin(var_data)
        var_max = np.nanmax(var_data)
        var_data = 2 * (var_data - var_min) / (var_max - var_min) - 1
        norm_dic = {'var_min': var_min, 'var_max': var_max}
    elif normalization_mode == 'zscore':
        mean = np.nanmean(var_data)
        std = np.nanstd(var_data)
        var_data = (var_data - mean) / std
        print(f"VAR : {var}, mean : {mean}, std : {std}")
        norm_dic = {'mean' : mean, 'std' : std}
    elif normalization_mode == 'robust':
        median = np.nanmedian(var_data)
        iqr = np.nanpercentile(var_data, 75) - np.nanpercentile(var_data, 25)
        var_data = (var_data - median) / iqr
        norm_dic = {'median' : median, 'iqr' : iqr}
    elif normalization_mode == 'quant95':
        q5 = np.nanpercentile(var_data, 5)
        q95 = np.nanpercentile(var_data, 95)
        var_data = (var_data - q5) / (q95 - q5)
        norm_dic = {'q5' : q5, 'q95' : q95}
    elif normalization_mode == 'quant99':
        q1 = np.nanpercentile(var_data, 1)
        q99 = np.nanpercentile(var_data, 99)
        var_data = (var_data - q1) / (q99 - q1)
        norm_dic = {'q1' : q1, 'q99' : q99}
    else:
        raise ValueError("Invalid normalization method")

    with h5py.File(h5file_path, 'a') as norm_file: # Organization  : /{var}/{label}
        if var not in norm_file:
            var_group = norm_file.create_group(var)
        else:
            var_group = norm_file[var]

        for label, value in norm_dic.items():
            if label in var_group:
                del var_group[label]
            var_group.create_dataset(label, data=value)
    return var_data


# largely inspired from https://github.com/schmidtjonathan/Climate2Weather/blob/main/data/pipeline.py
def unnormalize_ds(norm_data: np.array, var : str, normfile_path : str = "data/norm_params.h5", normalization_mode: str = None):
    if normfile_path is None:
        raise ValueError('Undefined normfile_path')


    with h5py.File(normfile_path, 'r') as norm_file:
        if var not in norm_file:
            raise KeyError(f"The variable hasn't been found in the file : Check normalization")
        var_group = norm_file[var]
        norm_dic = {key: var_group[key][()] for key in var_group.keys()}

    if normalization_mode is None:
        unnorm_data = norm_data
    elif normalization_mode == 'minmax_01':  # Values [0,1]
        var_min = norm_dic['var_min']
        var_max = norm_dic['var_max']
        unnorm_data = norm_data * (var_max - var_min) + var_min
    elif normalization_mode == 'minmax_11':  # Values [-1,1]
        var_min = norm_dic['var_min']
        var_max = norm_dic['var_max']
        unnorm_data = (norm_data + 1) / 2 * (var_max - var_min) + var_min
    elif normalization_mode == 'zscore':
        mean = norm_dic['mean']
        std = norm_dic['std']
        unnorm_data = norm_data * std + mean
    elif normalization_mode == 'robust':
        median = norm_dic['median']
        iqr = norm_dic['iqr']
        unnorm_data = norm_data * iqr + median
    elif normalization_mode == 'quant95':
        q5 = norm_dic['q5']
        q95 = norm_dic['q95']
        unnorm_data = norm_data * (q95 - q5) + q5
    elif normalization_mode == 'quant99':
        q1 = norm_dic['q1']
        q99 = norm_dic['q99']
        unnorm_data = norm_data * (q99 - q1) + q1
    else:
        raise ValueError("Invalid normalization method")
    return unnorm_data

def resize_array(arr: np.array, size: int , value=np.nan):
    # (*,y,x) -> (*, desired_shape, desired_shape) by cropping or padding with value
    *absorb, y_dim, x_dim = arr.shape

    if y_dim > size:
        cut_y = (y_dim - size) // 2
        arr = arr[..., cut_y: cut_y + size, :]
    if x_dim > size:
        cut_x = (x_dim - size) // 2
        arr = arr[..., :, cut_x: cut_x + size]
    *absorb, y_dim, x_dim = arr.shape

    pad_top = (size - y_dim) // 2
    pad_bottom = (size - y_dim) - pad_top
    pad_left = (size - x_dim) // 2
    pad_right = (size - x_dim) - pad_left

    pad_list = [(0, 0)] * (arr.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)]
    resized_arr = np.pad(arr, pad_list, \
                         mode="constant", \
                         constant_values=value)
    return resized_arr


def preprocess_xarray_to_numpy(dataset: xr.Dataset,
                               var_keeps: list[str] = ['RF', 'U10m', 'T2m'],
                               size= 64,
                               normalization_mode: str = None,
                               save_norm_path: str='data/norm_params.h5',
                               save_mask_path: str="data/mask.h5"):
    # INPUT :
    # dataset : xarray.Dataset with Coordinates (time, x, y) and variables with NaN for undefined regions
    # var_keeps : Variable to keep from the
    # ------
    # Returns Numpy Array -> (Time, Y, X, Channel), Channel = var_keeps
    # Mask specifies which values are undetemined, encoded as False
    data_list = []
    ds = reduction_dataset(dataset, var_keeps)
    # Mask
    mask = ~np.isnan(ds[var_keeps[0]].isel(time=0).values) # Constant mask over time
    # Date Information
    time_data = ds['time'].values
    # Process each variable
    for var in var_keeps:
        var_data = ds[var].values
        var_data = normalize_ds(var_data,var, normalization_mode,h5file_path=save_norm_path)
        var_data = np.nan_to_num(var_data, nan=0.0)
        data_list.append(var_data)
    data_np = np.stack(data_list, axis=-1)
    data_np = np.transpose(data_np, (0, 3, 1, 2))  # (T,Y,X,C) -> (T,C,Y,X)

    data_np = resize_array(data_np,size=size, value=0.)
    mask = resize_array(mask, size=size,value = False)

    if save_mask_path is not None:
        with h5py.File(save_mask_path, 'w') as hdf5_file:
            hdf5_file.create_dataset('dataset', data=mask)

    return data_np, mask, time_data

# TODO : Potentially remove
def create_trajectory(data: np.array, L: int = 32, overlapping=True):
    # Create overlapping time trajectories from (T, C, Y, X) array
    # L is the length of the trajectory
    # ------
    # overlapping :
    # True : (T, C, Y, X) -> (T-L+1, L, C, Y, X)
    # False : (T, C, Y, X) -> (T//L, L, C, Y, X)
    if overlapping == True:
        as_strided = np.lib.stride_tricks.as_strided
        resize = (data.shape[0] - L + 1, L) + data.shape[1:]
        stride = (data.strides[0],) + data.strides
        trajectory_data = as_strided(data, resize, stride)
    else:
        parts = data.shape[0] // L
        trajectory_data = np.zeros((parts, L) + data.shape[1:], dtype=data.dtype)
        for i in range(parts):
            trajectory_data[i] = data[i * L: (i + 1) * L]
    return trajectory_data

# TODO : Potentially remove
# Largely inspired from : https://github.com/francois-rozet/sda/blob/qg/experiments/kolmogorov/generate.py#L31
def generate(data: np.array):
    # Create File Dataset to prepare for the training from numpy array
    # -----------
    # INPUT :
    # data : (#Trajectory,L_Trajectory,Channels,Y,X)
    # --------
    np.random.shuffle(data)  # Shuffle along the first axis
    length = data.shape[0]
    i = int(0.8 * length)
    j = int(0.9 * length)

    splits = {
        'train': data[:i],
        'valid': data[i:j],
        'test': data[:j]
    }

    PATH = Path('.')
    for name, x in splits.items():
        PATH_DATA = PATH / 'data'
        PATH_DATA.mkdir(exist_ok=True)
        with h5py.File(PATH / f'data/{name}.h5', mode='w') as f:
            f.create_dataset('x', data=x, dtype=np.float32)
    print('[GENERATE]')
    print('Files are written')
    print('[\GENERATE]')




def main(input_folder: str = 'data/',
         start_year_train: int = 2021,
         end_year_train: int = 2021,
         year_test : int = 2022,
         var_keeps: list[str] = ['T2m', 'U10m'],
         # preprocess
         size=64,
         normalization_mode: str = 'zscore',
         save_mask_path: str = "data/mask.h5",
         ):
    # By default keeps all variables
    if var_keeps is None:
        var_keeps = ["CC", "EP", "ET", "LHF", "LWD", "RF", "RH2m", "Q2m"
                     "RO1", "RO2", "RO3", "RO4", "RO5", "RO6", "RU",
                     "SF", "SHF", "SL", "SLP", "SMB", "SN", "SP", "SQC",
                     "ST", "SWD", "SWDD", "T2m", "U10m", "U2m", "ZN"]
    print('[PREPROCESSING]')
    print(f"Folder Path: {input_folder}")
    print(f"Training : {start_year_train} to {end_year_train}")
    print(f"Test : {year_test}")
    print(f"variables to keep are : {var_keeps}")
    print(f"Last 2D of batch_size will be : {size}")
    print(f"Normalization strategy for data: {normalization_mode}")
    print(f"Mask save path: {save_mask_path}")


    # Preprocessing pipeline
    print('[TRAIN PREPROCESSING]')
    # Train
    train_ds = load_xarray(input_folder=input_folder, start_year=start_year_train, end_year=end_year_train, var_keeps=var_keeps)
    #assert (np.isinf(train_ds['T2m'].to_numpy()).any()), f"Still inf values !"
    train_np_ds, mask, train_time_date = preprocess_xarray_to_numpy(train_ds,var_keeps=var_keeps,size=size,normalization_mode=normalization_mode,save_mask_path=save_mask_path)
    print('[\TRAIN PREPROCESSING]')
    # Test
    print('[TEST PREPROCESSING')
    test_ds = load_xarray(input_folder=input_folder, start_year=year_test, end_year=year_test, var_keeps=var_keeps)
    test_np_ds, _, test_time_date  = preprocess_xarray_to_numpy(test_ds, var_keeps=var_keeps, size=size,
                                                normalization_mode=normalization_mode,save_norm_path='data/test_norm_params.h5', save_mask_path=save_mask_path )
    print('[\TEST PREPROCESSING]')


    # Save the files and make it compatible with Trajectory Dataset
    dataset = {'train': {'data': train_np_ds, 'date' : train_time_date},
               'test': {'data' : test_np_ds, 'date' : test_time_date}}
    print('[GENERATE]')
    print(type(train_time_date[0]))
    print(train_time_date[0])
    PATH = Path('.')
    for name, x in dataset.items():
        PATH_DATA = PATH / 'data'
        PATH_DATA.mkdir(exist_ok=True)
        with h5py.File(PATH / f'data/{name}.h5', mode='w') as f:
            f.create_dataset('data', data=x['data'], dtype=np.float32)
            f.create_dataset('date', data=x['date'].astype('S'), dtype=h5py.special_dtype(vlen=str))
        print(f"Write to {PATH / f'data/{name}.h5'}")
    print('[\GENERATE]')


    print('[\PREPROCESSING]')

if __name__ == '__main__':
    fire.Fire(main)
