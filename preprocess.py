import os
import json
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
def normalize_ds(var_data: np.array, var : str, normalization_mode: str = None  ):
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

    return var_data, norm_dic


# largely inspired from https://github.com/schmidtjonathan/Climate2Weather/blob/main/data/pipeline.py
def unnormalize_ds(norm_data: np.array, var : str, normfile_path : str = "data/norm_params.h5", normalization_mode: str = None):
    norm_params = load_norm_params(normfile_path)
    if var not in norm_params:
        raise ValueError(f"{var} doesn't exist in the normalization of {normfile_path}")
    norm_dic = norm_params[var]

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

# TODO : Combine resize array and tensor
def resize_array(arr: np.array, size: int, value=np.nan):
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

def resize_tensor(tensor: torch.Tensor, size: int, value=0.0):
    device = tensor.device
    *batch_dims, y_dim, x_dim = tensor.shape
    if y_dim > size:
        cut_y = (y_dim - size) // 2
        tensor = tensor[..., cut_y:cut_y + size, :]
    if x_dim > size:
        cut_x = (x_dim - size) // 2
        tensor = tensor[..., :, cut_x:cut_x + size]
    *batch_dims, y_dim, x_dim = tensor.shape
    pad_top = (size - y_dim) // 2
    pad_bottom = (size - y_dim) - pad_top
    pad_left = (size - x_dim) // 2
    pad_right = (size - x_dim) - pad_left
    resized_tensor = F.pad(
        tensor, 
        [pad_left, pad_right, pad_top, pad_bottom],
        mode='constant', 
        value=value
    )
    return resized_tensor

def preprocess_xarray_to_numpy(dataset: xr.Dataset,
                               var_keeps: list[str] = ['RF', 'U10m', 'T2m'],
                               size= 64,
                               normalization_mode: str = None):
    # INPUT :
    # dataset : xarray.Dataset with Coordinates (time, x, y) and variables with NaN for undefined regions
    # var_keeps : Variable to keep from the dataset
    # size : Redimension dataset : (*,Y,X) -> (*,size, size)
    # normalization_mode : choose how to normalize the data
    # ------
    # Returns dictionnary
    #
    norm_params = {}
    data_list = []
    ds = reduction_dataset(dataset, var_keeps)
    mask = ~np.isnan(ds[var_keeps[0]].isel(time=0).values) # Constant mask over time
    time_data = ds['time'].values
    for var in var_keeps:
        var_data = ds[var].values
        var_data, norm_params[var] = normalize_ds(var_data,var, normalization_mode)
        var_data = np.nan_to_num(var_data, nan=0.0)
        data_list.append(var_data)
    data_np = np.stack(data_list, axis=-1)
    data_np = np.transpose(data_np, (0, 3, 1, 2))  # (T,Y,X,C) -> (T,C,Y,X)

    data_np = resize_array(data_np,size=size, value=0.)
    mask = resize_array(mask, size=size,value = False)
    return {'data' : data_np, 'mask' : mask, 'time' : time_data, 'norm_params': norm_params}

def save_dataset_info(output_folder, args):
    with open(Path(output_folder) / "dataset_info.txt", "w") as f:
        json.dump(args, f, indent=4)
def load_norm_params(h5_path: str):
    norm_params = {}
    with h5py.File(h5_path, 'r') as f:
        for var in f['norm_params'].keys():
            norm_params[var] = {key: f[f'norm_params/{var}/{key}'][()] for key in f[f'norm_params/{var}'].keys()}
    return norm_params
def main(input_folder: str = 'data/',
         output_folder : str = 'data/processed',
         start_year_train: int = 2021,
         end_year_train: int = 2021,
         start_year_test : int = 2022,
         end_year_test : int = 2022,
         var_keeps: list[str] = ['T2m', 'U10m'],
         # preprocess
         size=64,
         normalization_mode: str = 'zscore',
         ):
    info = locals()
    PATH = Path(output_folder)
    PATH.mkdir(exist_ok=True)

    save_dataset_info(output_folder, info)

    # By default keeps all variables
    if var_keeps is None:
        var_keeps = ["CC", "EP", "ET", "LHF", "LWD", "RF", "RH2m", "Q2m"
                     "RO1", "RO2", "RO3", "RO4", "RO5", "RO6", "RU",
                     "SF", "SHF", "SL", "SLP", "SMB", "SN", "SP", "SQC",
                     "ST", "SWD", "SWDD", "T2m", "U10m", "U2m", "ZN"]
    print('[PREPROCESSING]')
    print(f"Folder Path: {input_folder}")
    print(f"Training : {start_year_train} to {end_year_train}")
    print(f"Test : {start_year_test} to {end_year_test}")
    print(f"variables to keep are : {var_keeps}")
    print(f"Last 2D of batch_size will be : {size}")
    print(f"Normalization strategy for data: {normalization_mode}")

    # Preprocessing pipeline
    print('[TRAIN PREPROCESSING]')
    # Train
    train_ds = load_xarray(input_folder=input_folder, start_year=start_year_train, end_year=end_year_train, var_keeps=var_keeps)
    train_data_dict = preprocess_xarray_to_numpy(train_ds,var_keeps=var_keeps,size=size,normalization_mode=normalization_mode)
    print('[\TRAIN PREPROCESSING]')
    # Test
    print('[TEST PREPROCESSING')
    test_ds = load_xarray(input_folder=input_folder, start_year=start_year_test, end_year=end_year_test, var_keeps=var_keeps)
    test_data_dict = preprocess_xarray_to_numpy(test_ds, var_keeps=var_keeps, size=size,normalization_mode=normalization_mode)
    print('[\TEST PREPROCESSING]')


    # Save the files
    dataset = {'train': train_data_dict,
               'test': test_data_dict}
    print('[GENERATE]')

    for name, data_dict in dataset.items():
        with h5py.File(PATH / f'{name}.h5', mode='w') as f:
            f.create_dataset('data', data=data_dict['data'], dtype=np.float32)
            f.create_dataset('date', data=data_dict['time'].astype('S'), dtype=h5py.special_dtype(vlen=str))
            norm_grp = f.create_group('norm_params')
            for var, norm_params in data_dict['norm_params'].items():
                var_grp = norm_grp.create_group(var)
                for key, value in norm_params.items():
                    var_grp.create_dataset(key, data=value)
        print(f"Write to {PATH / f'{name}.h5'}")
    with h5py.File(PATH / "mask.h5", mode='w') as f:
        f.create_dataset('dataset', data=train_data_dict['mask'])
    print(f"Write to {PATH / 'mask.h5'}")
    print('[\GENERATE]')


    print('[\PREPROCESSING]')

if __name__ == '__main__':
    fire.Fire(main)
