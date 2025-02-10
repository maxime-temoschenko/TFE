
from preprocess import *

ds = load_xarray()

# Test the preprocessing
np_ds, _ =  preprocess_xarray_to_numpy(ds)


print('[TEST] Constant Mask (From file) ')
with h5py.File('data/mask.h5', 'r') as mask_file:
    mask = mask_file['dataset'][:]
# Test the if the mask remains constant for every kept variables and across every timesteps
var_keeps = ['RF', 'U10m', 'T2m']
for var in var_keeps:
    mask_var = xr.where(~ (ds[var].isnull()), True, False)
    expanded_mask = xr.DataArray(mask,  dims=('y', 'x'),  coords={'y': ds['y'], 'x': ds['x']}  )
    assert (mask_var == expanded_mask).all(), f"The mask is not constant across all timesteps for variable : {var}"
print('[SUCCESS] Constant Mask ')

print('[TEST] Unormalization')
# Test unnormalization -> Be aware of FP precision
norms = [None, 'minmax_01', 'minmax_11', 'quant95', 'quant99', 'robust' ]
for norm in norms:
    normed = normalize_ds(np_ds, normalization_mode=norm)
    unnormed = unnormalize_ds(normed,  normalization_mode=norm)
    assert np_ds.shape == unnormed.shape, "Shape mismatch!"
    assert np.abs(np_ds - unnormed).max() < 10**(-4), \
        f"Bad Normalization and unnormalization equivalence : {norm}"

print('[SUCCESS] Unormalization')



print('[TEST] Xarray -> Numpy Conversion ')
# Test xarray to numpy conversion
assert np_ds.shape == (ds.dims['time'] ,len(var_keeps ) +1 ,ds.dims['y'] ,ds.dims['x'])
assert np_ds[50][1][20][30] == ds['U10m'][50][20][30] # Test Random
for i, var in enumerate(var_keeps):
    assert (np_ds[: ,i] == xr.where(mask == False, 0, ds[var])  ).all(), \
        f"The conversion do not works as expected for variable : {var}"
print('[SUCCESS] Xarray -> Numpy Conversion ')

print('[TEST] Trajectory Overlap')
# Test if trajectory dataset is correct
L = 32
strided_np_ds = create_trajectory(np_ds, L)
new_dim = np_ds.shape[0 ] - L +1
for i in range(new_dim):
    for j in range(L):
        assert (strided_np_ds[i][j] == np_ds[ i +j]).all(), f"Bad Trajectory for  : { i +j}th or [{i}][{j}]"
print('[SUCESS] Trajectory Overlap')

print('[TEST] Trajectory Partition')
traj_np_ds = create_trajectory(np_ds, L, overlapping=False)
new_dim = np_ds.shape[0 ]//L
for part in range(new_dim):
    for j in range(L):
        assert (traj_np_ds[part][j] == np_ds[part * L +j]).all(), f"Bad Trajectory for  : {part * L +j}th or [{part}][{j}]"
print('[SUCCESS] Trajectory Partition')