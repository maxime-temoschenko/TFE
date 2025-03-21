import h5py
from preprocess import *
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from typing import *
from datetime import datetime
from siren_pytorch import SirenNet

from score import ScoreUNet, VPSDE

ACTIVATIONS = {
    'ReLU': torch.nn.ReLU,
    'ELU': torch.nn.ELU,
    'GELU': torch.nn.GELU,
    'SELU': torch.nn.SELU,
    'SiLU': torch.nn.SiLU,
}


# Adapted from https://github.com/francois-rozet/sda/blob/qg/sda/utils.py#L58
class SequenceDataset(Dataset):
    def __init__(self,
                 file: Path,
                 window : int = None,
                 flatten: bool = False,
                 slicer_data : slice = slice(None)):
        super().__init__()
        with h5py.File(file, mode='r') as f:
            self.data = f['data'][slicer_data]
            self.date = f['date'][slicer_data]
        date = [datetime.fromisoformat(d.decode()[:-6]) for d in self.date]
        self.frac_hour = [d.hour / 23 for d in date]
        self.frac_day_of_year = [d.timetuple().tm_yday / 365 for d in date]
        self.window = window
        self.flatten = flatten
        # Spatial Encoding
        _, _, self.y, self.x = self.data.shape
        grid = torch.stack(
            torch.meshgrid(
                2*torch.pi*torch.arange(self.y)/self.y,
                2*torch.pi*torch.arange(self.x)/self.x,
                indexing='ij'
            )
        )
        self.spatial_encoding = torch.cat((grid.cos(), grid.sin()), dim=0)
    def __len__(self) -> int:
        return len(self.data) - self.window + 1
    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        data_x = torch.from_numpy(self.data)
        day_, hour_ = self.frac_day_of_year[i], self.frac_hour[i]
        i = torch.randint(0, len(self.data) - self.window + 1, size=()).item()
        traj_x = torch.narrow(data_x, dim=0, start=i, length=self.window)
        context = self.spatial_encoding
        if self.flatten:
            return traj_x.flatten(0, 1), { 'context' : context, 'frac_day_of_year' : day_, 'frac_hour_of_day' : hour_}

        return traj_x, { 'context' : context, 'frac_day_of_year' : day_[i], 'frac_hour_of_day' : hour_[i]}

class BatchDataset(Dataset):
    def __init__(self, file: Path, data_keyword = 'samples'):
        super().__init__()
        with h5py.File(file, mode='r') as f:
            self.data = f[data_keyword][:]  # shape: (N, channels, y, x)
        _, self.channels, self.y, self.x = self.data.shape
        grid_y = 2 * torch.pi * torch.arange(self.y, dtype=torch.float32) / self.y
        grid_x = 2 * torch.pi * torch.arange(self.x, dtype=torch.float32) / self.x
        grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing='ij'))
        self.spatial_encoding = torch.cat((grid.cos(), grid.sin()), dim=0)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, i):
        sample = torch.from_numpy(self.data[i])
        return sample, {'context': self.spatial_encoding}

def date_encoding(day : int, hour : int):
    day_emb = (np.sin(2*np.pi*day/365), np.cos(2*np.pi*day/365))
    hour_emb = (np.sin(2*np.pi*hour/365), np.cos(2*np.pi*hour/365))
    return day_emb+hour_emb


class DateEmbedding(nn.Module):
    def __init__(self, x: int = 64, y: int = 64, hidden_dim=256, num_layers=6, output_channels : int = 1, w0_initial=30, **kwargs):
        super().__init__()
        self.x = x
        self.y = y
        self.output_channels = output_channels
        self.nn = SirenNet(dim_in=4, dim_hidden=hidden_dim, num_layers=num_layers, dim_out=x * y * self.output_channels,
                            w0_initial=30)

    def forward(self, frac_day, frac_hour):
        day_sin, day_cos = torch.sin(2*torch.pi*frac_day), torch.cos(2*torch.pi*frac_day)
        hour_sin, hour_cos = torch.sin(2*torch.pi*frac_hour), torch.cos(2*torch.pi*frac_hour)
        input_nn = torch.stack((day_sin, day_cos, hour_sin, hour_cos), dim=1).float()
        output = self.nn(input_nn)
        output = output.view(-1, self.output_channels, self.y, self.x)
        # TODO : make it properly
        if self.x != 64 or self.y != 64:
            output = resize_tensor(output, 64,  value=0.0)
        return output 

class RawDateEmbedding(nn.Module):
    def __init__(self, x: int = 64, y: int = 64, hidden_dim=256, num_layers=5, w0_initial=30, **kwargs):
        super().__init__()
        self.x = x
        self.y = y
        self.nn = SirenNet(dim_in=2, dim_hidden=hidden_dim, num_layers=num_layers, dim_out=x * y,
                           final_activation=nn.Sigmoid(), w0_initial=30)

    def forward(self, frac_day, frac_hour):
        input_nn = torch.stack((frac_day, frac_hour), dim=1).float()
        output = self.nn(input_nn)
        return output.view(-1, self.y, self.x)
def clip_outliers(tensor, n_std=3):
    mean = np.nanmean(tensor)
    std = np.nanstd(tensor)
    return torch.clamp(tensor, min=mean-n_std*std, max=mean+n_std*std)

def plot_sample(batch, info, mask, samples, step=4, unnormalize=True, path_unnorm='data/norm_params.h5', 
                lat_lon_path='data/lat_lon.h5', show_borders=True):
    B, Z, Y, X = batch.shape
    redim_batch = batch.view(B, info['window'], info['channels'], Y, X).permute(0, 2, 1, 3, 4) 
    if unnormalize == True:
        for i in range(info['channels']):
            redim_batch[:,i, ...] = unnormalize_ds(redim_batch[:,i, ...], info['var_index'][i], 
                                                 normfile_path=path_unnorm, normalization_mode='zscore')
    
    traj = torch.where(mask.unsqueeze(0).unsqueeze(0).bool(), redim_batch, 
                      torch.tensor(float('nan'), dtype=redim_batch.dtype))
    data = traj
    s, variables, timesteps, y_dim, x_dim = data.shape
    selected_timesteps = range(0, timesteps, step)
    
    use_cartopy = show_borders and lat_lon_path is not None
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            with h5py.File(lat_lon_path, 'r') as f:
                lat_grid = f['LAT_INTERP'][:]
                lon_grid = f['LON_INTERP'][:]
                
            print(f"Loaded lat/lon grid with shape: {lat_grid.shape}")
            
        except Exception as e:
            print(f"Failed to load lat/lon data: {e}")
            use_cartopy = False

    fig, axes = plt.subplots(nrows=info['channels'] * samples, ncols=len(selected_timesteps), 
                            figsize=(15, 5 * samples), 
                            subplot_kw={'projection': ccrs.PlateCarree()} if use_cartopy else None)
    
    if len(selected_timesteps) == 1:
        axes = np.atleast_2d(axes).T
    
    cbar_axes = []
    
    for s_idx in range(samples):
        for var in range(variables):
            mean, std = np.nanmean(data[s_idx, var, :]), np.nanstd(data[s_idx, var, :])
            print(f"{info['var_index'][var]}  Mean : {mean}, Var: {std}")
            
            for i, t in enumerate(selected_timesteps):
                ax = axes[info['channels'] * s_idx + var, i]
                data[s_idx, var, t] = clip_outliers(data[s_idx, var, t])
                img_data = data[s_idx, var, t].numpy()

                img_data = np.ma.masked_invalid(img_data)
                
                cmap = plt.get_cmap('viridis' if var == 0 else 'plasma')
                cmap.set_bad(color='black')
                
                vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
                
                if use_cartopy:
                    img = ax.pcolormesh(lon_grid, lat_grid, np.flipud(img_data), 
                                       cmap=cmap, vmin=vmin, vmax=vmax, 
                                       transform=ccrs.PlateCarree())
                    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax.coastlines(linewidth=0.5)
                    ax.set_extent([lon_grid.min(), lon_grid.max(), 
                                  lat_grid.min(), lat_grid.max()], 
                                 crs=ccrs.PlateCarree())
                    ax.gridlines(draw_labels=False, alpha=0.0)
                else:
                    img = ax.imshow(img_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                if t == 0:
                    list_var = ["Temperature[°C]", "Wind Speed[M/S]"]
                    ax.set_ylabel(f'{list_var[var]}')
                if var == 0:
                    ax.set_title(f"{t} Hours")
                
                cbar_axes.append(fig.colorbar(img, ax=ax))

    for s_idx in range(samples):
        fig.text(
            -0.05, 1 - (2 * s_idx + 1) / (2 * samples),
            f"Sample {s_idx}",
            va='center', ha='center', rotation=90, fontsize=12, fontweight='bold'
        )
        
    fig.suptitle("Visualization of samples", fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig
def constructEmbedding(date_embedding, dic):
    date_embedding.eval()
    date_output = date_embedding(dic['frac_day_of_year'], dic['frac_hour_of_day'])
    return torch.cat((dic['context'], date_output), dim=1)

def constructNaiveContext(dic, size=64):
    device = dic['context'].device
    day_frac = dic['frac_day_of_year']
    hour_frac = dic['frac_hour_of_day']
    if not isinstance(day_frac, torch.Tensor):
        day_frac = torch.tensor(day_frac, device=device)
    if not isinstance(hour_frac, torch.Tensor):
        hour_frac = torch.tensor(hour_frac, device=device)
    day_sin = torch.sin(2 * torch.pi * day_frac).view(-1, 1, 1)
    day_cos = torch.cos(2 * torch.pi * day_frac).view(-1, 1, 1)
    hour_sin = torch.sin(2 * torch.pi * hour_frac).view(-1, 1, 1)
    hour_cos = torch.cos(2 * torch.pi * hour_frac).view(-1, 1, 1)
    day_sin = day_sin.expand(-1, size, size)
    day_cos = day_cos.expand(-1, size, size)
    hour_sin = hour_sin.expand(-1, size, size)
    hour_cos = hour_cos.expand(-1, size, size)
    temporal_context = torch.stack([day_sin, day_cos, hour_sin, hour_cos], dim=1).float()
    print("temporal : ", temporal_context.shape)
    context = torch.cat((dic['context'], temporal_context), dim=1).float()
    print(context.shape)
    return context

def load_setup(CONFIG, path_data : str, checkpoint_path: str, device, date_embedding = False, date_embedding_args={},multi_gpu = True):
    setup = {}
    PATH_DATA = Path(path_data)
    with h5py.File(PATH_DATA / "mask.h5", "r") as f:
        setup['mask'] = torch.tensor(f["dataset"][:], dtype=torch.float32, device=device).unsqueeze(0)
        setup['mask_cpu'] = setup['mask'].detach().clone().cpu()
    #setup['trainset'] = SequenceDataset(PATH_DATA / "train.h5", window=CONFIG['window'], flatten=True)
    setup['validset'] = SequenceDataset(PATH_DATA / "test.h5", window=CONFIG['window'], flatten=True)
    setup['validloader'] = DataLoader(setup['validset'], batch_size=CONFIG['batch_size'], shuffle=True, num_workers=1, persistent_workers=True)
    score_unet = ScoreUNet(**CONFIG)
    score_unet.eval()
    vpsde  = VPSDE(score_unet, shape=(CONFIG['channels'], CONFIG['y'], CONFIG['x']), eta=CONFIG['eta']).to(device)
    vpsde.eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if multi_gpu == True:
        new_state_dict = {key.replace("module.", ""): value for key, value in checkpoint['model_state_dict'].items()}
    else:
        new_state_dict = checkpoint['model_state_dict']
    vpsde.load_state_dict(new_state_dict)
    if date_embedding == True:
        date_embedding = DateEmbedding(**date_embedding_args)
        date_embedding.eval()
        if multi_gpu == True :
            new_state_dict = {key.replace("module.", ""): value for key, value in checkpoint['date_embedding_state_dict'].items()}
        else:
            new_state_dict = checkpoint['date_embedding_state_dict']
        date_embedding.load_state_dict(new_state_dict)
        setup['date_embedding'] = date_embedding
        print(f'Loaded Date Embedding')
    setup['vpsde'] = vpsde
    print(f"Model restored from {checkpoint_path}, trained until epoch {checkpoint['epoch']}")

    return setup
