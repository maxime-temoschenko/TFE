import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
from torch.nn.functional import mse_loss, l1_loss
from pysteps.utils.spectral import rapsd
from typing import *

def calculate_rmse(prediction, target, mask=None, num_variables=2):
    """
    Calculate Root Mean Square Error between prediction and target.
    
    Args:
        prediction: Tensor of shape [batch, window*channels, height, width] or [window*channels, height, width]
        target: Tensor with the same shape as prediction
        mask: Optional mask for valid data points   
    """
    if mask is not None:
        prediction = prediction * mask
        target = target * mask
        
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        
    batch_size, channels, height, width = prediction.shape
    
    window = channels // num_variables
    
    # (Batch, window*num_variables, height, width) -> (Batch, window, num_variables, height, width)
    prediction_reshaped = prediction.reshape(batch_size, window, num_variables, height, width)
    target_reshaped = target.reshape(batch_size, window, num_variables, height, width)
    
    # (Batch, num_variables, window, height, width)
    prediction_reshaped = prediction_reshaped.permute(0, 2, 1, 3, 4)
    target_reshaped = target_reshaped.permute(0, 2, 1, 3, 4)
    
    
    # Overall RMSE per variable (across all timesteps)
    overall_rmse_per_var = torch.zeros(num_variables)
    # Ensemble RMSE 
    prediction_mean  = prediction_reshaped.mean(dim=0) # [num_variables, window, height, width]
    ensemble_rmse_per_var = torch.zeros(num_variables)
    for v in range(num_variables):
        if mask is not None:
            valid_points = mask.sum() * batch_size * window
            if valid_points > 0:
                var_error = ((prediction_reshaped[:, v] - target_reshaped[:, v]) ** 2).sum() / valid_points
                overall_rmse_per_var[v] = torch.sqrt(var_error)
                ensemble_rmse_per_var[v] = torch.sqrt(((prediction_mean[v] - target_reshaped[0, v]) ** 2).sum() / (mask.sum() * window))
        else:
            var_error = mse_loss(prediction_reshaped[:, v].reshape(-1, height, width), 
                              target_reshaped[:, v].reshape(-1, height, width))
            overall_rmse_per_var[v] = torch.sqrt(var_error)
    
    
    return {
        'per_variable': overall_rmse_per_var,
        'ensemble_rmse': ensemble_rmse_per_var
    }


def calculate_mae(prediction, target, mask=None, num_variables=2):
    """
    Calculate Mean Absolute Error between prediction and target.
    
    Args:
        prediction: Tensor of shape [batch, window*channels, height, width] or [window*channels, height, width]
        target: Tensor with the same shape as prediction
        mask: Optional mask for valid data points
        
    Returns:
        Dictionary containing:
         - per_var_time: MAE per variable and timestep
         - overall: Overall MAE across all variables and timesteps
         - per_variable: Overall MAE for each variable
    """
    if mask is not None:
        prediction = prediction * mask
        target = target * mask
        
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        
    batch_size, channels, height, width = prediction.shape

    window = channels // num_variables
    
    # [batch, window*channels, height, width] -> [batch, window, num_variables, height, width]
    prediction_reshaped = prediction.reshape(batch_size, window, num_variables, height, width)
    target_reshaped = target.reshape(batch_size, window, num_variables, height, width)
    
    # [batch, num_variables, window, height, width]
    prediction_reshaped = prediction_reshaped.permute(0, 2, 1, 3, 4)
    target_reshaped = target_reshaped.permute(0, 2, 1, 3, 4)
    
    # MAE per variable and timestep
    mae_per_var_time = torch.zeros(num_variables, window)
    
    # Overall MAE per variable (across all timesteps)
    overall_mae_per_var = torch.zeros(num_variables)
    
    for v in range(num_variables):
        # Calculate MAE per timestep for this variable
        for t in range(window):
            if mask is not None:
                valid_points = mask.sum()
                if valid_points > 0:
                    error = (prediction_reshaped[:, v, t] - target_reshaped[:, v, t]).abs().sum() / valid_points
                    mae_per_var_time[v, t] = error
            else:
                error = l1_loss(prediction_reshaped[:, v, t], target_reshaped[:, v, t])
                mae_per_var_time[v, t] = error
                
        # Calculate overall MAE for this variable (across all timesteps)
        if mask is not None:
            valid_points = mask.sum() * batch_size * window
            if valid_points > 0:
                var_error = (prediction_reshaped[:, v] - target_reshaped[:, v]).abs().sum() / valid_points
                overall_mae_per_var[v] = var_error
        else:
            var_error = l1_loss(prediction_reshaped[:, v].reshape(-1, height, width), 
                             target_reshaped[:, v].reshape(-1, height, width))
            overall_mae_per_var[v] = var_error
    
    # Overall MAE across all variables and timesteps
    if mask is not None:
        valid_points = mask.sum() * batch_size * channels
        if valid_points > 0:
            overall_error = (prediction - target).abs().sum() / valid_points
        else:
            overall_error = torch.tensor(float('nan'))
    else:
        overall_error = l1_loss(prediction, target)
    
    return {
        'per_var_time': mae_per_var_time,
        'overall': overall_error,
        'per_variable': overall_mae_per_var
    }


def calculate_ssim(prediction, target, mask=None):
    """
    Calculate Structural Similarity Index (SSIM) between prediction and target.
    
    Args:
        prediction: Tensor of shape [batch, channels, height, width] or [channels, height, width]
        target: Tensor with the same shape as prediction
        mask: Optional mask for valid data points
        
    Returns:
        Dictionary containing:
         - per_var_time: SSIM per variable and timestep
         - per_variable: Overall SSIM for each variable
    """
    # Convert to numpy for skimage
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
        
    # Ensure we have batch dimension
    if prediction.ndim == 3:
        prediction = np.expand_dims(prediction, axis=0)
        target = np.expand_dims(target, axis=0)
        
    batch_size, channels, height, width = prediction.shape
    
    num_variables = 2  # T2m and U10m
    window = channels // num_variables
    
    # First reshape to separate the interleaved structure
    prediction_reshaped = prediction.reshape(batch_size, window, num_variables, height, width)
    target_reshaped = target.reshape(batch_size, window, num_variables, height, width)
    
    # Then transpose to get [batch, var, time, height, width]
    prediction_reshaped = np.transpose(prediction_reshaped, (0, 2, 1, 3, 4))
    target_reshaped = np.transpose(target_reshaped, (0, 2, 1, 3, 4))
    
    # SSIM per variable and timestep
    ssim_per_var_time = np.zeros((num_variables, window))
    
    # Overall SSIM per variable (across all timesteps)
    overall_ssim_per_var = np.zeros(num_variables)
    
    for v in range(num_variables):
        # Calculate SSIM per timestep for this variable
        timestep_ssim_values = []
        
        for t in range(window):
            timestep_ssim_sum = 0
            for b in range(batch_size):
                pred_slice = prediction_reshaped[b, v, t]
                target_slice = target_reshaped[b, v, t]
                
                # Normalize to [0, 1] for SSIM calculation
                pred_min, pred_max = np.nanmin(pred_slice), np.nanmax(pred_slice)
                target_min, target_max = np.nanmin(target_slice), np.nanmax(target_slice)
                
                if pred_max > pred_min and target_max > target_min:
                    pred_norm = (pred_slice - pred_min) / (pred_max - pred_min)
                    target_norm = (target_slice - target_min) / (target_max - target_min)
                    
                    if mask is not None:
                        # Create a mask array for this specific slice
                        mask_slice = mask[0] if mask.ndim == 4 else mask
                        
                        # Calculate SSIM only on masked region
                        ssim_value = ssim(pred_norm, target_norm, 
                                         data_range=1.0, 
                                         gaussian_weights=True, 
                                         sigma=1.5, 
                                         use_sample_covariance=False,
                                         mask=mask_slice > 0)
                    else:
                        ssim_value = ssim(pred_norm, target_norm, 
                                         data_range=1.0, 
                                         gaussian_weights=True, 
                                         sigma=1.5, 
                                         use_sample_covariance=False)
                    
                    timestep_ssim_sum += ssim_value
                    timestep_ssim_values.append(ssim_value)
            
            if batch_size > 0:
                ssim_per_var_time[v, t] = timestep_ssim_sum / batch_size
        
        # Calculate overall SSIM for this variable (average across all timesteps)
        if timestep_ssim_values:
            overall_ssim_per_var[v] = np.mean(timestep_ssim_values)
    
    return {
        'per_var_time': ssim_per_var_time,
        'per_variable': overall_ssim_per_var
    }


def calculate_wasserstein(prediction, target, mask=None):
    """
    Calculate Wasserstein Distance between prediction and target.
    
    Args:
        prediction: Tensor of shape [batch, channels, height, width] or [channels, height, width]
        target: Tensor with the same shape as prediction
        mask: Optional mask for valid data points
        
    Returns:
        Wasserstein distance per channel and timestep
    """
    # Convert to numpy
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Ensure we have batch dimension
    if prediction.ndim == 3:
        prediction = np.expand_dims(prediction, axis=0)
        target = np.expand_dims(target, axis=0)
        
    batch_size, channels, height, width = prediction.shape
    
    num_variables = 2  # T2m and U10m
    window = channels // num_variables
    
    # First reshape to separate the interleaved structure
    prediction_reshaped = prediction.reshape(batch_size, window, num_variables, height, width)
    target_reshaped = target.reshape(batch_size, window, num_variables, height, width)
    
    # Then transpose to get [batch, var, time, height, width]
    prediction_reshaped = np.transpose(prediction_reshaped, (0, 2, 1, 3, 4))
    target_reshaped = np.transpose(target_reshaped, (0, 2, 1, 3, 4))
    
    wasserstein_per_var_time = np.zeros((num_variables, window))
    
    for v in range(num_variables):
        for t in range(window):
            for b in range(batch_size):
                pred_slice = prediction_reshaped[b, v, t]
                target_slice = target_reshaped[b, v, t]
                
                if mask is not None:
                    # Apply mask
                    mask_slice = mask[0] if mask.ndim == 3 else mask
                    pred_values = pred_slice[mask_slice > 0]
                    target_values = target_slice[mask_slice > 0]
                else:
                    pred_values = pred_slice.flatten()
                    target_values = target_slice.flatten()
                
                # Remove NaN values
                pred_values = pred_values[~np.isnan(pred_values)]
                target_values = target_values[~np.isnan(target_values)]
                
                if len(pred_values) > 0 and len(target_values) > 0:
                    # Calculate Wasserstein distance
                    w_distance = wasserstein_distance(pred_values, target_values)
                    wasserstein_per_var_time[v, t] += w_distance / batch_size
    
    return wasserstein_per_var_time

def compute_rapsd(data: Union[np.ndarray, torch.Tensor], 
                             resolution_km: float,
                             channel_names: List[str] = ["Temperature", "Wind Speed"]):
    """
    Compute RAPSD for your specific data format [BATCH, 24, 64, 64] where 
    channels and timesteps are interleaved (12 timesteps * 2 channels = 24).
    
    Parameters:
    -----------
    data : np.ndarray or torch.Tensor
        Shape should be (batch_size, 24, 64, 64) where 24 = 12 timesteps * 2 channels
    resolution_km : float
        Spatial resolution in kilometers per grid cell
    channel_names : List[str]
        Names of channels for plotting
    
    Returns:
    --------
    wavelengths : np.ndarray
        Wavelengths in kilometers
    rapsd_values : List[np.ndarray]
        RAPSD values for each channel, averaged over timesteps and batches
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    batch_size, total_steps, height, width = data.shape
    n_channels = 2
    timesteps = total_steps // n_channels
    channel_rapsds = [[[[] for _ in range(timesteps)] for _ in range(n_channels)] for _ in range(batch_size)]
    for b in range(batch_size):
        for t in range(timesteps):
            for c in range(n_channels):
                idx = t * n_channels + c
                field = data[b, idx]
                spectrum, radii = rapsd(field, fft_method=np.fft, return_freq=True)
                channel_rapsds[b][c][t].append(spectrum)  
    channel_rapsds = np.array(channel_rapsds)  # (batch_size, n_channels, timesteps, spectrum_length)

    avg_rapsd_batch = np.mean(channel_rapsds, axis=(2))  # Shape: (n_channels, spectrum_length)

    wavelengths = resolution_km * (2.0 * np.pi) / radii  # Use `radii`, which is fixed
    return wavelengths, radii,  avg_rapsd_batch, channel_rapsds

def calculate_anomaly_correlation(prediction, target, mask=None, climatology=None):
    """
    Calculate Anomaly Correlation Coefficient (ACC) between prediction and target.
    
    Args:
        prediction: Tensor of shape [batch, channels, height, width] or [channels, height, width]
        target: Tensor with the same shape as prediction
        mask: Optional mask for valid data points
        climatology: Reference climatology for calculating anomalies
        
    Returns:
        ACC per channel and timestep
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Ensure we have batch dimension
    if prediction.ndim == 3:
        prediction = np.expand_dims(prediction, axis=0)
        target = np.expand_dims(target, axis=0)
        
    batch_size, channels, height, width = prediction.shape
    
    num_variables = 2  # T2m and U10m
    window = channels // num_variables
    
    # First reshape to separate the interleaved structure
    prediction_reshaped = prediction.reshape(batch_size, window, num_variables, height, width)
    target_reshaped = target.reshape(batch_size, window, num_variables, height, width)
    
    # Then transpose to get [batch, var, time, height, width]
    prediction_reshaped = np.transpose(prediction_reshaped, (0, 2, 1, 3, 4))
    target_reshaped = np.transpose(target_reshaped, (0, 2, 1, 3, 4))
    
    # If climatology not provided, use mean of target
    if climatology is None:
        if mask is not None:
            mask_expanded = np.expand_dims(mask, axis=0) if mask.ndim == 3 else mask
            mask_expanded = np.repeat(mask_expanded, batch_size * num_variables * window, axis=0)
            mask_reshaped = mask_expanded.reshape(batch_size, num_variables, window, height, width)
            
            # Calculate climatology as mean over batch for each variable and time step
            climatology = np.zeros((num_variables, window, height, width))
            for v in range(num_variables):
                for t in range(window):
                    valid_data = target_reshaped[:, v, t] * mask_reshaped[:, v, t]
                    valid_count = np.sum(mask_reshaped[:, v, t] > 0, axis=0)
                    valid_count[valid_count == 0] = 1  # Avoid division by zero
                    climatology[v, t] = np.sum(valid_data, axis=0) / valid_count
        else:
            # Calculate climatology as mean over batch for each variable and time step
            climatology = np.mean(target_reshaped, axis=0)
    
    acc_per_var_time = np.zeros((num_variables, window))
    
    for v in range(num_variables):
        for t in range(window):
            for b in range(batch_size):
                pred_slice = prediction_reshaped[b, v, t]
                target_slice = target_reshaped[b, v, t]
                clim_slice = climatology[v, t]
                
                # Calculate anomalies
                pred_anomaly = pred_slice - clim_slice
                target_anomaly = target_slice - clim_slice
                
                if mask is not None:
                    # Apply mask
                    mask_slice = mask[0] if mask.ndim == 4 else mask
                    pred_anomaly = pred_anomaly * mask_slice
                    target_anomaly = target_anomaly * mask_slice
                
                # Flatten arrays
                pred_anomaly_flat = pred_anomaly.flatten()
                target_anomaly_flat = target_anomaly.flatten()
                
                if mask is not None:
                    mask_flat = mask_slice.flatten()
                    pred_anomaly_flat = pred_anomaly_flat[mask_flat > 0]
                    target_anomaly_flat = target_anomaly_flat[mask_flat > 0]
                
                # Calculate correlation
                if len(pred_anomaly_flat) > 0:
                    # Remove NaN values
                    valid_indices = ~(np.isnan(pred_anomaly_flat) | np.isnan(target_anomaly_flat))
                    pred_valid = pred_anomaly_flat[valid_indices]
                    target_valid = target_anomaly_flat[valid_indices]
                    
                    if len(pred_valid) > 0:
                        correlation = np.corrcoef(pred_valid, target_valid)[0, 1]
                        if not np.isnan(correlation):
                            acc_per_var_time[v, t] += correlation / batch_size
    
    return acc_per_var_time


def calculate_energy_score(predictions, target, mask=None):
    """
    Calculate Energy Score for ensemble predictions vs target.
    Energy Score is a multivariate generalization of CRPS.
    
    Args:
        predictions: List of tensors, each with shape [channels, height, width]
        target: Tensor with the same shape as each prediction
        mask: Optional mask for valid data points
        
    Returns:
        Energy score value (lower is better)
    """
    if not isinstance(predictions, list):
        raise ValueError("predictions should be a list of ensemble members")
    
    # Convert to numpy
    preds_np = [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p for p in predictions]
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Ensure all predictions have the same shape
    n_ensemble = len(preds_np)
    
    # Flatten spatial dimensions for each channel and apply mask if provided
    if mask is not None:
        flat_mask = mask.flatten()
        valid_indices = flat_mask > 0
        
        preds_flat = [p.reshape(p.shape[0], -1)[:, valid_indices] for p in preds_np]
        target_flat = target.reshape(target.shape[0], -1)[:, valid_indices]
    else:
        preds_flat = [p.reshape(p.shape[0], -1) for p in preds_np]
        target_flat = target.reshape(target.shape[0], -1)
    
    # Calculate energy score
    es_term1 = 0
    for i in range(n_ensemble):
        for j in range(i+1, n_ensemble):
            # Calculate Euclidean distance between ensemble members
            diff = preds_flat[i] - preds_flat[j]
            es_term1 += np.sqrt(np.sum(diff**2, axis=0)).mean()
    
    es_term1 = es_term1 * 2 / (n_ensemble * (n_ensemble - 1))
    
    es_term2 = 0
    for i in range(n_ensemble):
        # Calculate Euclidean distance between ensemble member and target
        diff = preds_flat[i] - target_flat
        es_term2 += np.sqrt(np.sum(diff**2, axis=0)).mean()
    
    es_term2 = es_term2 / n_ensemble
    
    # Energy score is term2 - 0.5*term1
    energy_score = es_term2 - 0.5 * es_term1
    
    return energy_score

def calculate_sample_variance(prediction,  mask=None, num_variables=2):
    if mask is not None:
        prediction = prediction * mask 
    print('hello')

    batch_size, channels, height, width = prediction.shape

    window = channels // num_variables
    
    # [batch, window*channels, height, width] -> [batch, window, num_variables, height, width]
    prediction_reshaped = prediction.reshape(batch_size, window, num_variables, height, width)
   
    # [batch, num_variables, window, height, width]
    prediction_reshaped = prediction_reshaped.permute(0, 2, 1, 3, 4)
    
    # Variance and RMSEs metrics
    prediction_mean  = prediction_reshaped.mean(dim=0) # [num_variables, window, height, width]
    print(prediction_mean.shape)
    variance_sample = torch.zeros((batch_size, num_variables,window, height, width))
    variance_per_variable = torch.zeros((num_variables))
    for i  in range(batch_size):
        for var in range(num_variables):
            variance_sample[i, var] = (prediction_reshaped[i,var] - prediction_mean[var]).pow(2)
    valid_points = mask.sum() * batch_size * window
    if valid_points > 0:
        for v in range(num_variables):
            variance_per_variable[v] = variance_sample[:, v].sum() / valid_points

    return variance_per_variable



def calculate_metrics(predictions, target, mask=None, var_names=None, metric_names=None):
    """
    Calculate multiple metrics between predictions and target.
    
    Args:
        predictions: Single tensor or list of tensors (for ensemble metrics)
        target: Ground truth tensor
        mask: Optional mask for valid data points
        var_names: Names of variables (default: ['T2m', 'U10m'])
        metric_names: List of metrics to calculate (default: all available)
        
    Returns:
        Dictionary containing all computed metrics
    """
    if var_names is None:
        var_names = ['T2m', 'U10m']
    
    if metric_names is None:
        metric_names = ['rmse', 'mae', 'ssim', 'wasserstein', 'anomaly_correlation']
    
    results = {}
    
    # For single prediction metrics
    if not isinstance(predictions, list):
        single_pred = predictions
        
        if 'rmse' in metric_names:
            rmse_per_var_time, overall_rmse = calculate_rmse(single_pred, target, mask)
            results['rmse'] = {
                'per_var_time': rmse_per_var_time,
                'overall': overall_rmse
            }
            
        if 'mae' in metric_names:
            mae_per_var_time, overall_mae = calculate_mae(single_pred, target, mask)
            results['mae'] = {
                'per_var_time': mae_per_var_time,
                'overall': overall_mae
            }
            
        if 'ssim' in metric_names:
            ssim_per_var_time = calculate_ssim(single_pred, target, mask)
            results['ssim'] = {
                'per_var_time': ssim_per_var_time
            }
            
        if 'wasserstein' in metric_names:
            wasserstein_per_var_time = calculate_wasserstein(single_pred, target, mask)
            results['wasserstein'] = {
                'per_var_time': wasserstein_per_var_time
            }
            
        if 'anomaly_correlation' in metric_names:
            acc_per_var_time = calculate_anomaly_correlation(single_pred, target, mask)
            results['anomaly_correlation'] = {
                'per_var_time': acc_per_var_time
            }
    
    # For ensemble prediction metrics
    else:
        if 'energy_score' in metric_names:
            es = calculate_energy_score(predictions, target, mask)
            results['energy_score'] = es
        
        if 'sample_variance' in metric_names:
            sample_variance = calculate_sample_variance(predictions, target, mask)
            results['sample_variance'] = sample_variance
        
        # Calculate single-prediction metrics for each ensemble member
        ensemble_results = []
        for i, pred in enumerate(predictions):
            member_results = calculate_metrics(
                pred, target, mask, var_names, 
                [m for m in metric_names if m != 'energy_score' and m != 'sample_variance']
            )
            ensemble_results.append(member_results)
        
        results['ensemble'] = ensemble_results
    
    return results


def plot_metric_comparison(metrics_list, sample_ids=None, var_names=None):
    """
    Plot comparison of metrics across different samples.
    
    Args:
        metrics_list: Single metrics dictionary or list of dictionaries, each containing metrics from calculate_metrics()
        sample_ids: List of sample identifiers
        var_names: List of variable names
        
    Returns:
        Figure with metric comparisons
    """
    import matplotlib.pyplot as plt
    
    # Convert single dictionary to list if needed
    if not isinstance(metrics_list, list):
        metrics_list = [metrics_list]
    
    if var_names is None:
        var_names = ['T2m', 'U10m']
    
    if sample_ids is None:
        sample_ids = [f'Sample {i+1}' for i in range(len(metrics_list))]
    

    all_metrics = set()
    for metric_dict in metrics_list:
        if isinstance(metric_dict, dict):  
            for metric_name, metric_data in metric_dict.items():
                if isinstance(metric_data, dict) and 'per_var_time' in metric_data.keys():
                    all_metrics.add(metric_name)
    print(all_metrics)
    var_time_metrics = sorted(list(all_metrics))
    single_value_metrics = ['energy_score']
    
    n_samples = len(metrics_list)
    n_metrics = len(var_time_metrics)
    n_vars = len(var_names)
    
    if n_metrics > 0:
        # Create subplots grid
        fig, axes = plt.subplots(n_metrics, n_vars, figsize=(5*n_vars, 4*n_metrics))
        if n_metrics == 1 and n_vars == 1:
            axes = np.array([[axes]])
        elif n_metrics == 1:
            axes = axes.reshape(1, -1)
        elif n_vars == 1:
            axes = axes.reshape(-1, 1)
        
        # Find the window size (time steps) from the first available metric
        window = None
        for metric_dict in metrics_list:
            if not isinstance(metric_dict, dict):
                continue
                
            for metric_name in var_time_metrics:
                if metric_name in metric_dict and 'per_var_time' in metric_dict[metric_name]:
                    per_var_time = metric_dict[metric_name]['per_var_time']
                    if isinstance(per_var_time, torch.Tensor) or isinstance(per_var_time, np.ndarray):
                        window = per_var_time.shape[1]
                        break
            if window is not None:
                break
        
        if window is None:
            raise ValueError("Could not determine window size from metrics")
        
        # Plot each metric
        for m_idx, metric_name in enumerate(var_time_metrics):
            for v_idx, var_name in enumerate(var_names):
                ax = axes[m_idx, v_idx]
                
                for s_idx, (metric_dict, sample_id) in enumerate(zip(metrics_list, sample_ids)):
                    if not isinstance(metric_dict, dict):
                        continue
                        
                    if metric_name in metric_dict and 'per_var_time' in metric_dict[metric_name]:
                        metric_values = metric_dict[metric_name]['per_var_time']
                        if isinstance(metric_values, torch.Tensor):
                            metric_values = metric_values.detach().cpu().numpy()
                            
                        # Plot metric over time
                        if v_idx < metric_values.shape[0]:  # Check if this variable exists in metrics
                            ax.plot(range(window), metric_values[v_idx], label=sample_id)
                
                ax.set_title(f'{metric_name.upper()} - {var_name}')
                ax.set_xlabel('Time Step')
                ax.set_ylabel(metric_name.upper())
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    # For single value metrics or ensemble metrics
    '''
    elif any(isinstance(metric_dict, dict) and metric in metric_dict 
             for metric_dict in metrics_list for metric in single_value_metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric_name in single_value_metrics:
            values = []
            valid_samples = []
            
            for s_idx, (metric_dict, sample_id) in enumerate(zip(metrics_list, sample_ids)):
                if isinstance(metric_dict, dict) and metric_name in metric_dict:
                    values.append(metric_dict[metric_name])
                    valid_samples.append(sample_id)
            
            if values:
                ax.bar(valid_samples, values, label=metric_name.upper())
        
        ax.set_title('Single Value Metrics Comparison')
        ax.set_ylabel('Metric Value')
        ax.set_xlabel('Sample')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    '''
    return None
