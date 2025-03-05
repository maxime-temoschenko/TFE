import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
from torch.nn.functional import mse_loss, l1_loss


def calculate_rmse(prediction, target, mask=None, num_variables=2):
    """
    Calculate Root Mean Square Error between prediction and target.
    
    Args:
        prediction: Tensor of shape [batch, window*channels, height, width] or [window*channels, height, width]
        target: Tensor with the same shape as prediction
        mask: Optional mask for valid data points   
    Returns:
        RMSE per channel and timestep, and overall RMSE
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
    print(prediction_reshaped.shape)
    prediction_reshaped = prediction_reshaped.permute(0, 2, 1, 3, 4)
    target_reshaped = target_reshaped.permute(0, 2, 1, 3, 4)
    

    rmse_per_var_time = torch.zeros(num_variables, window)
    for v in range(num_variables):
        for t in range(window):
            if mask is not None:
                valid_points = mask.sum()
                if valid_points > 0:
                    error = ((prediction_reshaped[:, v, t] - target_reshaped[:, v, t]) ** 2).sum() / valid_points
                    rmse_per_var_time[v, t] = torch.sqrt(error)
            else:
                error = mse_loss(prediction_reshaped[:, v, t], target_reshaped[:, v, t])
                rmse_per_var_time[v, t] = torch.sqrt(error)
    
    # Overall RMSE
    if mask is not None:
        valid_points = mask.sum() * batch_size * channels
        if valid_points > 0:
            overall_error = ((prediction - target) ** 2).sum() / valid_points
            overall_rmse = torch.sqrt(overall_error)
        else:
            overall_rmse = torch.tensor(float('nan'))
    else:
        overall_error = mse_loss(prediction, target)
        overall_rmse = torch.sqrt(overall_error)
    
    return rmse_per_var_time, overall_rmse


def calculate_mae(prediction, target, mask=None, num_variables=2):
    """
    Calculate Mean Absolute Error between prediction and target.
    
    Args:
        prediction: Tensor of shape [batch, window*channels, height, width] or [window*channels, height, width]
        target: Tensor with the same shape as prediction
        mask: Optional mask for valid data points
        
    Returns:
        MAE per channel and timestep, and overall MAE
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
    
    mae_per_var_time = torch.zeros(num_variables, window)
    
    for v in range(num_variables):
        for t in range(window):
            if mask is not None:
                valid_points = mask.sum()
                if valid_points > 0:
                    error = (prediction_reshaped[:, v, t] - target_reshaped[:, v, t]).abs().sum() / valid_points
                    mae_per_var_time[v, t] = error
            else:
                error = l1_loss(prediction_reshaped[:, v, t], target_reshaped[:, v, t])
                mae_per_var_time[v, t] = error
    
    if mask is not None:
        valid_points = mask.sum() * batch_size * channels
        if valid_points > 0:
            overall_error = (prediction - target).abs().sum() / valid_points
        else:
            overall_error = torch.tensor(float('nan'))
    else:
        overall_error = l1_loss(prediction, target)
    
    return mae_per_var_time, overall_error


def calculate_ssim(prediction, target, mask=None):
    """
    Calculate Structural Similarity Index (SSIM) between prediction and target.
    
    Args:
        prediction: Tensor of shape [batch, channels, height, width] or [channels, height, width]
        target: Tensor with the same shape as prediction
        mask: Optional mask for valid data points
        
    Returns:
        SSIM per channel and timestep
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
    
    ssim_per_var_time = np.zeros((num_variables, window))
    
    for v in range(num_variables):
        for t in range(window):
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
                    
                    ssim_per_var_time[v, t] += ssim_value / batch_size
    
    return ssim_per_var_time


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

def calculate_sample_variance(predictions, target, mask=None, num_variables=2):
    """
    Calculate variance and dispersion statistics of ensemble predictions with respect to ground truth.
    The batch dimension represents different ensemble members.
    
    Args:
        predictions: Tensor with shape [batch/ensemble, window*channels, height, width]
                    where batch dimension represents different ensemble members
        target: Ground truth tensor with shape [window*channels, height, width] or
                [1, window*channels, height, width]
        mask: Optional mask for valid data points
        num_variables: Number of variables in the data (default: 2)
        
    Returns:
        Dictionary containing variance statistics:
            - sample_variance: Variance across samples for each variable and timestep
            - mean_deviation: Mean absolute deviation from ground truth
            - ensemble_mean_rmse: Error of the ensemble mean prediction vs ground truth
            - spread_skill_ratio: Ratio of ensemble spread to error (ideal value is 1.0)
            - global_var_per_var: Global variance across all timesteps for each variable 
            - global_var_all: Global variance across all variables and timesteps
    """
    # Convert to numpy arrays if they are torch tensors
    if isinstance(predictions, torch.Tensor):
        predictions_np = predictions.detach().cpu().numpy()
    else:
        predictions_np = predictions
        
    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = target
        
    if mask is not None and isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    
    # Ensure target has batch dimension (of size 1)
    if target_np.ndim == 3:
        target_np = np.expand_dims(target_np, axis=0)
    
    # Get dimensions
    n_samples, channels, height, width = predictions_np.shape
    window = channels // num_variables
    
    # Reshape predictions and target to [batch/ensemble, var, time, height, width]
    pred_reshaped = predictions_np.reshape(n_samples, window, num_variables, height, width)
    pred_reshaped = np.transpose(pred_reshaped, (0, 2, 1, 3, 4))
    
    target_reshaped = target_np.reshape(target_np.shape[0], window, num_variables, height, width)
    target_reshaped = np.transpose(target_reshaped, (0, 2, 1, 3, 4))
    
    # Calculate ensemble mean
    ensemble_mean = np.mean(pred_reshaped, axis=0, keepdims=True)  # [1, var, time, height, width]
    
    # Initialize arrays for results
    sample_variance = np.zeros((num_variables, window))
    mean_deviation = np.zeros((num_variables, window))
    ensemble_mean_rmse = np.zeros((num_variables, window))
    spread_skill_ratio = np.zeros((num_variables, window))
    
    # Calculate statistics for each variable and timestep
    for v in range(num_variables):
        for t in range(window):
            # Extract all sample predictions and ground truth
            sample_preds = pred_reshaped[:, v, t]  # [n_samples, height, width]
            gt = target_reshaped[0, v, t]  # [height, width]
            
            # Apply mask if provided
            if mask_np is not None:
                mask_slice = mask_np if mask_np.ndim == 2 else mask_np.squeeze()
                valid_mask = mask_slice > 0
                
                if valid_mask.sum() > 0:
                    # Apply mask to each sample and ground truth
                    sample_values = np.array([sample[valid_mask] for sample in sample_preds])
                    gt_values = gt[valid_mask]
                    
                    # Calculate variance across samples
                    sample_variance[v, t] = np.var(sample_values, axis=0).mean()
                    
                    # Calculate mean absolute deviation from ground truth
                    mean_deviation[v, t] = np.mean(np.abs(sample_values - gt_values))
                    
                    # Get ensemble mean RMSE
                    ens_mean_values = ensemble_mean[0, v, t][valid_mask]
                    ensemble_mean_rmse[v, t] = np.sqrt(np.mean((ens_mean_values - gt_values) ** 2))
            else:
                # Flatten spatial dimensions
                sample_values = sample_preds.reshape(n_samples, -1)
                gt_values = gt.flatten()
                
                # Remove NaN values
                valid_indices = ~np.isnan(gt_values)
                for i in range(n_samples):
                    valid_indices = valid_indices & ~np.isnan(sample_values[i])
                
                if np.any(valid_indices):
                    sample_values = sample_values[:, valid_indices]
                    gt_values = gt_values[valid_indices]
                    
                    sample_variance[v, t] = np.var(sample_values, axis=0).mean()
                    mean_deviation[v, t] = np.mean(np.abs(sample_values - gt_values))
                    
                    ens_mean_values = ensemble_mean[0, v, t].flatten()[valid_indices]
                    ensemble_mean_rmse[v, t] = np.sqrt(np.mean((ens_mean_values - gt_values) ** 2))
            
            # Calculate spread-skill ratio
            if ensemble_mean_rmse[v, t] > 0:
                ensemble_spread = np.sqrt(sample_variance[v, t])
                spread_skill_ratio[v, t] = ensemble_spread / ensemble_mean_rmse[v, t]
    
    # Calculate global variance per variable (across all timesteps)
    global_var_per_var = np.zeros(num_variables)
    
    # Calculate global variance across all variables and timesteps
    all_samples_all_vars = []
    all_gt_all_vars = []
    
    for v in range(num_variables):
        all_samples_var = []
        all_gt_var = []
        
        for t in range(window):
            if mask_np is not None:
                mask_slice = mask_np if mask_np.ndim == 2 else mask_np.squeeze()
                valid_mask = mask_slice > 0
                
                if valid_mask.sum() > 0:
                    for s in range(n_samples):
                        # Get values for this sample, variable, timestep at valid locations
                        sample_values = pred_reshaped[s, v, t][valid_mask]
                        
                        # Remove NaN values
                        valid_indices = ~np.isnan(sample_values)
                        if np.any(valid_indices):
                            all_samples_var.extend(sample_values[valid_indices])
                            
                            if len(all_samples_all_vars) <= s:
                                all_samples_all_vars.append([])
                            all_samples_all_vars[s].extend(sample_values[valid_indices])
                    
                    # Get ground truth values for this variable, timestep at valid locations
                    gt_values = target_reshaped[0, v, t][valid_mask]
                    valid_indices = ~np.isnan(gt_values)
                    if np.any(valid_indices):
                        all_gt_var.extend(gt_values[valid_indices])
                        all_gt_all_vars.extend(gt_values[valid_indices])
            else:
                for s in range(n_samples):
                    # Flatten and get values for this sample, variable, timestep
                    sample_values = pred_reshaped[s, v, t].flatten()
                    
                    # Remove NaN values
                    valid_indices = ~np.isnan(sample_values)
                    if np.any(valid_indices):
                        all_samples_var.extend(sample_values[valid_indices])
                        
                        if len(all_samples_all_vars) <= s:
                            all_samples_all_vars.append([])
                        all_samples_all_vars[s].extend(sample_values[valid_indices])
                
                # Get ground truth values for this variable, timestep
                gt_values = target_reshaped[0, v, t].flatten()
                valid_indices = ~np.isnan(gt_values)
                if np.any(valid_indices):
                    all_gt_var.extend(gt_values[valid_indices])
                    all_gt_all_vars.extend(gt_values[valid_indices])
        
        # Calculate global variance for this variable
        if all_samples_var:
            # Reshape to [n_samples, n_points]
            all_samples_var_array = np.array([all_samples_var]).reshape(n_samples, -1)
            global_var_per_var[v] = np.var(all_samples_var_array, axis=0).mean()
    
    # Calculate global variance across all variables and timesteps
    global_var_all = 0
    if all_samples_all_vars:
        # Make sure each sample has the same length
        min_length = min(len(arr) for arr in all_samples_all_vars)
        all_samples_all_vars_array = np.array([arr[:min_length] for arr in all_samples_all_vars])
        global_var_all = np.var(all_samples_all_vars_array, axis=0).mean()
    
    return {
        'sample_variance': sample_variance,                 # variance per variable and timestep
        'mean_deviation': mean_deviation,                   # mean abs deviation from ground truth
        'ensemble_mean_rmse': ensemble_mean_rmse,           # RMSE of ensemble mean vs ground truth
        'spread_skill_ratio': spread_skill_ratio,           # ratio of ensemble spread to error
        'global_var_per_var': global_var_per_var,           # global variance per variable (across all timesteps)
        'global_var_all': global_var_all                    # global variance across all vars and timesteps
    }


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
