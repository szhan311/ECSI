import argparse
from datetime import datetime
from tqdm import tqdm
import torch as th
import k_diffusion as K

def get_workdir(exp):
    workdir = f'./workdir/{exp}'
    return workdir

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
    
def get_time():
    now = datetime.now()
    return now.strftime("%H%M%S")

def get_date():
    now = datetime.now()
    return now.strftime("%m%d")

def compute_feather(extractor, imgs, n, batch_size):
      features_list = []
      for start_idx in tqdm(range(0, n, batch_size), desc="Extracting features"):
            end_idx = min(start_idx + batch_size, n)
            features = extractor(imgs[start_idx: end_idx])
            features = features.detach().cpu()
            features_list.append(features)
      return th.cat(features_list)

import torch
import torch.nn as nn
import torch.nn.functional as F
class FeatureDifferenceCalculator:
    def __init__(self, feature_extractor, device=None):
        """
        Initialize with a pre-trained feature extractor (e.g., ResNet, VGG)
        Args:
            feature_extractor: Pre-trained model for feature extraction
            device: torch.device to use ('cuda' or 'cpu'). If None, automatically selects available device
        """
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move feature extractor to device
        self.feature_extractor = feature_extractor.to(self.device)
        self.feature_extractor.eval()  # Set to evaluation mode
        
    def extract_features(self, batch):
        """
        Extract features from an image batch
        Args:
            batch: Tensor of shape (B, C, H, W)
        Returns:
            features: Tensor of extracted features
        """
        # Move batch to device
        batch = batch.to(self.device)
        
        # Don't use no_grad() for batch1 as we need gradients for it
        features = self.feature_extractor(batch)
        return features
    
    def calculate_feature_difference(self, batch1, batch2):
        """
        Calculate difference between two batches in feature space
        Args:
            batch1: First batch of images (B, C, H, W)
            batch2: Second batch of images (B, C, H, W)
        Returns:
            feature_diff: Difference in feature space
        """
        # Move batches to device
        batch1 = batch1.to(self.device)
        batch2 = batch2.to(self.device)
        
        features1 = self.extract_features(batch1)
        features2 = self.extract_features(batch2)
        
        return features1 - features2
    
    def calculate_residual(self, feature_diff):
        """
        Calculate residual from feature difference
        Args:
            feature_diff: Feature space difference tensor
        Returns:
            residual: Computed residual
        """
        # Ensure feature_diff is on correct device
        feature_diff = feature_diff.to(self.device)
        
        # L2 norm of the feature difference
        residual = torch.norm(feature_diff, p=2, dim=1)
        return residual
    
    def calculate_gradient(self, batch1, batch2, requires_grad=True):
        """
        Calculate gradient of the residual with respect to batch1
        Args:
            batch1: First batch of images to compute gradients with respect to
            batch2: Second batch of images (reference batch, no gradients needed)
            requires_grad: Whether to compute gradients
        Returns:
            gradients: Computed gradients with respect to batch1
        """
        # Move batches to device
        batch1 = batch1.to(self.device)
        batch2 = batch2.to(self.device)
        
        # Ensure batch1 requires gradients since we'll compute derivatives with respect to it
        if requires_grad:
            batch1.requires_grad_(True)
        
        # Ensure batch2 doesn't require gradients as it's just a reference
        batch2 = batch2.detach()
        
        # Calculate feature difference (features1 - features2)
        feature_diff = self.calculate_feature_difference(batch1, batch2)
        
        # Calculate residual
        residual = self.calculate_residual(feature_diff)
        
        # Calculate gradient with respect to batch1
        gradients = torch.autograd.grad(
            outputs=residual.sum(),  # Sum residuals to get scalar for gradient computation
            inputs=batch1,           # Computing gradient with respect to batch1
            create_graph=True,       # Allows computing higher-order derivatives if needed
            retain_graph=True        # Keeps computation graph for potential future backward passes
        )[0]
        
        return gradients
  
def compute_fid(real_imgs, sample_imgs, num_samples, batch_size, device):
      # calculate fid
      extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
      reals_features = compute_feather(extractor, real_imgs, num_samples, batch_size)
      fakes_features = compute_feather(extractor, sample_imgs, num_samples, batch_size)
      fid = K.evaluation.fid(fakes_features, reals_features)
      return fid

def compute_fid_kid(real_imgs, sample_imgs, num_samples, batch_size, device):
      # calculate fid and kid
      extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
      reals_features = compute_feather(extractor, real_imgs, num_samples, batch_size)
      fakes_features = compute_feather(extractor, sample_imgs, num_samples, batch_size)
      fid = K.evaluation.fid(fakes_features, reals_features)
      kid = K.evaluation.kid(fakes_features, reals_features)
      return fid, kid
  

