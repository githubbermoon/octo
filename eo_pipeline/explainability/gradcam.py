"""
GradCAM for Visual Explanations

Provides Gradient-weighted Class Activation Mapping for
visualizing what spatial regions influence model predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Generates heatmaps showing which image regions are most important
    for a particular prediction.
    
    Example:
        >>> gradcam = GradCAM(model, target_layer="encoder.layer4")
        >>> heatmap = gradcam.generate(image, target_class=5)
        >>> overlay = gradcam.overlay_on_image(image, heatmap)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Union[str, nn.Module],
        use_cuda: bool = True
    ):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layer: Name or module of target layer for CAM
            use_cuda: Use CUDA if available
        """
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if self.use_cuda:
            self.model = self.model.cuda()
        
        # Get target layer
        if isinstance(target_layer, str):
            self.target_layer = self._get_layer_by_name(target_layer)
        else:
            self.target_layer = target_layer
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _get_layer_by_name(self, name: str) -> nn.Module:
        """Get layer by name using dot notation."""
        names = name.split('.')
        module = self.model
        for n in names:
            module = getattr(module, n)
        return module
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap.
        
        Args:
            input_tensor: Input image (B, C, H, W) or (C, H, W)
            target_class: Target class (None = predicted class)
            normalize: Normalize heatmap to [0, 1]
            
        Returns:
            Heatmap array (H, W) or (B, H, W)
        """
        # Handle single image
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        
        # Forward pass
        self.model.eval()
        input_tensor.requires_grad = True
        output = self.model(input_tensor)
        
        # Handle segmentation output
        if output.dim() == 4:
            # For segmentation: use spatial average
            output = output.mean(dim=(2, 3))
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Create one-hot for target class
        one_hot = torch.zeros_like(output)
        for i in range(output.shape[0]):
            if isinstance(target_class, int):
                one_hot[i, target_class] = 1
            else:
                one_hot[i, target_class[i]] = 1
        
        # Backward pass
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute GradCAM
        heatmaps = self._compute_cam()
        
        if normalize:
            heatmaps = self._normalize(heatmaps)
        
        # Resize to input size
        h, w = input_tensor.shape[2:]
        heatmaps = F.interpolate(
            torch.tensor(heatmaps).unsqueeze(1),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1).numpy()
        
        if heatmaps.shape[0] == 1:
            return heatmaps[0]
        return heatmaps
    
    def _compute_cam(self) -> np.ndarray:
        """Compute CAM from gradients and activations."""
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1)
        
        # ReLU (only positive contributions)
        cam = F.relu(cam)
        
        return cam.cpu().numpy()
    
    def _normalize(self, heatmap: np.ndarray) -> np.ndarray:
        """Normalize heatmap to [0, 1]."""
        for i in range(heatmap.shape[0]):
            hmin, hmax = heatmap[i].min(), heatmap[i].max()
            if hmax - hmin > 1e-8:
                heatmap[i] = (heatmap[i] - hmin) / (hmax - hmin)
            else:
                heatmap[i] = 0
        return heatmap
    
    def overlay_on_image(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet"
    ) -> np.ndarray:
        """
        Overlay heatmap on image for visualization.
        
        Args:
            image: Original image (C, H, W) or (H, W, C)
            heatmap: GradCAM heatmap (H, W)
            alpha: Overlay transparency
            colormap: Matplotlib colormap name
            
        Returns:
            Overlaid image
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            
            # Ensure image is HWC
            if image.ndim == 3 and image.shape[0] < image.shape[-1]:
                image = np.transpose(image, (1, 2, 0))
            
            # Get colormap
            cmap = cm.get_cmap(colormap)
            heatmap_colored = cmap(heatmap)[..., :3]
            
            # Normalize image to [0, 1]
            if image.max() > 1:
                image = image / 255.0
            
            # For multi-band images, use first 3 bands as RGB
            if image.shape[-1] > 3:
                image = image[..., :3]
            
            # Overlay
            overlaid = alpha * heatmap_colored + (1 - alpha) * image
            overlaid = np.clip(overlaid, 0, 1)
            
            return overlaid
            
        except ImportError:
            logger.warning("Matplotlib required for overlay visualization")
            return heatmap
    
    def generate_multi_class(
        self,
        input_tensor: torch.Tensor,
        classes: List[int]
    ) -> Dict[int, np.ndarray]:
        """
        Generate GradCAM for multiple classes.
        
        Args:
            input_tensor: Input image
            classes: List of target classes
            
        Returns:
            Dictionary mapping class to heatmap
        """
        heatmaps = {}
        for cls in classes:
            heatmaps[cls] = self.generate(input_tensor.clone(), target_class=cls)
        return heatmaps


class GradCAMPlusPlus(GradCAM):
    """
    GradCAM++ for improved localization.
    
    Uses weighted combination of positive partial derivatives
    for better handling of multiple instances of same class.
    """
    
    def _compute_cam(self) -> np.ndarray:
        """Compute GradCAM++ with improved weighting."""
        # Second derivative weighting
        grad_2 = self.gradients ** 2
        grad_3 = self.gradients ** 3
        
        # Compute alpha weights
        sum_activation = self.activations.sum(dim=(2, 3), keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activation * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # Apply ReLU to gradients
        positive_gradients = F.relu(self.gradients)
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)
        
        return cam.cpu().numpy()


class ScoreCAM(GradCAM):
    """
    Score-CAM: Gradient-free class activation mapping.
    
    Uses activation maps as masks and measures their impact
    on model output. More stable than gradient-based methods.
    """
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate Score-CAM heatmap."""
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        
        # Get activations
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        activations = self.activations
        b, k, h, w = activations.shape
        
        # Upsample activations to input size
        input_h, input_w = input_tensor.shape[2:]
        upsampled = F.interpolate(
            activations,
            size=(input_h, input_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize each activation map
        upsampled = upsampled.reshape(b, k, -1)
        upsampled = (upsampled - upsampled.min(dim=2, keepdim=True)[0]) / \
                    (upsampled.max(dim=2, keepdim=True)[0] - upsampled.min(dim=2, keepdim=True)[0] + 1e-8)
        upsampled = upsampled.reshape(b, k, input_h, input_w)
        
        # Score each activation map
        scores = torch.zeros(b, k)
        if self.use_cuda:
            scores = scores.cuda()
        
        with torch.no_grad():
            # Get base prediction
            base_output = self.model(input_tensor)
            if base_output.dim() == 4:
                base_output = base_output.mean(dim=(2, 3))
            
            if target_class is None:
                target_class = base_output.argmax(dim=1)
            
            # Score each activation as mask
            for i in range(k):
                masked = input_tensor * upsampled[:, i:i+1]
                output = self.model(masked)
                if output.dim() == 4:
                    output = output.mean(dim=(2, 3))
                
                output = F.softmax(output, dim=1)
                
                for j in range(b):
                    tc = target_class[j] if isinstance(target_class, torch.Tensor) else target_class
                    scores[j, i] = output[j, tc]
        
        # Weighted combination
        scores = F.softmax(scores, dim=1)
        cam = (scores.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=1)
        cam = F.relu(cam)
        
        # Resize
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(input_h, input_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        heatmaps = cam.cpu().numpy()
        
        if normalize:
            heatmaps = self._normalize(heatmaps)
        
        if heatmaps.shape[0] == 1:
            return heatmaps[0]
        return heatmaps


class LayerCAM(GradCAM):
    """
    Layer-CAM for class activation at any layer.
    
    More fine-grained than GradCAM while maintaining
    class-discriminative ability.
    """
    
    def _compute_cam(self) -> np.ndarray:
        """Compute Layer-CAM with spatial gradient information."""
        # Element-wise product of gradients and activations
        cam = F.relu(self.gradients * self.activations)
        
        # Sum over channels
        cam = cam.sum(dim=1)
        
        return cam.cpu().numpy()
