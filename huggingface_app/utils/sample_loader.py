"""
Sample Image Loader
===================

Provides sample images for demonstration.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Sample image directory
SAMPLE_DIR = Path(__file__).parent.parent / "samples"


def get_sample_images() -> Dict[str, List[str]]:
    """Get list of available sample images by category."""
    samples = {
        "uhi": [],
        "lulc": [],
        "plastic": [],
        "xai": [],
    }
    
    if SAMPLE_DIR.exists():
        for category in samples.keys():
            category_dir = SAMPLE_DIR / category
            if category_dir.exists():
                samples[category] = [
                    f.name for f in category_dir.iterdir() 
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
                ]
    
    return samples


def load_sample(category: str) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a sample image for the specified category.
    
    If real samples exist, loads them. Otherwise, generates synthetic samples.
    
    Args:
        category: One of 'uhi', 'lulc', 'plastic', 'xai'
        
    Returns:
        Sample image(s) as numpy array(s)
    """
    sample_path = SAMPLE_DIR / category
    
    # Try to load real sample if exists
    if sample_path.exists():
        images = list(sample_path.glob("*.png")) + list(sample_path.glob("*.jpg"))
        if images:
            from PIL import Image
            img = Image.open(images[0]).convert("RGB")
            return np.array(img)
    
    # Generate synthetic sample
    if category == "uhi":
        return generate_synthetic_thermal()
    elif category == "lulc":
        return generate_synthetic_lulc_pair()
    elif category == "plastic":
        return generate_synthetic_water()
    elif category == "xai":
        return generate_synthetic_satellite()
    else:
        return generate_synthetic_satellite()


def generate_synthetic_thermal(size: int = 256) -> np.ndarray:
    """Generate synthetic thermal image with heat patterns."""
    from scipy.ndimage import gaussian_filter
    
    # Base temperature field
    y, x = np.ogrid[:size, :size]
    
    # Urban heat island in center
    center_y, center_x = size // 2, size // 2
    urban = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * (size//4)**2))
    
    # Add some variation
    noise = np.random.rand(size, size)
    noise = gaussian_filter(noise, sigma=10)
    
    # Combine
    thermal = 0.3 + 0.4 * urban + 0.3 * noise
    
    # Add hot spots
    for _ in range(5):
        hy, hx = np.random.randint(size//4, 3*size//4, 2)
        hot_spot = np.exp(-((y - hy)**2 + (x - hx)**2) / (2 * (size//16)**2))
        thermal += hot_spot * 0.15
    
    # Normalize to 0-255
    thermal = ((thermal - thermal.min()) / (thermal.max() - thermal.min()) * 255).astype(np.uint8)
    
    # Create grayscale "thermal" image
    return np.stack([thermal, thermal, thermal], axis=-1)


def generate_synthetic_lulc_pair(size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic before/after images showing urban expansion."""
    from scipy.ndimage import gaussian_filter
    
    # Create base landscape
    y, x = np.ogrid[:size, :size]
    
    # Vegetation base (green)
    vegetation = np.ones((size, size, 3), dtype=np.uint8)
    vegetation[:, :, 0] = 50  # R
    vegetation[:, :, 1] = 120  # G
    vegetation[:, :, 2] = 50  # B
    
    # Add texture to vegetation
    texture = np.random.rand(size, size)
    texture = gaussian_filter(texture, sigma=3)
    vegetation[:, :, 1] = (120 + texture * 30).astype(np.uint8)
    
    # BEFORE: Small urban center
    before = vegetation.copy()
    urban_radius_before = size // 6
    urban_mask = ((y - size//2)**2 + (x - size//2)**2) < urban_radius_before**2
    before[urban_mask] = [180, 180, 180]  # Gray urban
    
    # Add some roads
    road_width = 3
    before[size//2-road_width:size//2+road_width, :] = [100, 100, 100]
    before[:, size//2-road_width:size//2+road_width] = [100, 100, 100]
    
    # AFTER: Expanded urban area
    after = vegetation.copy()
    urban_radius_after = size // 3
    urban_mask = ((y - size//2)**2 + (x - size//2)**2) < urban_radius_after**2
    after[urban_mask] = [200, 200, 200]  # Gray urban
    
    # Add more roads
    after[size//2-road_width:size//2+road_width, :] = [100, 100, 100]
    after[:, size//2-road_width:size//2+road_width] = [100, 100, 100]
    
    # Add sprawl patterns
    for _ in range(10):
        sy, sx = np.random.randint(0, size, 2)
        sprawl_size = np.random.randint(10, 30)
        sprawl_mask = ((y - sy)**2 + (x - sx)**2) < sprawl_size**2
        after[sprawl_mask] = [190, 190, 190]
    
    return before, after


def generate_synthetic_water(size: int = 256) -> np.ndarray:
    """Generate synthetic image with water body and debris."""
    from scipy.ndimage import gaussian_filter
    
    y, x = np.ogrid[:size, :size]
    
    # Land (brownish-green)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = [100, 120, 80]  # Base land color
    
    # Water body (lake/river shape)
    # Create irregular water shape
    water_mask = np.zeros((size, size), dtype=bool)
    
    # Main water body
    center_y = size // 2 + np.random.randint(-size//8, size//8)
    for i in range(size):
        width = int(size//4 + size//8 * np.sin(i * 4 * np.pi / size))
        center_x = size//2 + int(size//8 * np.sin(i * 2 * np.pi / size))
        left = max(0, center_x - width)
        right = min(size, center_x + width)
        if center_y - size//8 < i < center_y + size//8:
            water_mask[i, left:right] = True
    
    # Expand water area
    from scipy.ndimage import binary_dilation
    water_mask = binary_dilation(water_mask, iterations=5)
    water_mask = gaussian_filter(water_mask.astype(float), sigma=3) > 0.3
    
    # Color water (blue with variation)
    water_blue = 150 + gaussian_filter(np.random.rand(size, size), sigma=5) * 50
    image[water_mask, 0] = 30
    image[water_mask, 1] = 80
    image[water_mask, 2] = water_blue[water_mask].astype(np.uint8)
    
    # Add floating debris/plastic (yellowish spots)
    debris_count = np.random.randint(5, 15)
    for _ in range(debris_count):
        dy = np.random.randint(0, size)
        dx = np.random.randint(0, size)
        if water_mask[dy, dx]:
            debris_size = np.random.randint(3, 8)
            debris_mask = ((y - dy)**2 + (x - dx)**2) < debris_size**2
            image[debris_mask & water_mask, 0] = 200
            image[debris_mask & water_mask, 1] = 180
            image[debris_mask & water_mask, 2] = 100
    
    return image


def generate_synthetic_satellite(size: int = 256) -> np.ndarray:
    """Generate synthetic satellite-like RGB image."""
    from scipy.ndimage import gaussian_filter
    
    y, x = np.ogrid[:size, :size]
    
    # Create varied landscape
    image = np.zeros((size, size, 3), dtype=np.float32)
    
    # Vegetation (green tones)
    veg_mask = np.random.rand(size, size) > 0.4
    veg_mask = gaussian_filter(veg_mask.astype(float), sigma=15) > 0.5
    image[veg_mask, 0] = 50 + np.random.rand() * 30
    image[veg_mask, 1] = 100 + np.random.rand() * 50
    image[veg_mask, 2] = 40 + np.random.rand() * 20
    
    # Urban (gray tones)
    urban_center = (size//3, size//2)
    urban_mask = ((y - urban_center[0])**2 + (x - urban_center[1])**2) < (size//4)**2
    image[urban_mask, 0] = 150 + np.random.rand() * 30
    image[urban_mask, 1] = 150 + np.random.rand() * 30
    image[urban_mask, 2] = 150 + np.random.rand() * 30
    
    # Water (blue)
    water_center = (2*size//3, size//2)
    water_mask = ((y - water_center[0])**2 + (x - water_center[1])**2) < (size//5)**2
    image[water_mask, 0] = 30
    image[water_mask, 1] = 60
    image[water_mask, 2] = 120
    
    # Add texture
    texture = gaussian_filter(np.random.rand(size, size), sigma=2) * 20
    for c in range(3):
        image[:, :, c] += texture
    
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image


def save_sample(image: np.ndarray, category: str, filename: str) -> Path:
    """Save a sample image."""
    from PIL import Image
    
    category_dir = SAMPLE_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = category_dir / filename
    Image.fromarray(image).save(save_path)
    
    return save_path
