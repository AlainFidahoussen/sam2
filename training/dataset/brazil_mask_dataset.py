"""
Custom dataset class for Brazil mask data
"""

import glob
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
from iopath.common.file_io import g_pathmgr

from training.dataset.vos_raw_dataset import VOSFrame, VOSRawDataset, VOSVideo
from training.dataset.vos_segment_loader import MultiplePNGSegmentLoader
from PIL import Image
import numpy as np

# Import visualization utilities
try:
    import sys
    sys.path.append('/home/fidaa/Documents/FTI/Projects/ATS/brazil-mask-sam')
    from brazil_mask.utils.visualization import create_training_visualization, prepare_tensorboard_image
except ImportError:
    # If visualization utils are not available, disable bbox visualization
    create_training_visualization = None
    prepare_tensorboard_image = None


class CombinedImageVOSFrame(VOSFrame):
    """
    Custom VOSFrame that combines multiple image types into RGB channels
    """
    def __init__(self, frame_idx, image_paths, combine_types):
        """
        Initialize frame with multiple image paths
        
        Args:
            frame_idx: Frame index
            image_paths: List of 3 image paths [R_path, G_path, B_path]
            combine_types: List of 3 image types [R_type, G_type, B_type]
        """
        # Don't call super().__init__ to avoid setting single image_path
        self.frame_idx = frame_idx
        self.image_paths = image_paths
        self.combine_types = combine_types
        self.data = None  # Will be loaded when needed
        
    @property 
    def image_path(self):
        """
        Override image_path property to handle combined loading
        This property is accessed by the VOS dataset loading logic
        """
        return self._get_combined_image_path()
        
    def _get_combined_image_path(self):
        """
        Return a special identifier that the loading logic can recognize
        """
        return f"COMBINED:{':'.join(self.image_paths)}"
        
    def load_combined_image(self):
        """
        Load and combine the three images into RGB channels
        Returns a PIL Image with combined channels
        """
        if self.data is not None:
            return self.data
            
        try:
            # Load all three images as grayscale
            channels = []
            for img_path in self.image_paths:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")
                    
                # Load as grayscale and convert to array
                with g_pathmgr.open(img_path, "rb") as f:
                    img = Image.open(f).convert('L')  # Convert to grayscale
                    img_array = np.array(img)
                    channels.append(img_array)
            
            # Stack the three grayscale images as RGB channels
            combined_array = np.stack(channels, axis=2)  # Shape: (H, W, 3)
            
            # Convert back to PIL Image
            combined_image = Image.fromarray(combined_array.astype(np.uint8), mode='RGB')
            
            # Cache the result
            self.data = combined_image
            return combined_image
            
        except Exception as e:
            print(f"Error combining images {self.image_paths}: {e}")
            # Fallback: create a blank RGB image
            return Image.new('RGB', (512, 512), (0, 0, 0))


class BrazilMaskRawDataset(VOSRawDataset):
    """
    Dataset class for Brazil mask data with sequential naming:
    - images: 00001_imagetype.png
    - masks: 00001_imagetype_mask.png
    """
    
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        num_frames=1,
        image_type=None,
        combine_images=None,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.num_frames = num_frames
        self.image_type = image_type
        self.combine_images = combine_images
        
        # Load file mapping to understand sequential naming
        self.file_mapping = {}
        self.seq_to_case = {}
        mapping_path = Path(img_folder).parent / "file_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, "r") as f:
                self.file_mapping = json.load(f)
            # Create reverse mapping from sequential ID to case ID
            for seq_id, mapping_info in self.file_mapping.items():
                self.seq_to_case[seq_id] = mapping_info["case_id"]
        
        # Read the subset defined in file_list_txt (case IDs)
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                case_ids = [line.strip() for line in f if line.strip()]
        else:
            # Extract case IDs from sequential files
            all_images = glob.glob(os.path.join(img_folder, "*.png"))
            case_ids = set()
            for img_path in all_images:
                basename = os.path.basename(img_path)
                # Extract sequential ID from filename like "00001_imagetype.png"
                parts = basename.split('_')
                if len(parts) >= 2:
                    seq_id = parts[0]  # First part is sequential ID
                    if seq_id in self.seq_to_case:
                        case_ids.add(self.seq_to_case[seq_id])
            case_ids = sorted(list(case_ids))
        
        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [line.strip() for line in f if line.strip()]
        else:
            excluded_files = []
        
        # Filter out excluded cases
        self.case_ids = [case_id for case_id in case_ids if case_id not in excluded_files]
        
        print(f"BrazilMaskRawDataset: Found {len(self.case_ids)} cases")
        if len(self.case_ids) > 0:
            print(f"First few cases: {self.case_ids[:3]}")
    
    
    def get_video(self, idx):
        """
        Return a VOSVideo object for the given case
        """
        case_id = self.case_ids[idx]
        
        # Initialize seq_ids_for_case to avoid UnboundLocalError
        seq_ids_for_case = []
        
        # Handle different modes: single image type vs combined images
        if self.combine_images:
            # For combined images, we need to find seq_ids that have ALL required image types
            seq_ids_by_type = {img_type: [] for img_type in self.combine_images}
            
            for seq_id, mapping_info in self.file_mapping.items():
                if mapping_info["case_id"] == case_id:
                    img_type = mapping_info["image_type"]
                    if img_type in seq_ids_by_type:
                        seq_ids_by_type[img_type].append(seq_id)
                        seq_ids_for_case.append(seq_id)  # Track for segment loader
            
            # Check that we have all required image types
            missing_types = []
            for img_type in self.combine_images:
                if not seq_ids_by_type[img_type]:
                    missing_types.append(img_type)
            
            if missing_types:
                raise ValueError(f"Case {case_id} missing image types: {missing_types}")
            
            # Use the first available seq_id for each type (they should have same case_id)
            case_images = []
            for img_type in self.combine_images:  # Maintain order: R, G, B
                seq_id = seq_ids_by_type[img_type][0]  # Take first available
                mapping_info = self.file_mapping[seq_id]
                image_filename = f"{seq_id}_{mapping_info['image_type']}.png"
                image_path = os.path.join(self.img_folder, image_filename)
                if os.path.exists(image_path):
                    case_images.append(image_path)
        else:
            # Original single image type logic
            for seq_id, mapping_info in self.file_mapping.items():
                if mapping_info["case_id"] == case_id:
                    if self.image_type:
                        # Check if this sequential ID has the preferred image type
                        if mapping_info["image_type"] == self.image_type:
                            seq_ids_for_case.append(seq_id)
                    else:
                        seq_ids_for_case.append(seq_id)
            
            if not seq_ids_for_case:
                if self.image_type:
                    raise ValueError(f"No images found for case {case_id} with type {self.image_type}")
                else:
                    raise ValueError(f"No images found for case {case_id}")
            
            # Build image paths using sequential naming
            case_images = []
            for seq_id in seq_ids_for_case:
                mapping_info = self.file_mapping[seq_id]
                image_filename = f"{seq_id}_{mapping_info['image_type']}.png"
                image_path = os.path.join(self.img_folder, image_filename)
                if os.path.exists(image_path):
                    case_images.append(image_path)
        
        case_images = sorted(case_images)
        
        if not case_images:
            if self.image_type:
                raise ValueError(f"No images found for case {case_id} with type {self.image_type}")
            else:
                raise ValueError(f"No images found for case {case_id}")
        
        # Handle frame creation based on mode
        frames = []
        if self.combine_images:
            # For combined images, create a single frame that combines multiple image types
            if len(case_images) != 3:
                raise ValueError(f"Expected 3 images for combining, got {len(case_images)}")
            
            # Create a custom frame that will combine the images
            frame = CombinedImageVOSFrame(0, image_paths=case_images, combine_types=self.combine_images)
            frames.append(frame)
        else:
            # For single image training, use the first available image
            for frame_idx in range(min(self.num_frames, len(case_images))):
                image_path = case_images[frame_idx]
                frames.append(VOSFrame(frame_idx, image_path=image_path))
        
        # Create video object
        video = VOSVideo(case_id, idx, frames)
        
        # Create segment loader - point to mask directory for this case
        mask_dir = self.gt_folder  # Masks are in flat structure too
        segment_loader = BrazilMaskSegmentLoader(mask_dir, case_id, seq_ids_for_case, self.file_mapping)
        
        return video, segment_loader
    
    def __len__(self):
        return len(self.case_ids)
    


class BrazilMaskSegmentLoader:
    """
    Segment loader for Brazil mask PNG files with sequential naming
    """
    
    def __init__(self, mask_dir, case_id, seq_ids_for_case, file_mapping):
        self.mask_dir = mask_dir
        self.case_id = case_id
        self.seq_ids_for_case = seq_ids_for_case
        self.file_mapping = file_mapping
        
        # Find all mask files for this case using sequential naming
        self.mask_files = []
        for seq_id in seq_ids_for_case:
            mapping_info = file_mapping[seq_id]
            mask_filename = f"{seq_id}_{mapping_info['image_type']}_mask.png"
            mask_path = os.path.join(mask_dir, mask_filename)
            if os.path.exists(mask_path):
                self.mask_files.append(mask_path)
        
        self.mask_files = sorted(self.mask_files)
    
    def load(self, frame_idx, obj_ids=None):
        """
        Load mask for given frame - required by sampler
        Returns dict of {object_id: mask_tensor}
        """
        import torch
        from PIL import Image
        import numpy as np
        
        if not self.mask_files:
            # Return empty result if no masks
            return {}
        
        # For single frame training, use the first mask file
        mask_path = self.mask_files[0]
        
        try:
            mask_img = Image.open(mask_path).convert('L')  # Convert to grayscale
            mask_array = np.array(mask_img)
            
            # Convert to binary mask (assume non-zero pixels are mask)
            mask_binary = (mask_array > 0).astype(np.bool_)
            
            # Convert to tensor
            mask_tensor = torch.from_numpy(mask_binary)
            
            # Return as dict with single object
            return {1: mask_tensor}
            
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return {}
    
    
    def __call__(self, frame_idx, obj_ids):
        """
        Load mask for given frame and object IDs - required by VOSDataset
        """
        return self.load(frame_idx, obj_ids)