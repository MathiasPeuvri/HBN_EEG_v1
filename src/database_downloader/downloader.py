"""
HBN-EEG Dataset Downloader

This module provides functionality to download HBN-EEG datasets from AWS S3.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Union
import logging
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class DatabaseDownloaderConfig(BaseModel):
    """
    Configuration for HBN-EEG dataset downloads.
    See docs/downloader_api.md for usage examples.
    """
    data_root: Path = Field(default=Path("./database"))
    dataset_name: str = Field(default="R1_L20")
    s3_bucket: str = Field(default="s3://nmdatasets/NeurIPS25/")
    tasks: List[str] = Field(default=[
        "rest", "surround", "movie", 
        "contrast", "sequence", "symbol"])
    sampling_rate: int = Field(default=500)
    n_channels: int = Field(default=128)
    
    class Config:
        frozen = True


class HBNDownloader:
    """
    Downloads HBN-EEG datasets from AWS S3.
    Args:
        sampling_rate: Sampling rate in Hz (100 or 500). Default: 100
        mini: Whether to download mini releases (R{}_mini_L100_bdf format). Default: False
    """
    
    def __init__(self, config: DatabaseDownloaderConfig, sampling_rate: int = 100, mini: bool = False):
        self.config = config
        self.sampling_rate = sampling_rate
        self.mini = mini
        
        # Configure bucket and prefix based on sampling rate
        if sampling_rate == 100:
            # 100Hz resampled data (filtered 0.5-50Hz)
            self.bucket_name = "nmdatasets"
            self.base_prefix = "NeurIPS25"
        elif sampling_rate == 500:
            # 500Hz raw data
            self.bucket_name = "fcp-indi"
            self.base_prefix = "data/Projects/HBN/BIDS_EEG"
        else:
            raise ValueError(f"Unsupported sampling rate: {sampling_rate}. Supported: 100Hz, 500Hz")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def list_available_releases(self) -> List[str]:
        """
        Returns list of available release names.(e.g., ['R1_L100', 'R2_L100', ...])
        """
        if self.sampling_rate == 100:
            # Known 100Hz releases
            return ['R1_L100', 'R2_L100', 'R3_L100', 'R4_L100', 'R5_L100', 
                    'R6_L100', 'R7_L100', 'R8_L100', 'R9_L100', 'R10_L100', 'R11_L100']
        else:
            # Known 500Hz releases (simplified names)
            return ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']


    def download_using_aws_cli(
        self, 
        release: str, 
        target_dir: Optional[Path] = None,
        exclude_derivatives: bool = False
    ) -> Path:
        """
        Download a release at configured sampling rate.
        
        Args:
            release: Release name (e.g., 'R1_L100' for 100Hz or 'cmi_bids_R1' for 500Hz)
            target_dir: Target directory (defaults to config.data_root/release)
            exclude_derivatives: Whether to exclude derivatives folder (saves storage)
            
        Returns:
            Path to downloaded dataset
            
        Raises:
            RuntimeError: If download fails or AWS CLI not available
        """
        # Check if AWS CLI is available
        if not shutil.which("aws"):
            raise RuntimeError(
                "AWS CLI not found. Install with: 'pip install awscli' or system package manager"
            )
        
        # Determine target directory - use simplified name
        if target_dir is None:
            # For 500Hz data, use simplified directory name (R1 instead of cmi_bids_R1)
            dir_name = release.replace('cmi_bids_', '') if release.startswith('cmi_bids_') else release
            target_dir = self.config.data_root / dir_name
        
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct AWS CLI command
        s3_url = f"s3://{self.bucket_name}/{self.base_prefix}/{release}/"
        cmd = [
            "aws", "s3", "cp",
            s3_url,
            str(target_dir),
            "--recursive",
            "--no-sign-request",
            "--region", "us-east-1"#, "--progress"
        ]
        
        # Add exclude option for derivatives if requested
        if exclude_derivatives:
            cmd.extend(["--exclude", "derivatives/*"])
            self.logger.info("Excluding derivatives folder to save storage space")
        
        try:
            # Run download with progress
            self.logger.info(f"Starting download: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=1007200  # 2 hours timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully downloaded {release} to {target_dir}")
                return target_dir
            else:
                raise RuntimeError(f"Download failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Download timeout after 1 hour for {release}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"AWS CLI download failed: {e.stderr}")


    def download_release(
        self, 
        release: str = "1", 
        exclude_derivatives: bool = True
    ) -> Path:
        """
        Download a release at configured sampling rate.
        """
        # Format release name based on sampling rate
        if self.sampling_rate == 100:
            if not release.endswith('_L100'):
                if self.mini:
                    formatted_release = f"R{release}_mini_L100_bdf"
                else:
                    formatted_release = f"R{release}_L100"
            else:
                formatted_release = release
        else:  # 500Hz
            if release.startswith('cmi_bids_'):
                formatted_release = release
            elif release.startswith('R'):
                formatted_release = f"cmi_bids_{release}"
            else:
                formatted_release = f"cmi_bids_R{release}"
        
        sampling_note = f"{self.sampling_rate}Hz {'resampled' if self.sampling_rate == 100 else 'raw'}"
        storage_note = f" (excluding derivatives - saves ~50% storage)" if exclude_derivatives else ""
        self.logger.info(f"Starting download of {sampling_note} release: {formatted_release}{storage_note}")
        
        return self.download_using_aws_cli(formatted_release, exclude_derivatives=exclude_derivatives)


    def verify_download(self, dataset_path: Path) -> Dict[str, bool]:
        """
        Verify the integrity of a downloaded dataset.
        
        Args:
            dataset_path: Path to the downloaded dataset
            
        Returns:
            Dictionary with verification results
        """
        results = {
            'path_exists': False,
            'has_participants_file': False,
            'has_eeg_files': False,
            'has_events_files': False,
            'bids_structure': False
        }
        
        try:
            # Check if path exists
            results['path_exists'] = dataset_path.exists() and dataset_path.is_dir()
            
            if not results['path_exists']:
                return results
            
            # Check for participants.tsv
            participants_file = dataset_path / "participants.tsv"
            results['has_participants_file'] = participants_file.exists()
            
            # Check for EEG files
            eeg_files = list(dataset_path.rglob("*.set"))
            results['has_eeg_files'] = len(eeg_files) > 0
            
            # Check for events files
            event_files = list(dataset_path.rglob("*events.tsv"))
            results['has_events_files'] = len(event_files) > 0
            
            # Check BIDS structure (sub-* directories with eeg subdirectories)
            subject_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('sub-')]
            bids_structure = False
            if subject_dirs:
                # Check if at least one subject has eeg directory
                for sub_dir in subject_dirs[:5]:  # Check first 5 subjects
                    eeg_dir = sub_dir / "eeg"
                    if eeg_dir.exists():
                        bids_structure = True
                        break
            
            results['bids_structure'] = bids_structure
            
            # Log results
            self.logger.info(f"Verification results for {dataset_path}:")
            for check, passed in results.items():
                status = "✓" if passed else "✗"
                self.logger.info(f"  {status} {check}")
                
            return results
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            return results