import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database_downloader.downloader import HBNDownloader, DatabaseDownloaderConfig

# Could be used to loop through all releases !!!!!!
print("Downloading release 2 to mini format")
# Configure download location
config = DatabaseDownloaderConfig(data_root="./database")

# Create downloader instance
downloader = HBNDownloader(config, sampling_rate=100, mini=True)

# Download a release
dataset_path = downloader.download_release("2", exclude_derivatives=True)
print(f"Downloaded to: {dataset_path}")