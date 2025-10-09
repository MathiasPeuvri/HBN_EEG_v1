"""
Clean evaluation data preparation wrapper

This module creates evaluation datasets by calling existing database_to_*_shards functions
with R2_mini_L100_bdf database parameters. It acts as a lightweight wrapper without
duplicating preprocessing logic.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import existing refactored functions
from src.database_to_dataset.database_to_pretraining_shards import create_pretraining_shards
from src.database_to_dataset.database_to_posttraining_shards import create_posttraining_shards
from src.database_to_dataset.database_to_challenge1_shards import create_challenge1_shards


def create_evaluation_datasets(subjects_limit: Optional[int] = None, verbose: bool = True) -> Dict[str, List[Path]]:
    """
    Create evaluation datasets by calling existing shard creation functions.
    
    Args:
        subjects_limit: Maximum number of subjects to process (None = all available)
        verbose: Whether to print progress information
        
    Returns:
        Dict[str, List[Path]]: Dictionary mapping dataset type to created file paths
    """
    # Configure paths for evaluation
    base_path = Path("/home/mts/HBN_EEG_v1")
    db_root = base_path / "database"
    output_dir = base_path / "datasets" / "evaluation_datasets"
    
    if verbose:
        print("=" * 60)
        print("Creating Evaluation Datasets")
        print("=" * 60)
        print(f"Database: {db_root / 'R2_mini_L100_bdf'}")
        print(f"Output: {output_dir}")
        if subjects_limit:
            print(f"Subject limit: {subjects_limit}")
        print()

    created_files = {}
    
    # Get available subjects from database
    from src.loader.simple_loader import SimpleConfig, SimpleHBNLoader
    config = SimpleConfig(data_root=db_root, dataset_name="R2_mini_L100_bdf")
    loader = SimpleHBNLoader(config)
    all_subjects = loader.get_available_subjects()
    
    # Limit subjects if specified
    if subjects_limit is not None:
        subjects_to_process = all_subjects[:subjects_limit]
    else:
        subjects_to_process = all_subjects
    
    if verbose:
        print(f"Processing {len(subjects_to_process)} subjects: {subjects_to_process[:3]}..." if len(subjects_to_process) > 3 else f"Processing subjects: {subjects_to_process}")

    # Create pretraining dataset (continuous epochs from surroundSupp)
    if verbose:
        print("Creating pretraining evaluation dataset...")
    create_pretraining_shards(
        dataset_name="R2_mini_L100_bdf",
        database_root=db_root,
        savepath_root=output_dir,
        subjects=subjects_to_process,
        nb_subjects_per_shard=1000,  # Large number to avoid multiple shards for eval
        task_name="surroundSupp",
        runs=[1, 2],
        epoch_length=2.0,
        overlap=0.0,
        verbose=verbose
    )
    
    # Create posttraining dataset (event-based epochs from contrastChangeDetection)  
    if verbose:
        print("\nCreating posttraining evaluation dataset...")
    create_posttraining_shards(
        dataset_name="R2_mini_L100_bdf",
        database_root=db_root,
        savepath_root=output_dir,
        subjects=subjects_to_process,
        nb_subjects_per_shard=1000,  # Large number to avoid multiple shards for eval
        task_name="contrastChangeDetection",
        runs=[1, 2, 3],
        tmin=-1.5,
        tmax=0.5,
        target_events=['right_target', 'right_buttonPress', 'left_target', 'left_buttonPress'],
        verbose=verbose
    )
    
    # Create challenge1 dataset (pre-trial epochs for response prediction)
    if verbose:
        print("\nCreating challenge1 evaluation dataset...")
    create_challenge1_shards(
        dataset_name="R2_mini_L100_bdf", 
        database_root=db_root,
        savepath_root=output_dir,
        subjects=subjects_to_process,
        nb_subjects_per_shard=1000,  # Large number to avoid multiple shards for eval
        task_name="contrastChangeDetection",
        runs=[1, 2, 3],
        tmin=-2.0,
        tmax=0.0,
        target_events=['right_target', 'left_target'],
        verbose=verbose
    )
    
    # Find created files
    created_files = find_created_datasets(output_dir)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Dataset Creation Complete")
        print("=" * 60)
        for dataset_type, files in created_files.items():
            print(f"{dataset_type}: {len(files)} files")
            for file_path in files:
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  - {file_path.name} ({size_mb:.2f} MB)")
                    
    return created_files


def find_created_datasets(output_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all created dataset files in the output directory.
    
    Args:
        output_dir: Directory to search for dataset files
        
    Returns:
        Dict[str, List[Path]]: Dictionary mapping dataset type to file paths
    """
    created_files = {
        'pretraining': [],
        'posttraining': [], 
        'challenge1': []
    }
    
    if not output_dir.exists():
        return created_files
        
    for file_path in output_dir.glob("*.pkl"):
        if "pretraining_data_shard" in file_path.name:
            created_files['pretraining'].append(file_path)
        elif "posttraining_data_shard" in file_path.name:
            created_files['posttraining'].append(file_path)
        elif "challenge1_data_shard" in file_path.name:
            created_files['challenge1'].append(file_path)
            
    return created_files


def verify_evaluation_datasets(output_dir: Optional[Path] = None, verbose: bool = True) -> bool:
    """
    Verify that evaluation datasets were created successfully.
    
    Args:
        output_dir: Directory containing evaluation datasets
        verbose: Whether to print verification results
        
    Returns:
        bool: True if datasets exist and are valid
    """
    if output_dir is None:
        output_dir = Path("/home/mts/HBN_EEG_v1/datasets/evaluation_datasets")
        
    created_files = find_created_datasets(output_dir)
    
    if verbose:
        print("\nVerifying evaluation datasets:")
        print("-" * 40)
        
    all_valid = True
    
    for dataset_type, files in created_files.items():
        if not files:
            if verbose:
                print(f"✗ {dataset_type}: No files found")
            all_valid = False
        else:
            if verbose:
                print(f"✓ {dataset_type}: {len(files)} files")
                for file_path in files:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"    {file_path.name} ({size_mb:.2f} MB)")
    
    return all_valid


def main():
    """CLI interface for evaluation dataset creation."""
    parser = argparse.ArgumentParser(
        description="Create evaluation datasets using R2_mini_L100_bdf database",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--subjects-limit", 
        type=int,
        help="Maximum number of subjects to process (default: all available)"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true", 
        help="Only verify existing datasets without creating new ones"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if args.verify_only:
        success = verify_evaluation_datasets(verbose=verbose)
        sys.exit(0 if success else 1)
    else:
        try:
            create_evaluation_datasets(
                subjects_limit=args.subjects_limit,
                verbose=verbose
            )
            if verbose:
                verify_evaluation_datasets(verbose=True)
        except Exception as e:
            print(f"Error creating evaluation datasets: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()