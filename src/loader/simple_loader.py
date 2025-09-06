"""
Simplified HBN-EEG Data Loader

A minimal, efficient loader for HBN-EEG datasets focusing on core functionality.
Reduces complexity from ~1500 lines to ~250 lines while maintaining essential features.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import mne
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids


@dataclass
class SimpleConfig:
    """Minimal configuration for HBN-EEG dataset access."""
    data_root: Path = Path("database")
    dataset_name: str = "R1_L100"
    preload_data: bool = True
    
    @property
    def bids_root(self) -> Path:
        """Get the BIDS root directory path."""
        return self.data_root / self.dataset_name


# Task name mappings for common alternatives
TASK_MAPPINGS = {
    # RestingState
    "rest": "RestingState",
    "resting": "RestingState",
    "restingstate": "RestingState",
    # surroundSupp
    "surround": "surroundSupp",
    "surroundsuppression": "surroundSupp",
    "suppression": "surroundSupp",
    "surroundsupp": "surroundSupp",
    # contrastChangeDetection
    "contrast": "contrastChangeDetection",
    "contrastchange": "contrastChangeDetection",
    "changedetection": "contrastChangeDetection",
    "contrastchangedetection": "contrastChangeDetection",
    # seqLearning
    "sequence": "seqLearning8target",
    "sequencelearning": "seqLearning8target",
    "seq8": "seqLearning8target",
    "seqlearning8target": "seqLearning8target",
    "seq6": "seqLearning6target",
    "seqlearning6target": "seqLearning6target",
    # symbolSearch
    "symbol": "symbolSearch",
    "symbolsearch": "symbolSearch",
    "search": "symbolSearch",
    # DespicableMe
    "despicable": "DespicableMe",
    "despicableme": "DespicableMe",
    # DiaryOfAWimpyKid
    "diary": "DiaryOfAWimpyKid",
    "diaryofawimpykid": "DiaryOfAWimpyKid",
    "wimpykid": "DiaryOfAWimpyKid",
    # ThePresent
    "present": "ThePresent",
    "thepresent": "ThePresent",
    # FunwithFractals
    "fractals": "FunwithFractals",
    "funwithfractals": "FunwithFractals",
}


def normalize_task_name(task_name: str) -> str:
    """Normalize task name to official BIDS format."""
    task_lower = task_name.lower()
    if task_lower in ['movie', 'movies']:
          raise ValueError("Ambiguous task 'movie'. Please specify: DespicableMe, DiaryOfAWimpyKid, FunwithFractals, or ThePresent")
    return TASK_MAPPINGS.get(task_lower, task_name)


class SimpleHBNLoader:
    """
    loader for HBN-EEG BIDS dataset.

    Usage:
        config = SimpleConfig(data_root=Path("/path/to/data"))
        loader = SimpleHBNLoader(config)
        data = loader.get_data("NDARAC904DMU", "RestingState", run=1)
    """
    
    def __init__(self, config: Optional[SimpleConfig] = None):
        """Initialize the loader with configuration."""
        self.config = config or SimpleConfig()
        self.bids_root = self.config.bids_root
        
        # Set up minimal logging
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
        # Validate dataset exists
        if not self.bids_root.exists():
            raise FileNotFoundError(f"Dataset not found at {self.bids_root}")
        
        # Load participants data if available
        self._participants_df = None
        participants_path = self.bids_root / "participants.tsv"
        if participants_path.exists():
            try:
                self._participants_df = pd.read_csv(participants_path, sep="\t")
                self.logger.info(f"Loaded {len(self._participants_df)} participants")
            except Exception as e:
                self.logger.warning(f"Could not load participants file: {e}")
        
        # Configure MNE for minimal output
        mne.set_log_level("WARNING")
    
    def get_data(
        self,
        subject: str,
        task: str,
        run: Optional[int] = None,
        include_events: bool = True
    ) -> Dict[str, Any]:
        """
        Load EEG data for a specific subject and task.
        
        Args:
            subject: Subject ID (without 'sub-' prefix)
            task: Task name (accepts common alternatives)
            run: Run number for multi-run tasks
            include_events: Whether to load event data
            
        Returns:
            Dictionary containing:
                - 'raw': mne.io.Raw object with EEG data
                - 'events': pandas DataFrame of events (if include_events=True)
                - 'metadata': Basic subject info from participants.tsv
        """
        # Normalize task name
        task = normalize_task_name(task)
        
        # Load EEG data
        raw = self._load_raw_data(subject, task, run)
        
        # Build result dictionary
        result = {
            "raw": raw,
            "metadata": self.get_subject_info(subject)
        }
        
        # Load events if requested
        if include_events:
            result["events"] = self._load_events_data(subject, task, run)
        else:
            result["events"] = None
        
        return result
    
    def _load_raw_data(
        self,
        subject: str,
        task: str,
        run: Optional[int] = None
    ) -> mne.io.Raw:
        """Load raw EEG data with fallback strategies."""
        # Try BIDS loading first
        for ext in [".set", ".bdf"]:
            try:
                bids_path = BIDSPath(
                    root=self.bids_root,
                    subject=subject,
                    task=task,
                    run=run,
                    suffix="eeg",
                    extension=ext
                )
                raw = read_raw_bids(bids_path, verbose=False)
                if self.config.preload_data:
                    raw.load_data()
                return raw
            except Exception as e:
                self.logger.debug(f"BIDS loading failed for {ext}: {e}")
        
        # Fallback to direct file reading
        data_path = self._find_data_file(subject, task, run)
        
        if not data_path:
            raise FileNotFoundError(
                f"Cannot find EEG file for subject {subject}, task {task}"
                + (f", run {run}" if run else "")
            )
        
        # Load file based on extension
        if data_path.suffix == '.bdf':
            raw = mne.io.read_raw_bdf(str(data_path), preload=self.config.preload_data, verbose=False)
        else:
            raw = mne.io.read_raw_eeglab(str(data_path), preload=self.config.preload_data, verbose=False)
        return raw
    
    def _load_events_data(
        self,
        subject: str,
        task: str,
        run: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Load events from TSV file."""
        # Build events filename
        if run is not None:
            filename = f"sub-{subject}_task-{task}_run-{run}_events.tsv"
        else:
            filename = f"sub-{subject}_task-{task}_events.tsv"
        
        events_path = self.bids_root / f"sub-{subject}" / "eeg" / filename
        
        if events_path.exists():
            try:
                return pd.read_csv(events_path, sep="\t")
            except Exception as e:
                self.logger.warning(f"Could not load events: {e}")
        
        return None

    def _find_data_file(self, subject: str, task: str, run: Optional[int] = None) -> Optional[Path]:
        """Find the data file path without loading it."""
        # Try both .set and .bdf extensions
        for ext in ['.set', '.bdf']:
            if run is not None:
                filename = f"sub-{subject}_task-{task}_run-{run}_eeg{ext}"
            else:
                filename = f"sub-{subject}_task-{task}_eeg{ext}"
            
            data_path = self.bids_root / f"sub-{subject}" / "eeg" / filename
            
            if data_path.exists():
                return data_path
                
            # Try case variations
            for task_variant in [task, task.lower(), task.capitalize()]:
                if run is not None:
                    test_filename = f"sub-{subject}_task-{task_variant}_run-{run}_eeg{ext}"
                else:
                    test_filename = f"sub-{subject}_task-{task_variant}_eeg{ext}"
                test_path = self.bids_root / f"sub-{subject}" / "eeg" / test_filename
                if test_path.exists():
                    return test_path
                
        return None
    
    def data_exists(self, subject: str, task: str, run: Optional[int] = None) -> bool:
        """Check if data file exists without loading it."""
        return self._find_data_file(subject, task, run) is not None
    
    def get_available_subjects(self) -> List[str]:
        """Get list of available subject IDs."""
        subjects = []
        for item in self.bids_root.iterdir():
            if item.is_dir() and item.name.startswith("sub-"):
                subjects.append(item.name[4:])  # Remove 'sub-' prefix
        return sorted(subjects)
    
    def get_subject_tasks(self, subject: str) -> List[str]:
        """Get available tasks for a specific subject."""
        subject_dir = self.bids_root / f"sub-{subject}" / "eeg"
        if not subject_dir.exists():
            return []
        
        tasks = set()
        for pattern in ["*.set", "*.bdf"]:
            for eeg_file in subject_dir.glob(pattern):
                # Parse task from filename
                parts = eeg_file.stem.split("_")
                for part in parts:
                    if part.startswith("task-"):
                        tasks.add(part[5:])  # Remove 'task-' prefix
                        break
        
        return sorted(list(tasks))
    
    def get_subject_info(self, subject: str) -> Optional[Dict[str, Any]]:
        """Get basic subject information from participants file."""
        if self._participants_df is None:
            return None
        
        subject_id = f"sub-{subject}"
        matching = self._participants_df[
            self._participants_df["participant_id"] == subject_id
        ]
        
        if len(matching) == 0:
            return None
        
        # Return as simple dictionary
        row = matching.iloc[0]
        return {
            "subject_id": subject,
            "age": row.get("age"),
            "sex": row.get("sex"),
            "handedness": row.get("handedness")
        }
    
    # Backward compatibility method
    def load_subject_task(self, subject: str, task: str) -> mne.io.Raw:
        """Load EEG data (backward compatibility)."""
        result = self.get_data(subject, task, include_events=False)
        return result["raw"]