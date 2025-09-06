"""
Configuration module for HBN-EEG dataset loader.

This module defines configuration classes and data models used throughout
the enhanced EEG loader implementation.

Overkill for the simple loader, many legacy code, but works !
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

# Load environment variables
load_dotenv()


class DatabaseLoaderConfig(BaseModel):
    """
    Configuration for HBN-EEG dataset access.
    """
    data_root: Path = Field(
        default=Path("database"),
        description="Root directory containing dataset samples", )
    dataset_name: str = Field(
        default="R1_L100", description="Name of the specific dataset subdirectory")
    cache_dir: Optional[Path] = Field(
        default=None, description="Directory for caching processed data")
    preload_data: bool = Field(
        default=True, description="Whether to preload EEG data into memory by default")
    hed_schema_version: str = Field(
        default="8.3.0", description="HED schema version for tag validation")
    log_level: str = Field(
        default="ERROR", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    max_memory_gb: float = Field(
        default=8.0, description="Maximum memory usage in GB for data loading")

    @validator("data_root")
    def validate_data_root(cls, v):
        """Validate that data root exists."""
        if not isinstance(v, Path):
            v = Path(v)
        # Make path absolute if relative
        if not v.is_absolute():
            v = Path.cwd() / v
        return v

    @validator("cache_dir")
    def validate_cache_dir(cls, v):
        """Create cache directory if it doesn't exist."""
        if v is not None:
            if not isinstance(v, Path):
                v = Path(v)
            if not v.is_absolute():
                v = Path.cwd() / v
            # Create directory if it doesn't exist
            v.mkdir(parents=True, exist_ok=True)
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @property
    def bids_root(self) -> Path:
        """Get the BIDS root directory path."""
        return self.data_root / self.dataset_name

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        use_enum_values = True


class Subject(BaseModel):
    """
    Subject information and clinical information / metadata

    Args:
        subject_id: Unique subject identifier (without 'sub-' prefix)
        age: Subject's age in years
        sex: Subject's biological sex ('M' or 'F')
        handedness: Handedness score (-1 to 1, where -1 is left-handed)
        p_factor: General psychopathology factor score
        attention: Attention problems score
        internalizing: Internalizing problems score
        externalizing: Externalizing problems score
    """

    subject_id: str = Field(description="Subject identifier without 'sub-' prefix")
    age: Optional[float] = Field(default=None, description="Age in years", ge=0, le=100)
    sex: Optional[str] = Field(default=None, description="Biological sex (M/F)")
    handedness: Optional[float] = Field(
        default=None, description="Handedness score (-1 to 1)", ge=-1.0, le=1.0
    )
    p_factor: Optional[float] = Field(
        default=None, description="General psychopathology factor score"
    )
    attention: Optional[float] = Field(
        default=None, description="Attention problems score"
    )
    internalizing: Optional[float] = Field(
        default=None, description="Internalizing problems score"
    )
    externalizing: Optional[float] = Field(
        default=None, description="Externalizing problems score"
    )

    @validator("sex")
    def validate_sex(cls, v):
        """Validate sex values."""
        if v is not None and v.upper() not in ["M", "F", "MALE", "FEMALE"]:
            raise ValueError("Sex must be 'M', 'F', 'Male', or 'Female'")
        return v.upper() if v else v

    @validator("subject_id")
    def validate_subject_id(cls, v):
        """Ensure subject ID doesn't have 'sub-' prefix."""
        if v.startswith("sub-"):
            return v[4:]  # Remove 'sub-' prefix
        return v

    @classmethod
    def from_participants_row(cls, row):
        """Create a Subject instance from a participants DataFrame row."""
        try:
            import pandas as pd
            # Handle DataFrame vs Series
            if isinstance(row, pd.DataFrame) and len(row) > 0:
                # DataFrame with multiple rows - take first row as Series
                row_data = row.iloc[0]
            elif isinstance(row, pd.Series):
                # Already a Series - use directly
                row_data = row
            else:
                # Other object type
                row_data = row
            
            # Extract participant ID
            if hasattr(row_data, 'get'):
                participant_id = row_data.get('participant_id', '')
            elif hasattr(row_data, '__getitem__'):
                participant_id = row_data['participant_id'] if 'participant_id' in row_data else ''
            else:
                participant_id = str(row_data) if row_data else ''
            
            # Remove 'sub-' prefix to get subject ID
            if isinstance(participant_id, str) and participant_id.startswith('sub-'):
                subject_id = participant_id[4:]
            else:
                subject_id = str(participant_id) if participant_id else ''
            
            # Extract other fields safely
            def safe_get(field_name):
                try:
                    if hasattr(row_data, 'get'):
                        return row_data.get(field_name)
                    elif hasattr(row_data, '__getitem__') and field_name in row_data:
                        return row_data[field_name]
                    else:
                        return None
                except:
                    return None
            
            return cls(
                subject_id=subject_id,
                age=safe_get('age'),
                sex=safe_get('sex'),
                handedness=safe_get('handedness'),
                p_factor=safe_get('p_factor'),
                attention=safe_get('attention'),
                internalizing=safe_get('internalizing'),
                externalizing=safe_get('externalizing')
            )
            
        except Exception as e:
            # Fallback: create a minimal Subject object
            return cls(subject_id='', age=None, sex=None, handedness=None,
                      p_factor=None, attention=None, internalizing=None, externalizing=None)

    def __str__(self) -> str:
        """String representation of subject."""
        return f"Subject({self.subject_id}, age={self.age}, sex={self.sex})"


class TaskConfig(BaseModel):
    """
    Configuration for specific EEG tasks.
    Args:
        task_name: Official task name as used in BIDS filenames
        common_names: Alternative names users might use for this task
        description: Brief description of the experimental task
    """

    task_name: str = Field(description="Official BIDS task name")
    common_names: List[str] = Field(
        default_factory=list, description="Alternative names for this task")
    description: Optional[str] = Field(default=None, description="Task description")
    expected_duration: Optional[float] = Field(
        default=None, description="Expected duration in seconds")
    event_codes: Dict[str, Union[int, str]] = Field(
        default_factory=dict, description="Mapping of event names to codes")

    def matches_name(self, name: str) -> bool:
        """Check if a given name matches this task."""
        name_lower = name.lower()
        return name_lower == self.task_name.lower() or name_lower in [
            cn.lower() for cn in self.common_names
        ]


# Default task configurations for HBN-EEG dataset
DEFAULT_TASKS = [
    TaskConfig(
        task_name="RestingState",
        common_names=["rest", "resting", "restingstate"],
        description="Eyes open/closed resting state recording",
        expected_duration=400.0,
        event_codes={
            "resting_start": 90,
            "instructed_toOpenEyes": 20,
            "instructed_toCloseEyes": 30,
        },
    ),
    TaskConfig(
        task_name="DespicableMe",
        common_names=["movie", "movies", "despicable"],
        description="Movie watching task - Despicable Me",
        expected_duration=600.0,
        event_codes={},
    ),
    TaskConfig(
        task_name="surroundSupp",
        common_names=["surround", "surroundsuppression", "suppression"],
        description="Surround suppression visual task",
        expected_duration=300.0,
        event_codes={},
    ),
    TaskConfig(
        task_name="contrastChangeDetection",
        common_names=["contrast", "contrastchange", "changedetection"],
        description="Contrast change detection task",
        expected_duration=250.0,
        event_codes={},
    ),
    TaskConfig(
        task_name="seqLearning8target",
        common_names=["sequence", "sequencelearning", "seq8"],
        description="Sequence learning task with 8 targets",
        expected_duration=400.0,
        event_codes={"seqLearning_start": 91, "learningBlock_1": 31},
    ),
    TaskConfig(
        task_name="seqLearning6target",
        common_names=["seq6"],
        description="Sequence learning task with 6 targets",
        expected_duration=400.0,
        event_codes={},
    ),
    TaskConfig(
        task_name="symbolSearch",
        common_names=["symbol", "symbolsearch", "search"],
        description="Symbol search cognitive task",
        expected_duration=300.0,
        event_codes={},
    ),
]


def create_default_config() -> DatabaseLoaderConfig:
    """
    Create a default configuration for the HBN-EEG dataset.
    """
    return DatabaseLoaderConfig(
        data_root=Path("database"),
        dataset_name="R1_L100",
        preload_data=True,
        hed_schema_version="8.3.0",
        log_level="INFO",
    )


def get_task_config(task_name: str) -> Optional[TaskConfig]:
    """
    Get configuration for a specific task.

    Args:
        task_name: Name of the task to find configuration for

    Returns:
        TaskConfig object if found, None otherwise
    """
    for task_config in DEFAULT_TASKS:
        if task_config.matches_name(task_name):
            return task_config
    return None


def normalize_task_name(task_name: str) -> str:
    """
    Normalize a task name to the official BIDS format.

    Args:
        task_name: Input task name (may be alternative name)

    Returns:
        Official BIDS task name, or original name if not found
    """
    task_config = get_task_config(task_name)
    return task_config.task_name if task_config else task_name
