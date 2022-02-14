from dataclasses import dataclass
from omegaconf import MISSING
from typing import Any

class LMConfig:
    lm_path: str = ''  # Path to an (optional) kenlm language model for use with beam search (req\'d with trie)
    top_paths: int = 1  # Number of beams to return
    alpha: float = 0.0  # Language model weight
    beta: float = 0.0  # Language model word bonus (all words)
    cutoff_top_n: int = 40  # Cutoff_top_n characters with highest probs in vocabulary will be used in beam search
    cutoff_prob: float = 1.0  # Cutoff probability in pruning,default 1.0, no pruning.
    beam_width: int = 10  # Beam width to use
    lm_workers: int = 4  # Number of LM processes to use


@dataclass
class ModelConfig:
    precision: int = 32  # Set to 16 to use mixed-precision for inference
    cuda: bool = True
    checkpoint_path: str = '' # c

@dataclass
class DataConfig:
    audio_path: str = '' # Audio file to predict on
    size_window: int = 20480
    seq: Any = MISSING
    phone_labels: Any = MISSING
    nb_speakers: int = 1
    n_process_loader: int = 6
    file_extension: str = '.wav'
    max_size_loaded: int = 4000000000


@dataclass
class InferenceConfig:
    # lm: LMConfig = LMConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()


@dataclass
class TranscribeConfig(InferenceConfig):
    # audio_path: str = ''  # Audio file to predict on
    offsets: bool = False  # Returns time offset information
    destination: str  = '' # output saved
    layer: str = '' # layer wanted
    nGPU: int = -1 # Number of used GPUs
    csv_file: str = '' # csv file with list to transform (not mandatory)
