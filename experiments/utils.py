"""
Shared utilities for Speech Emotion Recognition experiments.
Contains dataset loading, training functions, and helper utilities.
"""

import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import types

try:
    from transformers.models.wavlm.modeling_wavlm import WavLMAttention
except ImportError:
    WavLMAttention = None


def format_device(device: torch.device) -> str:
    """Return a human-readable description of the torch device."""
    if device.type == "cuda":
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        device_name = torch.cuda.get_device_name(device_index)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            visible_devices = [d.strip() for d in visible.split(",") if d.strip()]
            if device_index < len(visible_devices):
                phys_id = visible_devices[device_index]
                return f"cuda:{device_index} (visible:{phys_id}, {device_name})"
        return f"cuda:{device_index} ({device_name})"
    return str(device)


# ============================================================================
# EMOTION MAPPING
# ============================================================================

EMOTION_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMODB_EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
RAVDESS_8CLASS_EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "calm"]


# ============================================================================
# RAVDESS DATASET LOADING
# ============================================================================


def load_ravdess(dataset_path: str) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load RAVDESS dataset. File format: 03-01-06-01-02-01-12.wav"""
    dataset_path = Path(dataset_path)

    emotion_code = {
        "01": 4, "02": 4, "03": 3, "04": 5, "05": 0, "06": 2, "07": 1, "08": 6,
    }

    audio_files = []
    labels = []

    actor_folders = list(dataset_path.glob("Actor_*"))
    if not actor_folders:
        actor_folders = list(dataset_path.glob("audio_speech_actors_*/Actor_*"))

    if not actor_folders:
        raise FileNotFoundError(f"No Actor_* folders found in {dataset_path}")

    for actor_folder in sorted(actor_folders):
        if not actor_folder.is_dir():
            continue
        for audio_file in sorted(actor_folder.glob("03-*.wav")):
            filename = audio_file.name
            parts = filename.split("-")
            if len(parts) < 3:
                continue
            emotion = parts[2]
            if emotion in emotion_code:
                audio_files.append(str(audio_file))
                labels.append(emotion_code[emotion])

    if not audio_files:
        raise ValueError(f"No valid audio files found in {dataset_path}")

    metadata = {
        "name": "RAVDESS",
        "num_samples": len(audio_files),
        "num_classes": 7,
        "language": "English",
        "speakers": 24,
        "label_names": EMOTION_NAMES,
    }
    return audio_files, labels, metadata


def load_ravdess_8class(dataset_path: str) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load RAVDESS with calm as separate class (8 classes)."""
    dataset_path = Path(dataset_path)

    emotion_code = {
        "01": 4, "02": 7, "03": 3, "04": 5, "05": 0, "06": 2, "07": 1, "08": 6,
    }

    audio_files = []
    labels = []

    actor_folders = list(dataset_path.glob("Actor_*"))
    if not actor_folders:
        actor_folders = list(dataset_path.glob("audio_speech_actors_*/Actor_*"))

    if not actor_folders:
        raise FileNotFoundError(f"No Actor_* folders found in {dataset_path}")

    for actor_folder in sorted(actor_folders):
        if not actor_folder.is_dir():
            continue
        for audio_file in sorted(actor_folder.glob("03-*.wav")):
            filename = audio_file.name
            parts = filename.split("-")
            if len(parts) < 3:
                continue
            emotion = parts[2]
            if emotion in emotion_code:
                audio_files.append(str(audio_file))
                labels.append(emotion_code[emotion])

    if not audio_files:
        raise ValueError(f"No valid audio files found in {dataset_path}")

    metadata = {
        "name": "RAVDESS (8-class)",
        "num_samples": len(audio_files),
        "num_classes": 8,
        "language": "English",
        "speakers": 24,
        "label_names": RAVDESS_8CLASS_EMOTION_NAMES,
    }
    return audio_files, labels, metadata


def load_ravdess_no_calm(dataset_path: str) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load RAVDESS excluding calm (emotion code 02)."""
    dataset_path = Path(dataset_path)

    emotion_code = {
        "01": 4, "03": 3, "04": 5, "05": 0, "06": 2, "07": 1, "08": 6,
    }

    audio_files = []
    labels = []

    actor_folders = list(dataset_path.glob("Actor_*"))
    if not actor_folders:
        actor_folders = list(dataset_path.glob("audio_speech_actors_*/Actor_*"))

    if not actor_folders:
        raise FileNotFoundError(f"No Actor_* folders found in {dataset_path}")

    for actor_folder in sorted(actor_folders):
        if not actor_folder.is_dir():
            continue
        for audio_file in sorted(actor_folder.glob("03-*.wav")):
            filename = audio_file.name
            parts = filename.split("-")
            if len(parts) < 3:
                continue
            emotion = parts[2]
            if emotion in emotion_code:
                audio_files.append(str(audio_file))
                labels.append(emotion_code[emotion])

    if not audio_files:
        raise ValueError(f"No valid audio files found in {dataset_path}")

    metadata = {
        "name": "RAVDESS (no calm)",
        "num_samples": len(audio_files),
        "num_classes": 7,
        "language": "English",
        "speakers": 24,
        "label_names": EMOTION_NAMES,
    }
    return audio_files, labels, metadata


def load_savee(dataset_path: str) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load SAVEE dataset. File format: DC_a01.wav"""
    dataset_path = Path(dataset_path)

    emotion_code = {
        "a": 0, "d": 1, "f": 2, "h": 3, "n": 4, "sa": 5, "su": 6,
    }

    audio_files = []
    labels = []

    wav_files = list(dataset_path.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {dataset_path}")

    for audio_file in sorted(wav_files):
        filename = audio_file.stem
        if "_" not in filename:
            continue
        parts = filename.split("_")
        if len(parts) != 2:
            continue
        speaker, emotion_part = parts
        emotion = "".join([c for c in emotion_part if not c.isdigit()])
        if emotion in emotion_code:
            audio_files.append(str(audio_file))
            labels.append(emotion_code[emotion])

    if not audio_files:
        raise ValueError(f"No valid audio files found in {dataset_path}")

    metadata = {
        "name": "SAVEE",
        "num_samples": len(audio_files),
        "num_classes": 7,
        "language": "English",
        "speakers": 4,
        "label_names": EMOTION_NAMES,
    }
    return audio_files, labels, metadata


def load_emodb(dataset_path: str) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load EmoDB dataset. File format: 03a01Fa.wav"""
    dataset_path = Path(dataset_path)

    emotion_code = {
        "W": 0, "E": 1, "A": 2, "F": 3, "N": 4, "T": 5, "L": 4,
    }

    audio_files = []
    labels = []

    wav_files = list(dataset_path.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {dataset_path}")

    for audio_file in sorted(wav_files):
        filename = audio_file.stem
        if len(filename) < 7:
            continue
        try:
            emotion = filename[5]
        except IndexError:
            continue
        if emotion in emotion_code:
            audio_files.append(str(audio_file))
            labels.append(emotion_code[emotion])

    if not audio_files:
        raise ValueError(f"No valid audio files found in {dataset_path}")

    metadata = {
        "name": "EmoDB",
        "num_samples": len(audio_files),
        "num_classes": 6,
        "language": "German",
        "speakers": 10,
        "label_names": EMODB_EMOTION_NAMES,
    }
    return audio_files, labels, metadata


def load_tess(dataset_path: str) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load TESS dataset. Directory: OAF_angry/*.wav"""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"TESS dataset path not found: {dataset_path}")

    emotion_map = {
        "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
        "neutral": 4, "sad": 5, "surprise": 6, "pleasant_surprise": 6,
    }

    audio_files: List[str] = []
    labels: List[int] = []
    speakers: set[str] = set()

    for emotion_dir in sorted(dataset_path.iterdir()):
        if not emotion_dir.is_dir():
            continue
        folder_name = emotion_dir.name
        if "_" in folder_name:
            speaker, raw_emotion = folder_name.split("_", 1)
        else:
            speaker = folder_name
            raw_emotion = folder_name

        speakers.add(speaker)
        normalized = raw_emotion.lower().replace(" ", "_").replace("-", "_")
        if normalized == "pleasant_surprised":
            normalized = "pleasant_surprise"
        if normalized not in emotion_map:
            continue
        label = emotion_map[normalized]

        wav_files = sorted(emotion_dir.glob("*.wav"))
        for wav in wav_files:
            audio_files.append(str(wav))
            labels.append(label)

    if not audio_files:
        raise ValueError(f"No valid audio files found in {dataset_path}")

    metadata = {
        "name": "TESS",
        "num_samples": len(audio_files),
        "num_classes": 7,
        "language": "English",
        "speakers": len(speakers),
        "label_names": EMOTION_NAMES,
    }
    return audio_files, labels, metadata


def load_crema_d(dataset_path: str) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load CREMA-D dataset. File format: 1001_DFA_ANG_XX.wav"""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"CREMA-D dataset path not found: {dataset_path}")

    audio_root = dataset_path / "AudioWAV"
    if not audio_root.exists():
        audio_root = dataset_path

    emotion_code = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
    label_names = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

    audio_files: List[str] = []
    labels: List[int] = []
    speakers: set[str] = set()

    wav_files = sorted(audio_root.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_root}")

    for audio_file in wav_files:
        parts = audio_file.stem.split("_")
        if len(parts) < 3:
            continue
        speaker_id = parts[0]
        emotion = parts[2]
        if emotion not in emotion_code:
            continue
        speakers.add(speaker_id)
        audio_files.append(str(audio_file))
        labels.append(emotion_code[emotion])

    if not audio_files:
        raise ValueError(f"No valid CREMA-D audio files found in {dataset_path}")

    metadata = {
        "name": "CREMA-D",
        "num_samples": len(audio_files),
        "num_classes": len(label_names),
        "language": "English",
        "speakers": len(speakers),
        "label_names": label_names,
    }
    return audio_files, labels, metadata


def load_emovo(dataset_path: str) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load EMOVO dataset. File format: neu-m1-b1.wav"""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"EMOVO dataset path not found: {dataset_path}")

    emotion_code = {"rab": 0, "dis": 1, "pau": 2, "gio": 3, "neu": 4, "tri": 5, "sor": 6}

    audio_files: List[str] = []
    labels: List[int] = []
    speakers: set[str] = set()

    wav_files = sorted(dataset_path.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {dataset_path}")

    for audio_file in wav_files:
        parts = audio_file.stem.split("-")
        if len(parts) != 3:
            continue
        emotion, speaker_id, _ = parts
        if emotion not in emotion_code:
            continue
        speakers.add(speaker_id)
        audio_files.append(str(audio_file))
        labels.append(emotion_code[emotion])

    if not audio_files:
        raise ValueError(f"No valid EMOVO audio files found in {dataset_path}")

    metadata = {
        "name": "EMOVO",
        "num_samples": len(audio_files),
        "num_classes": 7,
        "language": "Italian",
        "speakers": len(speakers),
        "label_names": EMOTION_NAMES,
    }
    return audio_files, labels, metadata


# ============================================================================
# EMOTION DATASET CLASS
# ============================================================================


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion recognition."""

    def __init__(
        self,
        audio_files: Optional[List[str]],
        labels: Optional[List[int]],
        sample_rate: int = 16000,
        max_length: float = 5.0,
        cache_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        self.cache_entries = cache_entries
        self.use_cache = cache_entries is not None
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_length)

        if self.use_cache:
            self.audio_files = None
            if cache_entries is None:
                raise ValueError("cache_entries cannot be None when use_cache=True")
            self.labels = [entry["label"] for entry in cache_entries]
        else:
            if audio_files is None or labels is None:
                raise ValueError("audio_files and labels are required when cache is not used")
            self.audio_files = audio_files
            self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.use_cache:
            entry = self.cache_entries[idx]
            cache_path = entry["path"]
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            waveform = cached["waveform"]
            attention_mask = cached["attention_mask"]
            label = int(cached.get("label", entry["label"]))
            return waveform, attention_mask, label

        audio_file = self.audio_files[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(audio_file)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze(0)
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        current_samples = waveform.shape[0]

        if current_samples > self.max_samples:
            waveform = waveform[: self.max_samples]
            attention_mask = torch.ones(self.max_samples, dtype=torch.float32)
        else:
            padding = self.max_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding), value=0.0)
            attention_mask = torch.cat(
                [
                    torch.ones(current_samples, dtype=torch.float32),
                    torch.zeros(padding, dtype=torch.float32),
                ]
            )

        return waveform, attention_mask, label


def _load_cache_entries(cache_index_path: str) -> List[Dict[str, Any]]:
    """Load cache metadata entries from index file."""
    index_path = Path(cache_index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Cache index not found: {cache_index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    base_dir = index_path.parent
    for entry in data.get("entries", []):
        entry_path = base_dir / entry["path"]
        entries.append({
            "id": entry.get("id"),
            "label": entry["label"],
            "path": str(entry_path),
            "original_path": entry.get("original_path"),
        })

    if not entries:
        raise ValueError(f"No entries found in cache index {cache_index_path}")
    return entries


def create_dataloaders(
    audio_files: List[str],
    labels: List[int],
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int = 42,
    use_cache: bool = False,
    cache_index_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    indices = list(range(len(audio_files)))
    train_indices, val_indices, train_labels, val_labels = train_test_split(
        indices, labels, test_size=val_split, random_state=seed, stratify=labels
    )

    cache_entries = None
    if use_cache:
        if not cache_index_path:
            raise ValueError("cache_index_path must be provided when use_cache=True")
        cache_entries = _load_cache_entries(cache_index_path)
        if len(cache_entries) != len(audio_files):
            raise ValueError(
                f"Cache entries ({len(cache_entries)}) do not match dataset size ({len(audio_files)})"
            )

    def build_dataset(idxs: List[int], subset_labels: List[int]) -> EmotionDataset:
        if use_cache and cache_entries is not None:
            subset_entries = [cache_entries[i] for i in idxs]
            return EmotionDataset(audio_files=None, labels=subset_labels, cache_entries=subset_entries)
        subset_audio = [audio_files[i] for i in idxs]
        return EmotionDataset(subset_audio, subset_labels)

    train_dataset = build_dataset(train_indices, train_labels)
    val_dataset = build_dataset(val_indices, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


# ============================================================================
# WAVLM PATCHES (Fix PyTorch 2.5 compatibility)
# ============================================================================


def apply_pretrained_patches(model):
    """Patch local for WavLM compatibility with PyTorch 2.5+."""
    _apply_local_wavlm_patch(model)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for waveforms, attention_masks, labels in pbar:
        waveforms = waveforms.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(waveforms, attention_mask=attention_masks)
        loss = criterion(logits, labels)

        if torch.isnan(loss):
            raise RuntimeError("NaN loss encountered during training")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def validate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Dict[str, Any]:
    """Validate model. Returns dict with metrics and raw predictions."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_classes = None

    with torch.no_grad():
        for waveforms, attention_masks, labels in tqdm(dataloader, desc="Validation", leave=False):
            waveforms = waveforms.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            logits = model(waveforms, attention_mask=attention_masks)
            num_classes = logits.shape[1]
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    _, ua, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    if num_classes is None:
        raise RuntimeError("Validation received no batches")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return {
        "loss": avg_loss, "accuracy": accuracy, "ua": ua,
        "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": cm, "predictions": all_preds, "labels": all_labels,
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_incremented_output_dir(base_output_dir: str) -> str:
    """Return the next numbered sibling directory for a dataset output."""
    base_path = Path(base_output_dir)
    parent = base_path.parent
    dataset_name = base_path.name

    max_index = 0
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        prefix, sep, suffix = child.name.partition("-")
        if sep != "-" or suffix != dataset_name:
            continue
        if prefix.isdigit():
            max_index = max(max_index, int(prefix))

    return str(parent / f"{max_index + 1}-{dataset_name}")


def save_checkpoint(model: nn.Module, filepath: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Checkpoint saved: {filepath}")


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    results_serializable = {}
    for key, value in results.items():
        if key == "confusion_matrix":
            results_serializable[key] = value.tolist()
        elif hasattr(value, "item"):
            results_serializable[key] = value.item()
        elif isinstance(value, (list, tuple)):
            results_serializable[key] = [
                v.item() if hasattr(v, "item") else v for v in value
            ]
        else:
            results_serializable[key] = value

    with open(filepath, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved: {filepath}")


def save_history(history: List[Dict[str, float]], filepath: str):
    """Save training history to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(filepath, index=False)
    print(f"History saved: {filepath}")


def plot_training_curves(history: List[Dict[str, float]], filepath: str):
    """Plot training curves (loss and accuracy)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"], label="Train Accuracy", marker="o")
    axes[1].plot(df["epoch"], df["val_acc"], label="Val Accuracy", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {filepath}")


def plot_confusion_matrix(cm, class_names: List[str], filepath: str):
    """Plot confusion matrix."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {filepath}")


def save_confusion_matrix_reports(cm, class_names: List[str], output_dir: str):
    """Save confusion matrix in CSV and plain-text formats."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    txt_path = os.path.join(output_dir, "confusion_matrix.txt")

    df.to_csv(csv_path, index=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix\n")
        f.write(df.to_string())
        f.write("\n")

    print(f"Confusion matrix CSV saved: {csv_path}")
    print(f"Confusion matrix text saved: {txt_path}")


def save_per_class_metrics(cm, class_names: List[str], output_dir: str):
    """Compute and save per-class precision, recall, F1 from confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)

    n = len(class_names)
    cm = cm[:n, :n]

    per_class = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(cm[i, :].sum())
        per_class.append([name, precision, recall, f1, support])

    precision_macro = sum(p[1] for p in per_class) / n
    recall_macro = sum(p[2] for p in per_class) / n
    f1_macro = sum(p[3] for p in per_class) / n
    total_support = sum(p[4] for p in per_class)

    total_tp = sum(cm[i, i] for i in range(n))
    total_preds = cm.sum()
    accuracy = total_tp / total_preds if total_preds > 0 else 0.0
    precision_weighted = sum(p[1] * p[4] for p in per_class) / total_support if total_support > 0 else 0.0
    recall_weighted = sum(p[2] * p[4] for p in per_class) / total_support if total_support > 0 else 0.0
    f1_weighted = sum(p[3] * p[4] for p in per_class) / total_support if total_support > 0 else 0.0

    csv_path = os.path.join(output_dir, "per_class_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1_score,support\n")
        for row in per_class:
            f.write(f"{row[0]},{row[1]:.4f},{row[2]:.4f},{row[3]:.4f},{row[4]}\n")
        f.write(f"macro_avg,{precision_macro:.4f},{recall_macro:.4f},{f1_macro:.4f},{total_support}\n")
        f.write(f"weighted_avg,{precision_weighted:.4f},{recall_weighted:.4f},{f1_weighted:.4f},{total_support}\n")
    print(f"Per-class metrics saved: {csv_path}")

    txt_path = os.path.join(output_dir, "per_class_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        from sklearn.metrics import classification_report
        all_labels = []
        all_preds = []
        for i in range(n):
            for j in range(n):
                count = int(cm[i, j])
                all_labels.extend([i] * count)
                all_preds.extend([j] * count)
        report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        f.write(report)
    print(f"Per-class report saved: {txt_path}")


def _patched_wavlm_self_attention(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    gated_position_bias: torch.Tensor,
    output_attentions: bool,
):
    key_padding_mask = None
    if attention_mask is not None:
        pad_positions = attention_mask.ne(1)
        key_padding_mask = torch.zeros_like(
            attention_mask, dtype=hidden_states.dtype, device=attention_mask.device
        )
        if pad_positions.any():
            key_padding_mask = key_padding_mask.masked_fill(pad_positions, float("-inf"))

    query = key = value = hidden_states.transpose(0, 1)
    bias_k = bias_v = None
    add_zero_attn = False

    attn_output, attn_weights = F.multi_head_attention_forward(
        query, key, value,
        self.embed_dim, self.num_heads,
        torch.empty([0], device=hidden_states.device),
        torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
        bias_k, bias_v, add_zero_attn, self.dropout,
        self.out_proj.weight, self.out_proj.bias,
        self.training, key_padding_mask,
        output_attentions, gated_position_bias,
        use_separate_proj_weight=True,
        q_proj_weight=self.q_proj.weight,
        k_proj_weight=self.k_proj.weight,
        v_proj_weight=self.v_proj.weight,
    )

    attn_output = attn_output.transpose(0, 1)
    if attn_weights is not None:
        attn_weights = attn_weights[:, None].broadcast_to(
            attn_weights.shape[:1] + (self.num_heads,) + attn_weights.shape[1:]
        )
    return attn_output, attn_weights


def _apply_local_wavlm_patch(model: torch.nn.Module) -> None:
    if WavLMAttention is None:
        return
    for module in model.modules():
        if isinstance(module, WavLMAttention) and not getattr(module, "_ser_patched", False):
            module.torch_multi_head_self_attention = types.MethodType(
                _patched_wavlm_self_attention, module
            )
            module._ser_patched = True
