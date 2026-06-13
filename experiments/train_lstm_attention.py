"""
Attention Pooling: LSTM unidirectional + additive attention.
WavLM-Large (frozen) -> LSTM -> Additive Attention -> Classifier.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from transformers import AutoModel

EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.append(str(EXPERIMENTS_DIR))

from utils import (
    load_ravdess,
    load_ravdess_no_calm,
    load_ravdess_8class,
    load_savee,
    load_emodb,
    load_tess,
    load_crema_d,
    load_emovo,
    create_dataloaders,
    apply_pretrained_patches,
    train_one_epoch,
    validate,
    save_checkpoint,
    save_results,
    save_history,
    plot_training_curves,
    plot_confusion_matrix,
    save_confusion_matrix_reports,
    save_per_class_metrics,
    EMOTION_NAMES,
    format_device,
    get_incremented_output_dir,
)


DATASET_LOADERS = {
    "ravdess": load_ravdess,
    "ravdess_no_calm": load_ravdess_no_calm,
    "ravdess_8class": load_ravdess_8class,
    "emodb": load_emodb,
    "savee": load_savee,
    "tess": load_tess,
    "crema_d": load_crema_d,
    "emovo": load_emovo,
}


def _load_presets_from_json() -> dict:
    preset_path = Path(__file__).parent / "presets.json"
    with preset_path.open("r") as f:
        data = json.load(f)
    presets = {}
    for preset_name, variants in data["presets"].items():
        if "attention" in variants:
            presets[preset_name] = variants["attention"]
    return presets


PRESETS = _load_presets_from_json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSTM + Additive Attention")
    parser.add_argument(
        "--preset", default="ravdess",
        help="ravdess | ravdess_no_calm | ravdess_8class | emodb | savee | tess | crema_d | emovo",
    )
    return parser.parse_args()


def build_feature_mask(attention_mask: torch.Tensor, feature_seq_len: int) -> torch.Tensor:
    valid_lengths = attention_mask.sum(dim=1)
    waveform_len = attention_mask.shape[1]
    scale = feature_seq_len / waveform_len
    feature_valid_lengths = (valid_lengths * scale).long().clamp(min=1, max=feature_seq_len)
    positions = torch.arange(feature_seq_len, device=attention_mask.device).unsqueeze(0)
    return positions < feature_valid_lengths.unsqueeze(1)


class LSTMAttentionModel(nn.Module):
    """WavLM-Large (frozen) -> LSTM unidirectional -> Additive Attention -> Classifier."""

    def __init__(
        self, pretrained_name: str, feature_dim: int, num_classes: int,
        hidden_dim: int, num_layers: int, dropout: float,
        attention_hidden_dim: int, attention_score_dropout: float,
        post_aggregation_dropout: float, cache_dir: str,
    ):
        super().__init__()
        self.pretrained = AutoModel.from_pretrained(
            pretrained_name, cache_dir=cache_dir, use_safetensors=True,
        )
        apply_pretrained_patches(self.pretrained)
        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=feature_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False,
        )

        self.W_score = nn.Linear(hidden_dim, attention_hidden_dim)
        self.v_score = nn.Linear(attention_hidden_dim, 1, bias=False)
        self.attn_dropout = nn.Dropout(attention_score_dropout)
        self.post_aggregation_dropout = nn.Dropout(post_aggregation_dropout)
        self.post_aggregation_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        self.pretrained.eval()
        with torch.no_grad():
            features = self.pretrained(x, attention_mask=attention_mask).last_hidden_state

        H_lstm, _ = self.lstm(features)

        U_score = torch.tanh(self.W_score(H_lstm))
        U_score = self.attn_dropout(U_score)
        e_score = self.v_score(U_score).squeeze(-1)

        if attention_mask is not None:
            feature_mask = build_feature_mask(attention_mask, H_lstm.shape[1])
            e_score = e_score.masked_fill(~feature_mask, float("-inf"))

        alpha_weight = F.softmax(e_score, dim=1)
        c_context = torch.bmm(alpha_weight.unsqueeze(1), H_lstm).squeeze(1)
        c_context = self.post_aggregation_dropout(c_context)
        c_context = self.post_aggregation_norm(c_context)
        logits = self.classifier(c_context)
        return logits


def main() -> None:
    args = parse_args()
    config = PRESETS[args.preset].copy()
    config["output_dir"] = get_incremented_output_dir(config["output_dir"])
    dataset_loader = DATASET_LOADERS[config["dataset_key"]]

    print("=" * 80)
    print("PELATIHAN LSTM + ATTENTION")
    print("=" * 80)
    print(f"Preset: {args.preset}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Output dir: {config['output_dir']}")
    print("WavLM-Large: frozen")
    print("Arsitektur: LSTM unidirectional + additive attention")
    print("=" * 80)

    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {format_device(device)}")

    print("\n[1/6] Memuat dataset...")
    audio_files, labels, metadata = dataset_loader(config["dataset_path"])
    dataset_label = metadata.get("name", config["dataset_name"])
    class_names = metadata.get("label_names", EMOTION_NAMES)
    print(f"Total sampel: {metadata['num_samples']}")
    print(f"Jumlah kelas: {metadata['num_classes']}")
    print(f"Jumlah speaker: {metadata['speakers']}")

    print("\n[2/6] Menyiapkan dataloader...")
    use_cache = config.get("use_waveform_cache", False)
    cache_index_path = config.get("cache_index_path")
    train_loader, val_loader = create_dataloaders(
        audio_files=audio_files, labels=labels,
        batch_size=config["batch_size"], val_split=config["val_split"],
        num_workers=config["num_workers"], seed=config["seed"],
        use_cache=use_cache, cache_index_path=cache_index_path,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    print("\n[3/6] Membuat model...")
    model = LSTMAttentionModel(
        pretrained_name=config["pretrained_name"],
        feature_dim=config["feature_dim"], num_classes=config["num_classes"],
        hidden_dim=config["hidden_dim"], num_layers=config["num_layers"],
        dropout=config["dropout"],
        attention_hidden_dim=config["attention_hidden_dim"],
        attention_score_dropout=config["attention_score_dropout"],
        post_aggregation_dropout=config["post_aggregation_dropout"],
        cache_dir=config["hf_cache"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameter: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n[4/6] Menyiapkan pelatihan...")
    class_weights = torch.tensor(config["class_weights"], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["label_smoothing"], weight=class_weights,
    )
    optimizer = AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=config["warmup_epochs"])
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=config["scheduler_tmax"] - config["warmup_epochs"],
        eta_min=config["min_lr"],
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config["warmup_epochs"]],
    )

    print("\n[5/6] Training berjalan...")
    best_val_acc = -1.0
    best_metrics = None
    patience_counter = 0
    history = []

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 80)

        train_loss, train_acc = train_one_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer,
            criterion=criterion, device=device,
        )
        val_metrics = validate(
            model=model, dataloader=val_loader, criterion=criterion, device=device,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val UA: {val_metrics['ua']:.4f} | Val Precision: {val_metrics['precision']:.4f}"
              f" | Val Recall: {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"Learning rate: {current_lr:.6f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
            "val_ua": val_metrics["ua"], "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"], "val_f1": val_metrics["f1"],
            "lr": current_lr,
        })

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = val_metrics.copy()
            patience_counter = 0
            checkpoint_path = os.path.join(config["output_dir"], "best_model.pth")
            save_checkpoint(model, checkpoint_path)
            print(f"Model terbaik diperbarui (Val Accuracy: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"Tidak ada peningkatan (patience {patience_counter}/{config['early_stop_patience']})")

        if patience_counter >= config["early_stop_patience"]:
            print("\nPenghentian dini aktif")
            break

    print("\n[6/6] Menyimpan hasil...")
    best_epoch = epoch - patience_counter
    final_results = {
        "model": "LSTM + Attention",
        "dataset": dataset_label,
        "encoder": "WavLM-Large (frozen)",
        "total_epochs": epoch, "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc, "best_val_ua": best_metrics["ua"],
        "best_val_precision": best_metrics["precision"],
        "best_val_recall": best_metrics["recall"],
        "best_val_f1": best_metrics["f1"],
        "trainable_params": trainable_params,
        "config": config,
    }
    save_results(final_results, os.path.join(config["output_dir"], "results.json"))
    save_history(history, os.path.join(config["output_dir"], "history.csv"))
    plot_training_curves(history, os.path.join(config["output_dir"], "training_curves.png"))
    plot_confusion_matrix(
        best_metrics["confusion_matrix"], class_names,
        os.path.join(config["output_dir"], "confusion_matrix.png"),
    )
    save_confusion_matrix_reports(
        best_metrics["confusion_matrix"], class_names, config["output_dir"],
    )
    save_per_class_metrics(
        best_metrics["confusion_matrix"], class_names, config["output_dir"],
    )

    print("\n" + "=" * 80)
    print("PELATIHAN SELESAI")
    print("=" * 80)
    print(f"Model: LSTM unidirectional + additive attention")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Best Val UA: {best_metrics['ua']:.4f}")
    print(f"Best Val Precision: {best_metrics['precision']:.4f}")
    print(f"Best Val Recall: {best_metrics['recall']:.4f}")
    print(f"Best Val F1: {best_metrics['f1']:.4f}")
    print(f"Trainable Params: {trainable_params:,} (~{trainable_params / 1e6:.2f}M)")
    print(f"Output: {config['output_dir']}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
