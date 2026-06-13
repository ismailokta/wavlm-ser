# Speech Emotion Recognition — WavLM + LSTM
# Two models: mean pooling (baseline) and additive attention

.PHONY: help train-mean train-attention train-all clean

PRESET ?= ravdess

help:
	@echo "Speech Emotion Recognition — WavLM + LSTM"
	@echo ""
	@echo "  make train-mean PRESET=<dataset>       - Train LSTM + Mean Pooling"
	@echo "  make train-attention PRESET=<dataset>   - Train LSTM + Additive Attention"
	@echo "  make train-all                          - Train both models on all 6 datasets"
	@echo "  make clean                              - Remove all results"
	@echo ""
	@echo "Datasets (PRESET): ravdess | ravdess_no_calm | ravdess_8class | emodb | savee | tess | crema_d | emovo"
	@echo ""
	@echo "Examples:"
	@echo "  make train-mean PRESET=ravdess"
	@echo "  make train-attention PRESET=emodb"
	@echo "  make train-all"

train-mean:
	@echo "=== LSTM + Mean Pooling — $(PRESET) ==="
	python experiments/train_lstm_mean.py --preset $(PRESET)

train-attention:
	@echo "=== LSTM + Attention — $(PRESET) ==="
	python experiments/train_lstm_attention.py --preset $(PRESET)

train-all:
	@echo "=== Training all datasets — Mean Pooling ==="
	@for ds in ravdess emodb savee tess crema_d emovo; do \
		$(MAKE) train-mean PRESET=$$ds; \
	done
	@echo "=== Training all datasets — Additive Attention ==="
	@for ds in ravdess emodb savee tess crema_d emovo; do \
		$(MAKE) train-attention PRESET=$$ds; \
	done

clean:
	@echo "Cleaning all results..."
	rm -rf results/mean/* results/attention/*
	@echo "Done."
