.PHONY: smoke prepare-cpt prepare-sft eval-ppl eval-chat test test-verbose

smoke:
	python scripts/smoke_test.py

prepare-cpt:
	python scripts/prepare_cpt_data.py --config configs/datasets.yaml --out data/manifests

prepare-sft:
	python scripts/prepare_sft_data.py --config configs/datasets.yaml --out data/manifests

eval-ppl:
	python scripts/eval_perplexity.py --base LiquidAI/LFM2.5-1.2B-Base --cpt outputs/lfm25-svenska-1.2b-cpt

eval-chat:
	python scripts/eval_chat.py --base LiquidAI/LFM2.5-1.2B-Base --cpt outputs/lfm25-svenska-1.2b-cpt

test:
	python -m pytest tests/ -x -q

test-verbose:
	python -m pytest tests/ -v
