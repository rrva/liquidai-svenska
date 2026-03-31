.PHONY: smoke prepare-cpt prepare-sft eval-ppl eval-chat

smoke:
	python scripts/smoke_test.py

prepare-cpt:
	python scripts/prepare_cpt_data.py --config configs/datasets.yaml --out data/manifests

prepare-sft:
	python scripts/prepare_sft_data.py --config configs/datasets.yaml --out data/manifests

eval-ppl:
	python scripts/eval_perplexity.py

eval-chat:
	python scripts/eval_chat.py
