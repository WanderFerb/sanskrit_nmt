# Sanskrit → English Neural Machine Translation

Custom Transformer sequence-to-sequence model for Sanskrit→English translation
(Natural Language Understanding — Assignment 2). The model is built from scratch in
PyTorch: multi-head attention, encoder/decoder stacks, masking, and beam search are all
implemented directly — no pre-trained translation weights.

## Repository contents

| File | Purpose |
|------|---------|
| `train.ipynb` | Trains the model on the provided data and writes all artifacts. |
| `evaluate.ipynb` | **Run this at evaluation time.** Reloads the saved model and scores any test file (including the private set), writing `submission.csv`. |
| `artifacts/best_model.pt` | Trained model weights. |
| `artifacts/spm_sa.model`, `artifacts/spm_en.model` | SentencePiece subword tokenizers (Sanskrit / English). |
| `artifacts/config.json` | Model configuration so `evaluate.ipynb` can rebuild the exact architecture. |
| `artifacts/training_curves.png` | Loss and dev-BLEU curves. |
| `submission.csv` | Predicted English translations (`Source_id, Sentence_en`). |
| `NMT_Report.pdf` | Report. |

## Data

Uses **only** the six provided files: `train_sa_10000.csv`, `train_en_10000.csv`,
`dev_sa_1000.csv`, `dev_en_1000.csv`, `test_sa_1000.csv`, `test_en_1000.csv`
(columns `Source_id` and `Sentence_sa` / `Sentence_en`).

## How to run

**Train:** open `train.ipynb`, set `DATA_DIR` to the folder holding the six CSVs, and run all
cells. It saves everything to `artifacts/` and writes `submission.csv`.

**Evaluate (used for grading):** open `evaluate.ipynb`, set `TEST_SA_PATH` (and `TEST_EN_PATH`
if references are given) to the released test file, and run all cells. It reloads the artifacts,
translates, writes `submission.csv`, and prints BLEU, F1 BERTScore, inference time, and parameter
count.

Every dependency installs in the first cell of each notebook. A fixed seed (42) makes training
reproducible.

## Model summary

Transformer encoder–decoder: `d_model=256`, 8 heads, 4+4 layers, `d_ff=1024`, dropout 0.1,
pre-norm residuals, tied target-embedding/output weights, sinusoidal positions. Trained with
label smoothing (0.1) and an Adam + Noam warmup schedule; decoded with beam search
(length-normalized). Subword vocabularies are learned with SentencePiece (unigram).

## Evaluation metrics

- **BLEU** — NLTK corpus BLEU, default weights.
- **BERTScore** — F1, `rescale_with_baseline=True`, `lang="en"`.
- **Efficiency** — total test-set inference time and total parameter count.

## Pre-trained model disclosure

The **translation model uses no pre-trained weights** — it is trained end to end on the provided
dataset only. The **only** pre-trained model involved is loaded **internally by the BERTScore
library** (a RoBERTa-large model) to compute the BERTScore metric during evaluation; it plays no
part in producing translations. No external APIs are used.
