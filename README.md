# uplift_b3_milestone_4

# IMDB Sentiment Analysis: VADER vs. NLTK + Naive Bayes vs. Claude Haiku 5.5

Three ways to classify the sentiment of movie reviews, compared on accuracy, macro-F1, performance on hard cases, speed, and cost:

| Tier | Method | Needs training data? | Cost |
|---|---|---|---|
| 1 | **VADER** - rule-based sentiment lexicon | No | Free |
| 2 | **NLTK + Naive Bayes** - bag-of-words classifier trained on labeled reviews | Yes | Free (CPU only) |
| 3 | **LLM zero-shot** - Claude Haiku 5.5 via the Anthropic API | No | Pay per token |

The full write-up (method, results, error analysis, conclusions) is in [`REPORT`](./report) *(add the link or file name of your exported report here)*.

## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/) (Kaggle) with 50,000 reviews, 25,000 labeled `positive` and 25,000 `negative`. There are no neutral labels. Models that can output `neutral` (VADER and the LLM) are counted as wrong when they do.

Download it and save it in the project root as `IMDB-Dataset.csv`. The file is not included in this repository.

## Repository structure

```
.
├── 01_classic_sentiment.ipynb        # VADER + NLTK/Naive Bayes, saves predictions and timings
├── 02_claude_sentiment.ipynb         # Claude Haiku 5.5 zero-shot classifier
├── 03_results_comparison.ipynb       # results table, charts, error analysis
├── IMDB-Dataset.csv                  # you provide this (see Dataset)
├── .env                              # your API key (never commit this)
├── requirements.txt
└── outputs (created when you run the notebooks)
    ├── test_predictions_classic.csv
    ├── classic_stats.json
    ├── IMDB_reviews_with_claude_sentiment.csv
    ├── claude_stats.json
    ├── results_table.csv
    └── fig1_metrics.png, fig2_confusion_matrices.png, fig3_speed.png
```

## Setup

**1. Python environment** (Python 3.10 or newer recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
textblob
vaderSentiment
wordcloud
anthropic
python-dotenv
jupyter
```

**2. NLTK data** (one time)

```python
import nltk
nltk.download('stopwords')
```

The first notebook also runs this download for you.

**3. Anthropic API key** (only needed for notebook 02)

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-key-here
```

Add `.env` to your `.gitignore` so the key is never committed:

```bash
echo ".env" >> .gitignore
```

## How to run

Run the notebooks **in order**. Each one reads files written by the one before it.

```bash
jupyter notebook
```

1. **`01_classic_sentiment.ipynb`** - Cleans the reviews, makes a stratified 80/20 train/test split (`random_state=42`), runs VADER, trains and evaluates the Naive Bayes model, and saves `test_predictions_classic.csv` and `classic_stats.json`. Takes a few minutes, mostly tokenizing 50,000 reviews.
2. **`02_claude_sentiment.ipynb`** - Draws 200 reviews from the **test split** (`random_state=1`), classifies each with Claude Haiku 5.5, and saves `IMDB_reviews_with_claude_sentiment.csv` and `claude_stats.json`. Makes 200 API calls one after another, so it takes several minutes and costs a small fraction of a dollar at Haiku 5.5's published rates.
3. **`03_results_comparison.ipynb`** - Builds the results table, three charts, and the error analysis from the saved files.

**Why 200 reviews for the comparison?** Calling an LLM on all 10,000 test reviews is slower and costs more. All three models are therefore compared on the same 200 held-out reviews so the comparison is fair. VADER and Naive Bayes are also scored on the full 10,000-review test set in notebook 01.

## Metrics

- **Accuracy** - share of reviews where the predicted label equals the human label.
- **Macro-F1** - F1 averaged over `positive` and `negative`. A `neutral` prediction counts as an error.
- **Hard-cases accuracy** - accuracy on the reviews where VADER and Naive Bayes disagree with each other.
- **Sec / 1k preds** - wall-clock seconds to produce 1,000 predictions on the machine that ran the notebooks. Naive Bayes includes tokenization.
- **Cost / 1k preds** - VADER is free, Naive Bayes runs on CPU only, and the LLM cost is (input tokens x input price + output tokens x output price), taken from the token counts returned by each API call.

## Reproducibility notes

- Data splits and samples use fixed random seeds, so the 200-review sample is the same on every run.
- LLM outputs and timings can vary slightly between runs and machines.
- Check Anthropic's [pricing page](https://docs.claude.com/en/docs/about-claude/pricing) and update `PRICE_IN_PER_MTOK` and `PRICE_OUT_PER_MTOK` in notebook 02 if the rates have changed.

## Troubleshooting

- **`FileNotFoundError: test_predictions_classic.csv`** - run notebook 01 to the end first.
- **`LookupError: Resource stopwords not found`** - run `nltk.download('stopwords')`.
- **Authentication error from Anthropic** - check that `.env` is in the folder you launch Jupyter from and contains `ANTHROPIC_API_KEY`.
- **Rate limit errors** - the Anthropic client retries automatically; if they persist, wait a minute and re-run the cell.
