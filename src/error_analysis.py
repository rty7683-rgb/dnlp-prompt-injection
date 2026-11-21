"""Export reviewed Model B errors without fitting or changing the model."""

import json

import numpy as np
import pandas as pd

from src.data import RESULTS_DIR, ROOT, load_config, load_verified_splits
from src.features import NgramCounts
from src.models.naive_bayes import MultinomialNaiveBayes
from src.preprocessing import normalize_text
from src.tokenizer import tokenize


REVIEWS = {
    "test": {
        54: ("unseen/OOV wording", "This is an ordinary German question about coal policy. Four of its nine tokens are unseen, leaving the classifier with limited lexical evidence."),
        95: ("unusual phrasing; unseen/OOV wording", "A benign opening question is followed by an instruction to change speaking style. The apparent typo in the instruction is unseen and may obscure a familiar lexical cue."),
        101: ("unusual phrasing; keyword dependence", "The same misspelled style instruction follows an added benign search phrase. Unigram counts pool the search terms and the instruction without distinguishing their roles."),
        106: ("multilingual expression", "A German vaccine question precedes an English request for embeddings. Unigram counts do not represent the language switch or the relationship between the two clauses."),
        111: ("long-context prompt; unusual phrasing", "The same misspelled style instruction appears after two benign segments. The extra context can supply competing evidence while the misspelled cue is still unseen."),
        114: ("long-context prompt; multilingual expression", "Two benign English topics precede a German override at the end. A unigram model cannot represent the location of that instruction or its relationship to the preceding text."),
        115: ("indirect injection; multilingual expression", "An instruction to disregard context is framed as a hypothetical question after English and German factual questions. The indirect framing and mixed context may make individual words less diagnostic."),
    },
    "validation": {
        379: ("negation/semantic ambiguity; keyword dependence", "The user asks about question length, which is a benign meta-question. Its familiar prompt-related wording can be ambiguous without the full question meaning."),
        510: ("unusual phrasing", "A German search phrase is followed by a terse instruction to ignore preceding material. Every token is known, so missing vocabulary does not explain this failure."),
        545: ("long-context prompt; multilingual expression", "Benign English and German requests precede a Spanish override. All tokens are known, but a unigram model cannot represent the switch or the order of the requests."),
    },
}


def main() -> None:
    splits = load_verified_splits(include_test=True)
    config = load_config()
    final = ROOT / "artifacts" / "final"
    model = MultinomialNaiveBayes.load(final / "model_b.npz")
    extractor = NgramCounts(tuple(config["model_b_ngram_range"]))
    extractor.vocabulary = json.loads((final / "features_b.json").read_text(encoding="utf-8"))
    saved = pd.read_csv(RESULTS_DIR / "test_predictions.csv").set_index("source_index")
    rows = []
    for split, reviewed in REVIEWS.items():
        frame = splits[split].set_index("source_index")
        docs = [tokenize(normalize_text(text, lowercase=config["lowercase"])) for text in frame["text"]]
        matrix, _ = extractor.transform(docs)
        gold = frame["label"].to_numpy()
        if split == "test":
            saved = saved.loc[frame.index]
            if not np.array_equal(saved["gold_label"], gold):
                raise RuntimeError("Test prediction labels do not match verified rows")
            scores = saved[["log_score_0", "log_score_1"]].to_numpy()
            predicted = saved["predicted_label"].to_numpy()
        else:
            scores = model.class_log_scores(matrix)
            predicted = model.classes_[scores.argmax(axis=1)]

        errors = gold != predicted
        error_indices = set(frame.index[errors])
        if not set(reviewed) <= error_indices or (split == "test" and set(reviewed) != error_indices):
            raise RuntimeError(f"{split}: manual review indices do not match prediction errors")

        totals = np.array([len(doc) for doc in docs])
        oov = totals - matrix.sum(axis=1)
        for name, mask in (("errors", errors), ("correct", ~errors)):
            print(split, name, int(mask.sum()), int(oov[mask].sum()), int(totals[mask].sum()))

        for index, (category, explanation) in reviewed.items():
            i = frame.index.get_loc(index)
            rows.append({
                "split": split,
                "source_index": index,
                "original_text": frame.at[index, "text"],
                "gold_label": gold[i],
                "predicted_label": predicted[i],
                "log_score_0": scores[i, 0],
                "log_score_1": scores[i, 1],
                "margin": abs(scores[i, 0] - scores[i, 1]),
                "error_type": "FP" if predicted[i] == 1 else "FN",
                "token_count": totals[i],
                "oov_token_count": oov[i],
                "oov_rate": oov[i] / totals[i] if totals[i] else 0.0,
                "possible_failure_category": category,
                "analysis": explanation,
            })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "error_analysis.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
