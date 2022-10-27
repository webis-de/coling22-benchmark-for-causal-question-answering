import re
import string
from rouge_score import rouge_scorer, scoring
from collections import Counter

from itertools import chain
from typing import List, Dict, Callable, Tuple, Union


def preprocess(str_: str) -> str:
    str_ = str_.lower()
    str_ = str_.translate(str.maketrans('', '', string.punctuation))
    str_ = re.sub(r'\b(a|an|the)\b', ' ', str_)
    str_ = ' '.join(str_.split())
    return str_


def _f1(pred: str, ground_truth: str) -> float:
    prediction_tokens = pred.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _em(pred: str, ground_truth: str) -> int:
    return int(pred == ground_truth)


def _calculate_measures(measure: Callable[[str, str], Union[List[float], float]],
                        predictions: List[str],
                        ground_truths: List[str]) -> Tuple[float, List[float]]:
    result = 0
    samples = []
    for pred, ground_truth in zip(predictions, ground_truths):
        value = max(measure(pred, answer) for answer in ground_truth)
        result += value
        samples.append(value)
    return result/len(predictions), samples


def _rouge_l(predictions: List[str], ground_truths: List[str]) \
        -> Tuple[Dict[str, float], List[float], List[float], List[float]]:
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    aggregator = scoring.BootstrapAggregator()
    samples_precision = []
    samples_recall = []
    samples_f1 = []
    for pred, gts in zip(predictions, ground_truths):
        score = scorer.score_multi(gts, pred)
        aggregator.add_scores(score)
        samples_precision.append(score['rougeL'].precision)
        samples_recall.append(score['rougeL'].recall)
        samples_f1.append(score['rougeL'].fmeasure)
    results = aggregator.aggregate()
    return {'rougeL_precision': results['rougeL'].mid.precision,
            'rougeL_recall': results['rougeL'].mid.recall,
            'rougeL_f1': results['rougeL'].mid.fmeasure}, \
        samples_precision, samples_recall, samples_f1


def all_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, Union[List[float], float]]:
    predictions = [preprocess(pred) for pred in predictions]
    ground_truths = [[preprocess(gt) for gt in gts] for gts in ground_truths]

    rougel, sample_rougel_precision, sample_rougel_recall, sample_rougel_f1 = _rouge_l(predictions, ground_truths)
    f1, sample_f1 = _calculate_measures(_f1, predictions, ground_truths)
    em, sample_em = _calculate_measures(_em, predictions, ground_truths)

    f1_em_ = {'f1': f1, 'em': em}
    samples = {'samples_f1': sample_f1, 'samples_exact_match': sample_em,
               'samples_rougeL_precision': sample_rougel_precision, 'samples_rougeL_recall': sample_rougel_recall,
               'samples_rougeL_f1': sample_rougel_f1}

    return dict(chain.from_iterable(d.items() for d in (f1_em_, rougel, samples)))
