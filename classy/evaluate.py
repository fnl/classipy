"""
.. py:module:: classy.generate
   :synopsis: Evaluate a classifier against an inverted index.

.. moduleauthor:: Florian Leitner <florian.leitner@gmail.com>
.. License: GNU Affero GPL v3 (http://www.gnu.org/licenses/agpl.html)
"""
from collections import Counter

import logging
from classy.data import load_index
from classy.learn import make_pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals import joblib
from sklearn import metrics

L = logging.getLogger(__name__)


def evaluate_model(args):
    L.debug("%s", args)

    if not args.model:
        cross_evaluation(args)
    else:
        simple_evaluation(args)


def simple_evaluation(args):
    data = load_index(args.index)

    if data.labels is None or len(data.labels) == 0:
        raise RuntimeError("input data has no labels to learn from")

    pipeline = joblib.load(args.model)
    predictions = pipeline.predict(data.index)
    evaluate(data.labels, predictions, data.label_names, data.min_label)

    if args.pr_curve:
        plot_pr_curve(pipeline, data)


def cross_evaluation(args):
    pipeline, parameters, data = make_pipeline(args)
    cross_val = StratifiedKFold(data.labels, n_folds=args.folds, shuffle=True)
    results = []

    for step, (train, test) in enumerate(cross_val):
        pipeline.fit(data.index[train], data.labels[train])
        targets = data.labels[test]
        predictions = pipeline.predict(data.index[test])
        print("\nCV Round", step + 1)
        L.debug("%s predictions/targets", len(predictions))
        c = Counter(targets)
        c = {data.label_names[k]: v for k, v in c.items()}
        L.debug("target counts: %s", c)
        c = Counter(predictions)
        c = {data.label_names[k]: v for k, v in c.items()}
        L.debug("prediction counts: %s", c)
        results.append(evaluate(
            targets, predictions, data.label_names, data.min_label
        ))

    print("\nCV Summary")
    print("Average Accuracy :", sum(i[0] for i in results) / len(results))
    print("Average Precision:", sum(i[1] for i in results) / len(results))
    print("Average Recall   :", sum(i[2] for i in results) / len(results))
    print("Average F-Score  :", sum(i[3] for i in results) / len(results))
    print("Average MCC-Score:", sum(i[4] for i in results) / len(results))


def evaluate(labels, predictions, label_names, min_label):
    if len(label_names) > 2:
        labels = labels.ravel()
        predictions = predictions.ravel()
        precision = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)
        mcc = metrics.matthews_corrcoef(labels, predictions)
        f_score = metrics.f1_score(labels, predictions)
        accuracy = metrics.accuracy_score(labels, predictions)
    else:
        precision = metrics.precision_score(labels, predictions,
                                            pos_label=min_label)
        recall = metrics.recall_score(labels, predictions,
                                      pos_label=min_label)
        f_score = metrics.f1_score(labels, predictions, pos_label=min_label)
        mcc = metrics.matthews_corrcoef(labels, predictions)
        accuracy = metrics.accuracy_score(labels, predictions)

    print("Accuracy :", "{: 0.3f}".format(accuracy))
    print("Precision:", "{: 0.3f}".format(precision))
    print("Recall   :", "{: 0.3f}".format(recall))
    print("F-Score  :", "{: 0.3f}".format(f_score))
    print("MCC-Score:", "{: 0.3f}".format(mcc))
    return accuracy, precision, recall, f_score, mcc


def plot_pr_curve(pipeline, data):
    import matplotlib.pyplot as plt
    from matplotlib.backends import backend
    L.debug("matplotlib backend: %s", backend)

    if backend == "agg":
        L.warn("matplotlib agg backend will not produce visible results")

    scores = pipeline.decision_function(data.index)
    n_classes = len(data.label_names)

    if n_classes > 2:
        precision = dict()
        recall = dict()
        average_precision = dict()

        for i in range(n_classes):
            precision[i], recall[i], _ = metrics.precision_recall_curve(
                data.labels[:, i], scores[:, i]
            )
            average_precision[i] = metrics.average_precision_score(
                data.labels[:, i], scores[:, i]
            )

        # Compute micro-average ROC curve and ROC area
        precision["m"], recall["m"], _ = metrics.precision_recall_curve(
            data.labels.ravel(), scores.ravel()
        )
        average_precision["m"] = metrics.average_precision_score(
            data.labels, scores, average="micro"
        )

        plt.clf()
        plt.title('Multi-class precision-recall curves')

        plt.plot(recall["m"], precision["m"],
                 label='micro-averaged PR curve (AUC={0:0.2f})'
                       ''.format(average_precision["m"]))
        label = 'PR curve of class {0} (AUC={1:0.2f})'

        for i in range(n_classes):
            plt.plot(recall[i], precision[i],
                     label=label.format(data.label_names[i],
                                        average_precision[i]))
    else:
        precision, recall, _ = metrics.precision_recall_curve(
            data.labels, scores, pos_label=data.min_label
        )
        average_precision = metrics.average_precision_score(
            data.labels, scores, average="micro"
        )
        plt.clf()
        plt.title('Precision-recall curve')
        label = 'PR curve (AUC={0:0.2f})'
        plt.plot(recall, precision, label=label.format(average_precision))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")
    plt.show()
