========
classipy
========

-------------------------------------
An automated text classification tool
-------------------------------------

`classipy` provides a command-line interace to develop and run a text classifier.

Overview
========

The classifiers supported are those that SciKit-Learn_ provides and that make sense to use for this scenario:
Ridge Regression, Linear SVM, Random Forest, Maximum Entropy/Logistic Regression, and the Naïve Bayes classifiers.
There is no support for Deep Learning because the more common case is that one only has a very small labeled set.
If you do have a large labeled corpus and are interested in using a Neural Network, it probably would be not that difficult to extend the sources, however.

The main contribution, however, is the enhanced feature generation process that is far more complex than what "off-the-shelf" SciKit-Learn tools have to offer.
First, `classipy` uses my own segtok_ sentence segmentation and word tokenization library.
Second, it can properly handle multiple text fields (e.g., title, abstract, body, ...).
Third, it can integrate metadata (annotations) either on a per-feature or per-instance basis.
Fourth, n-grams are not generated beyond sentence boundaries.
Fifth, k-shingles - i.e., all possible n-gram combinations of the article - can be used.
Finally, TF-IDF feature transformation, evaluation functions, and grid-search-based learning have been carefully tuned to make it easy to greatly accelerate the development of text classifiers.

Overall, this library's declared goal is to make text classification as simple as it ever could be made with the current state of research and using software that is (reasonably) production-ready.
For evaluation, you *should* be using `AUC PR`_ for a rank-based text classification result (see my `CrossValidated post`_ discussing why it should be preferred over AUC ROC, with a reference to the relevant paper).
For unranked evaluations, and as the global optimization metric, the `MCC Score`_ is used.

The general concept followed by `classipy` is to generate an inverted index (feature matrix) and a vocabulary (dictionary) from the input text.
With those two files, `classipy` then allows you to learn and evaluat a model.
You also can cross-validate a pre-parametrized classifier on the inverted index to tune the right feature generation process.
Once you are happy with the model you built, you can use it to run predictions on (unlabeled) text data-streams (i.e., UNIX pipes).

Install
=======

::

    pip3 install classipy

This libary has two strong depencies, like SciKit-Learn_ (and, thus, SciPy and NumPy) and segtok_ (and thus, regex_, a faster C-based regular expression library).
In addtion, but not required, to plot the main evaluation function, you will need to install matplotlib_ and some graphics library it can use, e.g. PySide_.

Usage
=====

`classipy` provides itself as a command-line script (`classi.py`) and uses a command (word) to run various kinds of tasks:

- `generate` generates inverted indices (feature matrices)
- `learn` allows you to train and develop a classifier model
- `evaluate` makes it possible to directly evaluate a classifier or to evaluate a learned model
- `predict` runs a model over (unseen/unlabled) input text or a provided feature matrix
- `labels` prints the (class) labels of an inverted index (feature matrix)
- `vocabulary` prints the (word) features of an index' vocabulary file

For details, use the built-in `--help` (`-h`) function.
The overall way to use this tool then is as follows:

The input text should be provided in classical CSV (as produced by Excel) or TSV (as used on UNIX) format.
By default, the columns should be: An ID column, as many text columns as desired, as many metadata/annotation columns as desired, and one label column.
Multi-label (multinomial) support is already built-in and automatically detected/handled.

The suggested first step should probably be that you somehow split the corpus in a training+development ("learning") and a test set.
Typically, such splits are 3:2, 4:1, or 9:1, and I suggest to use 4:1.
For splitting the learning set into development and training set during parameter grid-search, the tool internally uses a 1:3 split with 4-fold CV.
So, if you use an 4:1 learning:test split, `classipy` will make a 3:1 traininig:development split on the remaining data, for an overall 3:1:1 split.
This choice was made as to make the library tuned towards small annotated corpora/datasets where you need to set aside most of the little data you have as to not hopelessly overfit while developing your model.

Assuming you generated two sets, `learning.tsv` and `test.tsv`, the next step is to generated the inverted indices/feature matrices from the two sets.
First, generate the learning index `.idx` and bootstrap the vocabulary `.voc` from the learning set text `.tsv`::

    classi.py -VV generate --vocabulary the.voc learning.idx learning.tsv [YOUR OPTIONS]

This gives you the vocabulary your classifier will use and a combined training+development matrix.

Quickly check your feature generation has produced an index that can build a classifier somewhat in the performance range you are amining at::

    classi.py -VV evaluate learning.idx [YOUR OPTIONS]

If that is "good enough", generate a test matrix with the same vocabulary while you are at it::

    classi.py -VV generate --vocabulary the.voc test.idx test.tsv [SAME OPTIONS*]

Here, the vocabulary now is used to "select" the same features as used for the training set.
*Therefore, it is not necessary to use either of the two feature selection options here (`--select` or `--cutoff`).

Next, grid-search for the "perfect" parameters for you classifier and use them to build the "final" model::

    classi.py -VV learn --vocabulary the.voc --grid-search text.idx the.model [YOUR OPTIONS]

Note that this might take a while, even hours or days, if your vocabulary or text collection is huge (hundred-thousands or millions).
After the model has been built, you can now evaluate it the untouched test data you set aside in the beginning::

    classi.py -VV evaluate --vocabulary the.voc text.idx the.model [--pr-curve]

The only relevant option here, `--pr-curve`, can be used if you have matplotlib_ installed (and correctly configured...) to plot the precision-recall curve.
Assuming you are happy with the result, you now can classify new texts::

    classi.py predict --vocabulary the.voc --text [--score] [GENERATE OPTOINS] moar_text.tsv

`predict` can also read text in columnar format off the STDIN, so it can be used in UNIX data pipelines, and naturally also works with pre-generated index files.
Naturally, it can print the confidence scores for each prediction (binary labels: one score; multi-labels: one score for each label); see `--scores`.

Finally, `classipy` has a number of additional tricks up its sleeve that you can learn by reading the (command-line help) documentation.

.. _AUC PR:: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
.. _CrossValidated post:: http://stats.stackexchange.com/questions/7207/roc-vs-precision-and-recall-curves/158354#158354
.. _MCC Score:: https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

.. _SciKit-Learn:: http://scikit-learn.org/
.. _matplotlib:: http://matplotlib.org/
.. _PySide:: https://pypi.python.org/pypi/PySide
.. _regex:: https://pypi.python.org/pypi/regex
.. _segtok:: https://pypi.python.org/pypi/segtok