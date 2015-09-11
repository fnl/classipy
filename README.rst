========
classipy
========

-------------------------------------
An automated text classification tool
-------------------------------------

``classipy`` is a command-line tool for developing statistical models that the can be used classify text (streams).

Overview
========

The library is based on SciKit-Learn_ and provides classifiers that make sense to use for this scenario:
Ridge Regression, various SVMs, Random Forest, Maximum Entropy/Logistic Regression, and Naïve Bayes classifiers.
There is no support for Deep Learning because the more common case is that one only has a small labeled set.
Admittedly, however, adding neural word embeddings would be a useful ability to add to this tool.

The main addition of this tool to what SciKit Learn provides is a greatly enhanced feature generation process.
It is far more complex than what "off-the-shelf" SciKit-Learn tools have to offer and supports meta-data annotations.

1. ``classipy`` uses the segtok_ sentence segmentation and word tokenization library.
2. It can properly handle (and distinguish tokens from) multiple text fields (e.g., title, abstract, body, ...).
3. It can integrate and combine meta-data (annotations) on both a per-feature or a per-instance basis.
4. n-grams are not generated beyond word boundaries (i.e., not n-grams containing commas or dots, etc.).
5. k-shingles - all possible token combinations that can be generated from the text, not just successive tokens as in n-grams - can be added as another feature set.
6. TF-IDF feature transformation, feature extraction techniques, evaluation functions, and grid-search-based learning facilities are built-in.

All this has been carefully tuned to greatly accelerate and facilitate the development of high-end text classifiers.
For evaluation, this library provides the `AUC PR`_ score for a rank-based text classification result
(See my `CrossValidated post`_ discussing why it should be preferred over AUC ROC, with a reference to the relevant paper.)
For unranked evaluations, and as the global optimization metric, the `MCC Score`_ is used.
Standard functions, in particular F-measure or accuracy, are reported, too.

The general concept followed by ``classipy`` is to generate an inverted index (feature matrix) and a vocabulary (dictionary) from the input text.
With those two files, ``classipy`` then allows you to learn and evaluate a model.
You also can cross-validate a pre-parametrized classifier on the inverted index to tune the right feature generation process.
Once you are happy with the model you built, you can use it to run predictions on (unlabeled) text data-streams (i.e., UNIX pipes).
Multiclass (multinomial) support is built-in and automatically detected/handled (usually as OvR a.k.a. OvA).

.. _SciKit-Learn: http://scikit-learn.org/
.. _segtok: https://pypi.python.org/pypi/segtok
.. _AUC PR: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
.. _CrossValidated post: http://stats.stackexchange.com/questions/7207/roc-vs-precision-and-recall-curves/158354#158354
.. _MCC Score: https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

Install
=======

::

    pip3 install classipy

This package has two strong dependencies: SciKit-Learn_ (and, thus, SciPy and NumPy) and segtok_ (and thus, regex_, a faster C-based regular expression library).
In addition, but not required, to plot the main evaluation function (AUC PR), you will need to install matplotlib_ and configure some graphics library for it to use, e.g. PySide_.

.. _matplotlib: http://matplotlib.org/
.. _PySide: https://pypi.python.org/pypi/PySide
.. _regex: https://pypi.python.org/pypi/regex

Usage
=====

``classipy`` provides itself as a command-line script (``classipy``) and uses a command (word) to run various kinds of tasks:

- ``generate`` generates inverted indices (feature matrices)
- ``select`` can do feature selection over an already generated matrix
- ``learn`` allows you to train and develop a classifier model
- ``evaluate`` makes it possible to directly evaluate a classifier or to evaluate a learned model
- ``predict`` runs a model over (unseen/unlabled) input text or a provided feature matrix

For details and additional command-words, use the built-in ``--help`` (``-h``) function.
The overall way to use this tool is described in the following.

The input text should be provided in classical CSV (as produced by Excel) or TSV (as used on UNIX) format.
That is, for Excel, strings are enclosed by double quotes and double quotes in strings are "escaped" by repeating them (two double quotes).
For the TSV format, all fields are separated by tabs, there is no special string marker, and tabs inside strings are escaped with a backslash.
By default, the columns should be: An ID column, as many text columns as desired, as many metadata/annotation columns as desired, and one label column.
The tool has options to allow for ID and label columns in other positions.

The suggested first step is to split the corpus in a training+development ("learning") and a test set.
Typically, such splits are 3:2, 4:1, or 9:1, and I suggest to use 4:1.
The tool internally uses a 1:3 split with 4-fold CV to splitting the learning set into the development and training set during parameter grid-search.
So, if you use a 4:1 learning:test split, ``classipy`` will make a 3:1 traininig:development split on the remaining data, for an overall 3:1:1 split (train:dev:test).
This choice was made as to make the library tuned towards small annotated corpora/datasets where you need to set aside most of the little data you have as to not overfit too much.

Assuming you created two text files, ``learning.tsv`` and ``test.tsv``, the next step is to **generate** the inverted indices/feature matrices from the two sets.
First, generate the learning index ``.idx`` and bootstrap the vocabulary ``.voc`` from the learning set text ``.tsv``::

    classi.py -VV generate --vocabulary the.voc learning.idx learning.tsv [YOUR OPTIONS]

This gives you the vocabulary your classifier will use and a combined training+development matrix.
During this step, you already can apply feature selection techniques to hold the generated vocabulary at bay.

Quickly check your feature generation has produced an index that can build a classifier in the approximate range of the performance you are hoping for::

    classi.py -VV evaluate learning.idx [YOUR OPTIONS]

If that result is too poor, you probably should think about other features to generate first.
If it is "good enough", generate a test matrix with the same vocabulary while you are at it::

    classi.py -VV generate --vocabulary the.voc test.idx test.tsv [SAME OPTIONS*]

Here, the (existing) vocabulary ``the.voc`` now gets used to *select* only those features that have been used to create final (feature-selected) the training set index.
Therefore, you should never use any of the feature selection options when generating test indices (e.g., ``--select`` or ``--cutoff``).

Next, ``--grid-search`` for the "perfect" parameters for you classifier and use them to **learn** the "final" model::

    classi.py -VV learn --vocabulary the.voc --grid-search text.idx the.model [YOUR OPTIONS]

Note that this might take a while, even hours or days, if your vocabulary or text collection is huge or your model very comples (see the options provided by ``learn``).
After the model has been built, you can now **evaluate** it on the unseen and unused test data you set aside in the beginning::

    classi.py -VV evaluate --vocabulary the.voc text.idx the.model [--pr-curve]

The only option, ``--pr-curve``, can only be used if you have matplotlib_ installed (and correctly configured...) to plot the precision-recall curve.
Assuming you are happy with the result, you now can **predict** lables for new texts with ``the.model``::

    classi.py predict --vocabulary the.voc --text [--score] [GENERATE OPTOINS] moar_text.tsv

``predict`` can also read text in columnar format off the STDIN, so it can be used in UNIX data pipelines, and naturally also works with pre-generated index files.
Naturally, it can print the confidence scores for each prediction (binary labels: one score; multi-labels: one score for each label); see ``--scores``.

Finally, ``classipy`` has a number of additional tricks up its sleeve that you can learn by reading the (command-line help) documentation.
One noteworthy trick is to impute model parameters in the learning process: See ``--parameters`` in the ``classipy learn -h`` output.
Important here is the format of the parameters, which is: " ``GROUP`` __ ``PARAMETER`` = ``VALUE`` ", with all parameters separated by commas.
The following ``GROUP`` values are allowed:

- ``classify`` for parameters of the classifier.
- ``filter`` for parameters for the L1-penalized feature extraction model (LinearSVC_ [or LogisticRegression_ for SVM-based classifiers]).
- ``scale`` for the feature scaling (Normalizer_) class.
- ``transform`` for the TFIDFTransformer_ class.
- ``prune`` for the VarianceThreshold_ class used by grid-search-based models.

This then makes it possible to induce parameters either to build your own model on the fly or to direct the gird search.

.. _LinearSVC: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
.. _LogisticRegression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
.. _Normalizer: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
.. _TFIDFTransformer: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
.. _VarianceThreshold: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html

Legal
=====

License: `GNU Affero General Public License v3`_

Copyright (c) 2015, Florian Leitner. All rights reserved.

.. _GNU Affero General Public License v3: https://www.gnu.org/licenses/agpl-3.0.en.html

History
=======

- **1.0.0** initial release
