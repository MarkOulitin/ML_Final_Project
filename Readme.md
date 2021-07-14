### Description
This project attempts to implement and evaluate [Virtual Adversarial Training](https://arxiv.org/pdf/1704.03976.pdf) (or VAT for short).
Only supervised datasets are supported.

### Packages required
Run the following command:
`pip install --upgrade tensorflow keras pandas sklearn scipy scikit_posthocs`

**NOTE:** numpy will be automatically installed with tensorflow

The versions we worked with were:
```
Keras                    2.4.3
numpy                    1.19.5
pandas                   1.3.0
scikit-learn             0.24.2
scikit-posthocs          0.6.7
scipy                    1.7.0
sklearn                  0.0
tensorflow               2.5.0
```

### Running the program
`python main.py [-h] [-cpu] [-no-save] [--gpu-mem-limit GPU_MEM_LIMIT]
               <algorithm-kind>`
#### Running the evaluation on a dataset

By default `main.py` will evaluate all datasets sequentially. The datasets names are specified in `Datasets.py` in the return value of a function `get_datasets_names`.

In order to evaluate a subset of these datasets, put in a comment all datasets that you don't want to be evaluated.

**You must specify** an algorithm for evaluation over the chosen dataset(s). To do so, add a corresponding program argument.

#### Program arguments
* algorithm-kind can be one of the following: 'Article', 'OUR', 'Dropout'
  - Article: The original VAT algorithm
  - OUR: Train using VAT with our small change: Compute R_vadv using the actual label information instead of the predicted labels
  - Dropout: Normal training, no special algorithm
* \-cpu specifies the calculations should be forced to run on the CPU
* \--gpu-mem-limit specifies the maximum amount of VRAM allowed for running the program
* \-no-save specifies the program should not save evaluation results to files

#### Running the statistic tests
    python statistics_tests.py
In order to change the alpha or metrics for execution the statistics test, open `statistics_tests.py` and edit lines 135, 134 respectively.

The metrics that are available, are the columns in the file `Results.xlsx`.

### Result files
The `Results.xlsx` file contains the metrics and performance of all algorithm over all datasets.

The `Posthoc_Results.xlsx` file contains the matrix of post hoc test that compares each pair of algorithms.
In our case, post hoc test was executed with TPR metric.