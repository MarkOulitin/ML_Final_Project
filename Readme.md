### Description
This project attempts to implement and evaluate [Virtual Adversarial Training](https://arxiv.org/pdf/1704.03976.pdf) (or VAT for short).
Only supervised datasets are supported.

### Packages required
Run the following command:
`pip install --upgrade tensorflow keras pandas sklearn scipy scikit_posthocs`

NOTE: numpy will be automatically installed with tensorflow

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
* algorithm-kind can be one of the following: 'Article', 'OUR', 'Dropout'
  - Article: The original VAT algorithm
  - OUR: Train using VAT with our small change: Compute R_vadv using the actual label information instead of the predicted labels
  - Dropout: Normal training, no special algorithm
* \-cpu specifies the calculations should be forced to run on the CPU
* \--gpu-mem-limit specifies the maximum amount of VRAM allowed for running the program
* \-no-save specifies the program should not save evaluation results to files
