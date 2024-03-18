***<span style="font-size: 3em;">:warning:</span>You must agree to the [license](https://github.com/Sinhala-NLP/NSINA?tab=License-1-ov-file#readme) and terms of use before using the dataset in this repo.***

# Sinhala Headline Generation
This is a text generation task created with the [NSINA dataset](https://github.com/Sinhala-NLP/NSINA). This dataset is also released with the same license as NSINA. 



## Data
Data can be loaded into pandas dataframes using the following code. 

```python
from datasets import Dataset
from datasets import load_dataset

train = Dataset.to_pandas(load_dataset('sinhala-nlp/NSINA-Headlines', split='train'))
test = Dataset.to_pandas(load_dataset('sinhala-nlp/NSINA-Headlines', split='test'))
```

## Citation
If you are using the dataset or the models, please cite the following paper.

~~~
﻿@inproceedings{Nsina2024,
author={Hettiarachchi, Hansi and Premasiri, Damith and Uyangodage, Lasitha and Ranasinghe, Tharindu},
title={{NSINA: A News Corpus for Sinhala}},
booktitle={The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
year={2024},
month={May},
}
~~~