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