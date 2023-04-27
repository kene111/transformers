# transformers
This repository contains the Implementation of the Transformers Architecture from the research paper ["Attention is all you need"](https://arxiv.org/abs/1706.03762) using the pytorch framework. This is repo created for two purposes, One, to understand the inner workings of the architecture as it the backbone of many high performing large language models, and is also being used in computer vision related tasks. The second is being able to know what layers to alter, and how to alter it if the need be. This could for the purpose of model compression, model optimization, and so on.


### Repository Breakdown:

In the transformer_baseline folder:

  1. blocks: This folder contains the ```encoder``` and the ```decoder``` implementations.
  2. configurations: This folder contains a ```config``` python script, this is used to set all the necessary configurations used all through the network.
  3. embeddings: This folder contains the embedding scripts used create the embedding layer for the transformer network.
  4. layers: The folder contains the key layers spoken about in the network. This includes, the ``` multi-head attention layer```, the ```self-attention layer``` or ```scaled dot product attention layer```, the ```point-wise fully connected layer```, the ```linear layer```, and  the ```normalization layer```.
  6. models: This folder contain the end to end implementation of the ```decoder```, ```encoder```, and the ```transformer```.
  7. utils: This folder contains the ```mask``` and ```positional encoder``` scripts.

### To test the script works, run:
```python test/test_transformer.py```




