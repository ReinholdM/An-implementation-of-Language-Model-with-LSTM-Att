# An-implementation-of-Language-Model-with-LSTM-Attention
Windows | Pytorch
## The workflow of code as follows
Batch the datasets --  Embedding the tokens with nn.embedding from **Pytorch** -- LSTM~--~Attention with attention_width on the hidden in the dimension of batch_size -- Fc linear layers with softmax finally  

Something has to remind is that the inputs and labels are tokens with a length of dictionary and shift a token from the inputs with a length of dictionary. And the **\<eos\>** and **\<sos\>** is added into the dictionary.

```bash
usage  
python tmp.py
```

## The description of datasets
> The **train.txt | valid.txt | test.txt** is easy to understand.
>> The **wordlist.txt** is the lexicon of this datasets.

## Some logs

```python
| epoch   1 |   100/238382 batches | lr 20.00 | ms/batch 14.75 | loss  7.76 | perplexity  2350.92
| epoch   1 |   110/238382 batches | lr 20.00 | ms/batch 13.19 | loss  7.83 | perplexity  2508.62
| epoch   1 |   120/238382 batches | lr 20.00 | ms/batch 13.80 | loss  7.84 | perplexity  2549.26
| epoch   1 |   130/238382 batches | lr 20.00 | ms/batch 18.69 | loss  7.87 | perplexity  2621.23
```
