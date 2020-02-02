---
title: Different levels of collate_fn
tags:
 - pytorch
---


Some tasks require to provide many objects per data point.
To handle creating batches of these complex data points, pytorch lets you write
a `collate_fn` ([official documentation](https://pytorch.org/docs/stable/data.html#DataLoader-collate-fn)).

This is really useful if you're trying to perform a task like BERT training:
```python
encoder_input_ids, encoder_mask, decoder_input_ids, decoder_mask, token_type_ids = batch
```

or Visual Question Answering
```python
img, question_ids, answer_ids, question_mask, answer_mask = batch
```

## Default collate_fn

The default `collate_fn` of pytorch will just perform a `torch.stack()`
on each tensor it receives.

BERT example:
```python
class BERTDataset(Dataset):
  ...
  def __getitem__(self, idx):
    text = self.texts[idx]
    ids, special_tokens_mask, position_ids, token_type_ids = encode(text)
    return ids, special_tokens_mask, position_ids, token_type_ids
```

If you have guarantees that every data point will return the same size
tensors from your `encode` function, then a setup like this can just use the
default `collate_fn`.

## New variables in collate_fn
Maybe you don't have the guarantee of same tensor size mentioned above.
You may need to pad all your inputs so that you can stack them into a batch.

```python
# lets consider a slightly simpler dataset.
class BERTDataset(Dataset):
  def __getitem__(self, idx):
    ...
    return torch.tensor(ids), torch.tensor(special_tokens_mask)

def custom_collate(examples):
  # the fn gets called with a list of return values from your Datasest.__getitem__().
  # in this case, we get a list of tuples.
  #   [(input, mask, (input, mask), ...]
  # we call zip() on them to separate them out into lists of the individual values.
  #   [input, input, ...], [mask, mask, ...]
  inputs, special_tokens_masks = zip(*examples)

  # now you can apply your sequence padding to the whole batch.
  inputs = pad_sequence(inputs, batch_first=True)
  special_tokens_masks = pad_sequence(special_tokens_masks, batch_first=True)
  return inputs, special_tokens_masks
```

## Dynamic collate_fn
Perhaps due to your encoding/tokenization scheme, you need to override the
default `padding_value` of `pad_sequence()`.

The simplest sounding thing would be to create your custom collate via closure
or lambda:
```python
def custom_collate(examples, padding_value):
  inputs, special_tokens_masks = zip(*examples)

  inputs = pad_sequence(inputs, batch_first=True, padding_value=padding_value)
  special_tokens_masks = pad_sequence(special_tokens_masks, batch_first=True, padding_value)
  return inputs, special_tokens_masks

# you can wrap the above function and pass that into DataLoader
collate_wrapper = lambda x: custom_collate(examples, tokenizer.pad_token_id)
DataLoader = DataLoader(bert_dataset, collate_fn=collate_wrapper)
```

This will only _kinda sorta_ work.  Another feature that DataLoaders expose is
having multiple worker processes for creating batches (`DataLoader(num_wokers=8)`).
This uses python's `multiprocessing` module, which in turn uses `pickle`.

`Pickle` can only pickle functions/objects defined at the "top-level" of a file.
That means that are above closure/lambda-generated `collate_fn` won't work
if `num_workers > 0`.

**The solution** is to use a [callable object](https://stackoverflow.com/questions/573569/python-serialize-lexical-closures).

You can define a class with a `__call__()` function, and then invoke the object
like a function, which will trigger `__call__()`.  The same "dynamic"
`collate_fn` above can be reimplemented this way:

```python
class Collater
  def __init__(self, padding_value=0):
    self.padding_value = padding_value
  def __call__(self, examples):
    inputs, special_tokens_masks = zip(*examples)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=self.padding_value)
    special_tokens_masks = pad_sequence(special_tokens_masks, batch_first=True, self.padding_value)
    return inputs, special_tokens_masks

collater = Collater(padding_value=tokenizer.pad_token_id)
DataLoader = DataLoader(bert_dataset, collate_fn=collater)
```

## Conclusion
I had a lot of trouble debugging this the first time.  My error looked like:

```
Exception:

-- Process 0 terminated with the following error:
...
  File "/home/shoarora/miniconda/envs/3l-dr/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 279, in __iter__
    return _MultiProcessingDataLoaderIter(self)
  File "/home/shoarora/miniconda/envs/3l-dr/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 719, in __init__
    w.start()
  File "/home/shoarora/miniconda/envs/3l-dr/lib/python3.6/multiprocessing/process.py", line 105, in start
    self._popen = self._Popen(self)
  File "/home/shoarora/miniconda/envs/3l-dr/lib/python3.6/multiprocessing/context.py", line 223, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "/home/shoarora/miniconda/envs/3l-dr/lib/python3.6/multiprocessing/context.py", line 284, in _Popen
    return Popen(process_obj)
  File "/home/shoarora/miniconda/envs/3l-dr/lib/python3.6/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/home/shoarora/miniconda/envs/3l-dr/lib/python3.6/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/home/shoarora/miniconda/envs/3l-dr/lib/python3.6/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/home/shoarora/miniconda/envs/3l-dr/lib/python3.6/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
TypeError: cannot serialize '_io.TextIOWrapper' object
```

It was really unclear what mart of my DataLoader object was failing to pickle, and it took a while before
I realized that the issue was with the `collate_fn`.  

Hopefully this can help someone (even if it's just future me) avoid this problem in the future.
