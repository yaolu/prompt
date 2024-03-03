# Prompting
   
## 1. TL;DR Examples
### Example Code 1-1
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "Our group is part of the UCL Computer Science department, affiliated with "
    "CSML and based at 90, High Holborn, London. We also organise the South "
    "England Natural Language Processing Meetup. If you are interested in "
    "doing a PhD with us, please have a look at these instructions. We also "
    "host a weekly reading group, you can find more details here."
)
# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=32)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```
Output: 

`
We are also a member of the University of Cambridge's Computer Science Department. We are also a member of the University of Cambridge's Computer Science Department.
`



### Example Code 1-2
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "Our group is part of the UCL Computer Science department, affiliated with "
    "CSML and based at 90, High Holborn, London. We also organise the South "
    "England Natural Language Processing Meetup. If you are interested in "
    "doing a PhD with us, please have a look at these instructions. We also "
    "host a weekly reading group, you can find more details here."
    " TL;DR"
)
# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=32)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```
Output:

`
: We are a group of computer scientists who are interested in learning about the world of computer science. We are looking for people who are interested in learning about the
`

### Example Code 1-3
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "Our group is part of the UCL Computer Science department, affiliated with "
    "CSML and based at 90, High Holborn, London. We also organise the South "
    "England Natural Language Processing Meetup. If you are interested in "
    "doing a PhD with us, please have a look at these instructions. We also "
    "host a weekly reading group, you can find more details here."
    " tl;dr"
)
# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=32)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```
Output:

`
: We are a group of computer scientists who have been working on the problem of language processing for over 20 years. We are a group of people who have been
`

### Example Code 1-4
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "Our group is part of the UCL Computer Science department, affiliated with "
    "CSML and based at 90, High Holborn, London. We also organise the South "
    "England Natural Language Processing Meetup. If you are interested in "
    "doing a PhD with us, please have a look at these instructions. We also "
    "host a weekly reading group, you can find more details here."
    " tldr"
)
# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=32)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```
Output:

`
.org.uk
The University of Cambridge
The University of Cambridge is a research university in the UK. It is a research university with a focus
`


### Example Code 1-5
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "Our group is part of the UCL Computer Science department, affiliated with "
    "CSML and based at 90, High Holborn, London. We also organise the South "
    "England Natural Language Processing Meetup. If you are interested in "
    "doing a PhD with us, please have a look at these instructions. We also "
    "host a weekly reading group, you can find more details here."
    " tldr:"
)
# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=32)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```
Output:

`
 http://www.tldr.org/
The University of Cambridge
The University of Cambridge is a research university in the UK. It is
`


## 2. In-context Learning: Do you like this movie?
### Example Code 2-1: worst movie of this year

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "Review: featuring an oscar-worthy performance\nSentiment: positive\n"
    "Review: completely messed up\nSentiment: negative\n"
    "Review: masterpiece\nSentiment: positive\n"
    "Review: the action is stilted\nSentiment: negative\n"
    "Review: by far the worst movie of the year\nSentiment:"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=1)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```

Output: 
`
negative
`

### Example Code 2-2: best movie of this year
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "Review: featuring an oscar-worthy performance\nSentiment: positive\n"
    "Review: completely messed up\nSentiment: negative\n"
    "Review: masterpiece\nSentiment: positive\n"
    "Review: the action is stilted\nSentiment: negative\n"
    "Review: by far the best movie of the year\nSentiment:"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=1)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```

Output: 
`
negative
`

### Example Code 2-3: (=>) magic
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "Review: featuring an oscar-worthy performance => Sentiment: positive\n"
    "Review: completely messed up => Sentiment: negative\n"
    "Review: masterpiece => Sentiment: positive\n"
    "Review: the action is stilted => Sentiment: negative\n"
    "Review: by far the best movie of the year => Sentiment:"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=1)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```

Output: 
`
positive
`

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "Review: featuring an oscar-worthy performance => Sentiment: positive\n"
    "Review: completely messed up => Sentiment: negative\n"
    "Review: masterpiece => Sentiment: positive\n"
    "Review: the action is stilted => Sentiment: negative\n"
    "Review: by far the worst movie of the year => Sentiment:"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=1)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```

Output: 
`
negative
`

Accuracy: 100%

### Example Code 2-4: use GPT-3 template
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "featuring an oscar-worthy performance => positive\n"
    "completely messed up => negative\n"
    "masterpiece => positive\n"
    "the action is stilted => negative\n"
    "by far the best movie of the year =>"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=1)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```

Output: 
`
positive
`

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "featuring an oscar-worthy performance => positive\n"
    "completely messed up => negative\n"
    "masterpiece => positive\n"
    "the action is stilted => negative\n"
    "by far the worst movie of the year =>"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=1)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```

Output: 
`
positive
`

Accuracy: 50%

## 3. Prompt ordering and positional bias: like movie you this Do?

### Example code 3-1: context ordering sensitivity

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel    

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document_a = (
    "completely messed up => negative\n"
    "the action is stilted => negative\n"
    "masterpiece => positive\n"
    "featuring an oscar-worthy performance => positive\n"
    "by far the worst movie of the year =>"
)

document_b = (
    "featuring an oscar-worthy performance => positive\n"
    "masterpiece => positive\n"
    "completely messed up => negative\n"
    "the action is stilted => negative\n"
    "by far the worst movie of the year =>"
)

document_c = (
    "completely messed up => negative\n"
    "featuring an oscar-worthy performance => positive\n"
    "the action is stilted => negative\n"
    "masterpiece => positive\n"
    "by far the worst movie of the year =>"
)

for document in [document_a, document_b, document_c]:
    input_ids = tokenizer.encode(document, return_tensors='pt')
    model_output = model.generate(input_ids, do_sample=False, max_new_tokens=1)
    output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])
    print(output_text)

```

Output: 
`
positive, negative, positive
`


```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel    

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document_a = (
    "completely messed up => negative\n"
    "the action is stilted => negative\n"
    "masterpiece => positive\n"
    "featuring an oscar-worthy performance => positive\n"
    "by far the best movie of the year =>"
)

document_b = (
    "featuring an oscar-worthy performance => positive\n"
    "masterpiece => positive\n"
    "completely messed up => negative\n"
    "the action is stilted => negative\n"
    "by far the best movie of the year =>"
)

document_c = (
    "completely messed up => negative\n"
    "featuring an oscar-worthy performance => positive\n"
    "the action is stilted => negative\n"
    "masterpiece => positive\n"
    "by far the best movie of the year =>"
)

for document in [document_a, document_b, document_c]:
    input_ids = tokenizer.encode(document, return_tensors='pt')
    model_output = model.generate(input_ids, do_sample=False, max_new_tokens=1)
    output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])
    print(output_text)

```

Output: 
`
positive, positive, positive
`
### Example code 3-2: let's take a look at distribution

```python

from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "featuring an oscar-worthy performance => positive\n"
    "completely messed up => negative\n"
    "masterpiece => positive\n"
    "the action is stilted => negative\n"
    "by far the worst movie of the year =>"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

positive_token_id = 3967
negative_token_id = 4633
with torch.inference_mode():
    model_output = model(input_ids)
    prob_dist = model_output.logits[:, -1, :].softmax(dim=-1)
print(f"positive probability: {prob_dist[:, positive_token_id]}")
print(f"negative probability: {prob_dist[:, negative_token_id]}")
```

Output: 
`
positive probability: tensor([0.3157]) 
negative probability: tensor([0.2891])
`

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "featuring an oscar-worthy performance => positive\n"
    "completely messed up => negative\n"
    "masterpiece => positive\n"
    "the action is stilted => negative\n"
    "by far the best movie of the year =>"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

positive_token_id = 3967
negative_token_id = 4633
with torch.inference_mode():
    model_output = model(input_ids)
    prob_dist = model_output.logits[:, -1, :].softmax(dim=-1)
print(f"positive probability: {prob_dist[:, positive_token_id]}")
print(f"negative probability: {prob_dist[:, negative_token_id]}")
```
Output:
`
positive probability: tensor([0.3802]) 
negative probability: tensor([0.1540])
`

### Example code 3-3: simple calibration
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "featuring an oscar-worthy performance => positive\n"
    "completely messed up => negative\n"
    "masterpiece => positive\n"
    "the action is stilted => negative\n"
    "N/A =>"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

positive_token_id = 3967
negative_token_id = 4633
with torch.inference_mode():
    model_output = model(input_ids)
    prob_dist = model_output.logits[:, -1, :].softmax(dim=-1)
print(f"context-free positive probability: {prob_dist[:, positive_token_id]}")
print(f"context-free negative probability: {prob_dist[:, negative_token_id]}")

# orginal implementation: https://github.com/tonyzhaozh/few-shot-learning/blob/e04d8643be91c2cce63f33e07760ff75d5aa3ad0/run_classification.py#L121
num_classes = 2
p_cf = prob_dist[0, [positive_token_id, negative_token_id]].numpy()
calibration_weight_matrix = np.linalg.inv(np.identity(num_classes) * p_cf)
print(calibration_weight_matrix)

calibrated_prob = np.matmul(calibration_weight_matrix, np.expand_dims(p_cf, axis=-1))
# don't forget softmax
```

Output: 
`
context-free positive probability: tensor([0.0453])
context-free negative probability: tensor([0.1094])
[[22.07082467  0.        ]
 [ 0.          9.14129541]]
`


## 4. Label Selection

### Example code 4-1: cat versus dog
```python

from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# cat for positive, dog for negative
document = (
    "featuring an oscar-worthy performance => cat\n"
    "completely messed up => dog\n"
    "masterpiece => cat\n"
    "the action is stilted => dog\n"
    "by far the worst movie of the year =>"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

positive_token_id = 3797
negative_token_id = 3290
with torch.inference_mode():
    model_output = model(input_ids)
    prob_dist = model_output.logits[:, -1, :].softmax(dim=-1)
print(f"positive (cat) probability: {prob_dist[:, positive_token_id]}")
print(f"negative (dog) probability: {prob_dist[:, negative_token_id]}")
```

Output:
`
positive (cat) probability: tensor([0.1997])
negative (dog) probability: tensor([0.1463])
`

```python

from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# dog for positive, cat for negative
document = (
    "featuring an oscar-worthy performance => dog\n"
    "completely messed up => cat\n"
    "masterpiece => dog\n"
    "the action is stilted => cat\n"
    "by far the worst movie of the year =>"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

positive_token_id = 3290
negative_token_id = 3797
with torch.inference_mode():
    model_output = model(input_ids)
    prob_dist = model_output.logits[:, -1, :].softmax(dim=-1)
print(f"positive (dog) probability: {prob_dist[:, positive_token_id]}")
print(f"negative (cat) probability: {prob_dist[:, negative_token_id]}")
```
Output:
`
positive (dog) probability: tensor([0.2070])
negative (cat) probability: tensor([0.1719])
`
### Example code 4-2: explore label space
```python

from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "featuring an oscar-worthy performance => positive\n"
    "completely messed up => negative\n"
    "masterpiece => positive\n"
    "the action is stilted => negative\n"
    "by far the worst movie of the year =>"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

with torch.inference_mode():
    model_output = model(input_ids)
    prob_dist = model_output.logits[:, -1, :].softmax(dim=-1)
print([(p.item(), tokenizer.decode(token_id)) for (p, token_id) in zip(*prob_dist[0].topk(10))])
```
Output:
`
[(0.31566739082336426, ' positive'), (0.2891405522823334, ' negative'), (0.02993960492312908, ' bad'), (0.013837959617376328, ' good'), (0.009425444528460503, ' very'), (0.008499860763549805, ' great'), (0.005330370739102364, ' terrible'), (0.004886834882199764, ' not'), (0.004764571785926819, ' perfect'), (0.004176552407443523, ' no')]
`

### Example code 4-3: open question - can we use digits?
```python

from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "featuring an oscar-worthy performance => 0\n"
    "completely messed up => 1\n"
    "masterpiece => 0\n"
    "the action is stilted => 1\n"
    "by far the worst movie of the year =>"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

with torch.inference_mode():
    model_output = model(input_ids)
    prob_dist = model_output.logits[:, -1, :].softmax(dim=-1)
print([(p.item(), tokenizer.decode(token_id)) for (p, token_id) in zip(*prob_dist[0].topk(10))])
```
Output:
`
[(0.43802526593208313, ' 1'), (0.34042462706565857, ' 0'), (0.10940475016832352, ' 2'), (0.021524671465158463, ' 3'), (0.013400197960436344, ' 4'), (0.00964546948671341, ' 5'), (0.004922178573906422, ' 6'), (0.004260277841240168, ' 8'), (0.003846667939797044, ' 9'), (0.003295806935057044, ' 7')]
`

```python

from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "featuring an oscar-worthy performance => 7\n"
    "completely messed up => 11\n"
    "masterpiece => 7\n"
    "the action is stilted => 11\n"
    "by far the worst movie of the year =>"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

with torch.inference_mode():
    model_output = model(input_ids)
    prob_dist = model_output.logits[:, -1, :].softmax(dim=-1)
print([(p.item(), tokenizer.decode(token_id)) for (p, token_id) in zip(*prob_dist[0].topk(10))])
```

Output:
`
[(0.18294943869113922, ' 7'), (0.09998831897974014, ' 9'), (0.09698300063610077, ' 8'), (0.08610329777002335, ' 6'), (0.0774054080247879, ' 11'), (0.06776292622089386, ' 10'), (0.06541603803634644, ' 5'), (0.04681426286697388, ' 4'), (0.03447182849049568, ' 12'), (0.027355313301086426, ' 3')]
`

### Example code 4-4: I have wrong labels

```python

from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "featuring an oscar-worthy performance => negative\n"
    "completely messed up => positive\n"
    "masterpiece => positive\n"
    "the action is stilted => negative\n"
    "by far the worst movie of the year =>"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

with torch.inference_mode():
    model_output = model(input_ids)
    prob_dist = model_output.logits[:, -1, :].softmax(dim=-1)
print([(p.item(), tokenizer.decode(token_id)) for (p, token_id) in zip(*prob_dist[0].topk(10))])
```

Output: 
`
[(0.3884822428226471, ' positive'), (0.3839123547077179, ' negative'), (0.01633561961352825, ' bad'), (0.008512002415955067, ' good'), (0.005889976862818003, ' very'), (0.004109138157218695, ' great'), (0.0031330634374171495, ' negatively'), (0.002932012313976884, ' not'), (0.0027931600343436003, ' no'), (0.0025334602687507868, ' perfect')]
`
Previous Output: 
`
[(0.31566739082336426, ' positive'), (0.2891405522823334, ' negative'), (0.02993960492312908, ' bad'), (0.013837959617376328, ' good'), (0.009425444528460503, ' very'), (0.008499860763549805, ' great'), (0.005330370739102364, ' terrible'), (0.004886834882199764, ' not'), (0.004764571785926819, ' perfect'), (0.004176552407443523, ' no')]
`

## 5. Beyond Classification - Let's do complex tasks

### Example code 5-1: Can GPT2 do math?
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?\n"
    "A: The answer is 11.\n"
    "Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?\n"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=16)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```

Output:
`
A: The answer is 3.
Q: Roger has a lot of money
`




```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

document = (
    "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?\n"
    "A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.\n"
    "Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?\n"
)

# Generate input IDs from the document using the tokenizer
input_ids = tokenizer.encode(document, return_tensors='pt')

# Generate model output using the input IDs
model_output = model.generate(input_ids, do_sample=False, max_new_tokens=32)

# Decode the model output into text using the tokenizer
output_text = tokenizer.decode(model_output[0, input_ids.shape[1]:])

# Print the output text
print(output_text)
```

Output:
`
A: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
Q: Roger has
`






