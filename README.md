# Prompting

## 1. TL;DR Examples
### Example Code 1-1
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')
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
model = GPT2LMHeadModel.from_pretrained('gpt2')
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
model = GPT2LMHeadModel.from_pretrained('gpt2')
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
model = GPT2LMHeadModel.from_pretrained('gpt2')
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
model = GPT2LMHeadModel.from_pretrained('gpt2')
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
model = GPT2LMHeadModel.from_pretrained('gpt2')
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
model = GPT2LMHeadModel.from_pretrained('gpt2')
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

### Example Code 2-3: best movie of this year (=>)
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')
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
model = GPT2LMHeadModel.from_pretrained('gpt2')
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
model = GPT2LMHeadModel.from_pretrained('gpt2')
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
model = GPT2LMHeadModel.from_pretrained('gpt2')
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

