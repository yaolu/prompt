# prompt

Example Code
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
```
We are also a member of the University of Cambridge's Computer Science Department.
We are also a member of the University of Cambridge's Computer Science Department.
```
