from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np

# # Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# # Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# # Create an LLM.
# llm = LLM(model="facebook/opt-125m")
# # Generate texts from the prompts. The output is a list of RequestOutput objects
# # that contain the prompt, generated text, and other information.
# outputs = llm.generate(prompts, sampling_params)
# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

def load_dataset(file_path):
    '''
    example:
    >>> dataset=load_dataset("path/to/dataset.json")    # DataFrame

    >>> dataset=dataset.to_dict(orient='records')
    >>> print(dataset[:5])
    '''
    if file_path.startswith("/"):
        file_path = file_path[1:]
    # Load the dataset.
    try:
        dataset = pd.read_json(f"file://localhost/{file_path}", lines=True)
    except ValueError as e:
        print(f"Error loading JSON: {e}")
        return None
    # Display the first few rows of the dataset.
    # print(dataset.head())
    # Return the dataset.
    return dataset
