import openai
from openai import OpenAI

import os
import time
from functools import cache

from sentence_transformers import SentenceTransformer


class OpenAILLM:

    def __init__(self, llm_model_name, embedding_model_name):
        self.client = OpenAI()
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name

    def get_llm_response(self, prompt, max_tokens=1024, timeout=60):
        n_retries = 10
        for i in range(n_retries):
            try:
                chat_completion = self.client.chat.completions.create(model=self.llm_model_name, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, timeout=timeout)
                return chat_completion.choices[0].message.content
            except openai.APIError:
                print("openai.error.ServiceUnavailableError")
                pass
            except openai.APITimeoutError:
                print("openai.error.Timeout")
                pass
            except openai.APIError:
                print("openai.error.APIError")
                pass
            except openai.APIConnectionError:
                pass
            except openai.RateLimitError:
                print("openai.error.RateLimitError")
                time.sleep(10)
            # too many tokens
            except openai.BadRequestError:
                context_window = 3000 * 4 # max length in chars (every token is around 4 chars)
                prompt = prompt[:context_window]
                print("openai.error.InvalidRequestError")

        raise ValueError(f"OpenAI remains uncontactable after {n_retries} retries due to either Server Error or Timeout after {timeout} seconds")

    @cache
    def get_embeddings(self, query):
        response = openai.embeddings.create(
            input=query,
            model=self.embedding_model_name
        )
        embeddings = response.data[0].embedding
        return embeddings

class LocalLLM(OpenAILLM):

    def __init__(self, llm_model_name, embedding_model_name):
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

    @cache
    def get_embeddings(self, query):
        response = self.embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        embeddings = list(response[0])
        return embeddings

class MindsDBLLM(OpenAILLM):
    
    def __init__(self, llm_model_name, embedding_model_name):
        self.client = OpenAI(base_url="https://llm.mdb.ai", api_key=os.getenv("MINDSDB_API_KEY"))
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name