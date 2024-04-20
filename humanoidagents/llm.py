import openai
from openai import OpenAI

import os
import time
from functools import cache

from sentence_transformers import SentenceTransformer


class OpenAILLM:

    client = OpenAI()
    model_name = "gpt-3.5-turbo"

    @classmethod
    def get_llm_response(cls, prompt, max_tokens=1024, timeout=60):
        n_retries = 10
        for i in range(n_retries):
            try:
                chat_completion = cls.client.chat.completions.create(model=cls.model_name, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, timeout=timeout)
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

    @staticmethod
    @cache
    def get_embeddings(query):
        response = openai.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        embeddings = response.data[0].embedding
        return embeddings

class LocalLLM(OpenAILLM):
    
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    @classmethod
    @cache
    def get_embeddings(cls, query):
        response = cls.embedding_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        embeddings = list(response[0])
        return embeddings

class MindsDBLLM(OpenAILLM):
    # please note that this still calls embedding service from openai since MindsDB doesn't support embedding service
    
    client = OpenAI(base_url="https://llm.mdb.ai", api_key=os.getenv("MINDSDB_API_KEY"))
    model_name = "gpt-3.5-turbo" # this can be anything allowed by https://docs.mdb.ai/docs/api/models such as "mixtral-8x7b" or "gemini-1.5-pro"; setting to "gpt-3.5-turbo" as default
