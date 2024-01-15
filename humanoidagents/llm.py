import openai
from openai import OpenAI

from functools import cache



class OpenAILLM:

    client = OpenAI()
    
    @classmethod
    def get_llm_response(cls, prompt, max_tokens=None, timeout=60):
        n_retries = 10
        for i in range(n_retries):
            try:
                if isinstance(max_tokens, int):
                    chat_completion = cls.client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, timeout=timeout)
                else:
                    chat_completion = cls.client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], timeout=timeout)
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