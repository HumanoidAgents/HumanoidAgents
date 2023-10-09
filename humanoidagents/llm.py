import openai
from functools import cache

class OpenAILLM:
    @staticmethod
    def get_llm_response(prompt, max_tokens=None, timeout=60):
        n_retries = 10
        for i in range(n_retries):
            try:
                if isinstance(max_tokens, int):
                    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, request_timeout=timeout)
                else:
                    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], request_timeout=timeout)
                return chat_completion.choices[0].message.content
            except openai.error.ServiceUnavailableError:
                print("openai.error.ServiceUnavailableError")
                pass
            except openai.error.Timeout:
                print("openai.error.Timeout")
                pass
            except openai.error.APIError:
                print("openai.error.APIError")
                pass
            except openai.error.APIConnectionError:
                pass
            except openai.error.RateLimitError:
                print("openai.error.RateLimitError")
                time.sleep(10)
            # too many tokens
            except openai.error.InvalidRequestError:
                context_window = 3000 * 4 # max length in chars (every token is around 4 chars)
                prompt = prompt[:context_window]
                print("openai.error.InvalidRequestError")

        raise ValueError(f"OpenAI remains uncontactable after {n_retries} retries due to either Server Error or Timeout after {timeout} seconds")

    @cache
    @staticmethod
    def get_embeddings(query):
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']
        return embeddings
    