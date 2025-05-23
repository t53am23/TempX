!pip install uv
!uv run --prerelease allow --with git+https://github.com/attentionmech/smolbox smolbox inspect/mav --model_path meta-llama/Llama-3.2-1B --max_new_tokens 30 --prompt "If the current date in yyyy-mm-dd format is 2002-11-19 and I was born on 1999-05-18, what is my age in days in this format %d-%b-%Y?.  Give the translated answer of each without using Python. The answer is:"
