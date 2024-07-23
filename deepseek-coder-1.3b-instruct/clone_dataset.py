from huggingface_hub import login
from huggingface_hub import Repository

login("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",add_to_git_credential=True)

repo_url='https://huggingface.co/datasets/CodexAI/Test4Deepseek-coder'

repo = Repository(local_dir="data", clone_from=repo_url)