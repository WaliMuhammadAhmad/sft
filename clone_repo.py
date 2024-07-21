from huggingface_hub import login
from huggingface_hub import Repository

login("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",add_to_git_credential=True)

repo_id='CodexAI/Trainllama'
repo_url='https://huggingface.co/datasets/CodexAI/Train4llama'

repo = Repository(local_dir="eval", clone_from=repo_url)