from huggingface_hub import login
from huggingface_hub import Repository
from loggings import loggers
from access_token import validate_token

log = loggers('logs/repo.log')

try:
    TOKEN = validate_token()

    login(TOKEN,add_to_git_credential=True)

    repo_url='https://huggingface.co/datasets/CodexAI/Test4Deepseek-coder'
    repo = Repository(local_dir="data", clone_from=repo_url)
    if repo:
        log.info(f"Cloning Repository successfully from {repo_url}")

except:
    print("Cloning Repository Failed!")
    log.error(f"Cloning Repository failed from {repo_url}")