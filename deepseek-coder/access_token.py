from dotenv import load_dotenv
import requests
import os
from loggings import loggers

def validate_token():
    '''
    This function is used for validation HuggingFace access token, it's a good idea
    to validate the access token before clonning and pushing to huggingface hub.
    '''
    log = loggers('logs/token.log')

    try:
        load_dotenv()
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        try:
            response = requests.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {huggingface_token}"}
            )
            if response.status_code == 200:
                # print(f"Token is valid. Username: {response.json()['name']}")
                log.info(f"Hugging Face token is valid. Username: {response.json()['name']}")
                return huggingface_token
            else:
                print(f"Token [{huggingface_token}] is not valid.")
                log.warning(f"Hugging Face token [{huggingface_token}] is not valid. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Failed to validate Token [{huggingface_token}]. Error: {e}")
            log.error(f"Failed to validate Hugging Face token [{huggingface_token}]. Error: {e}")

    except Exception as e:
        print(f"Failed to load environment variables. Error: {e}")

# validate_token()