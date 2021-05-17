import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

slack_token = os.getenv('SLACK_TOKEN')


def post_message_to_slack(text):
    return requests.post(f'https://hooks.slack.com/services/{slack_token}', json.dumps({
        'text': text,
    }))
