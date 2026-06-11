import json
import os
import urllib.request


OPENAI_CHAT_COMPLETIONS_URL = 'https://api.openai.com/v1/chat/completions'


def openai_chat_completion(payload, timeout=60, api_key=None):
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY is not set')
    req = urllib.request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def openai_chat_text(payload, timeout=60, api_key=None):
    data = openai_chat_completion(payload, timeout=timeout, api_key=api_key)
    return (data['choices'][0]['message']['content'] or '').strip()


def openai_chat_json(payload, timeout=60, api_key=None):
    data = openai_chat_completion(payload, timeout=timeout, api_key=api_key)
    return json.loads(data['choices'][0]['message']['content'] or '{}')
