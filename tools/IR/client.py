import requests
import json


def query(query, kth):
    data = {'query': query, 'kth': kth}
    response = requests.get(
        'IR URL', params=data)
    return json.loads(response.content)

