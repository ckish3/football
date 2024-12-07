import os
import requests
import json
from typing import List

def retrieve_api_data(date_from: str, date_to: str) -> List[dict]:
    """ 
    Retrieve data from the football data api

    Args:
        date_from (str): The date from which to retrieve data
        date_to (str): The date to which to retrieve data

    Returns:
        List[dict]: A list of matches
    """
    headers = { 'X-Auth-Token': os.environ['FOOTBALL_DATA_TOKEN'] }
    
    uri = 'https://api.football-data.org/v4/matches'
    if date_from is not None and date_to is not None:
        uri += '?dateFrom=' + date_from + '&dateTo=' + date_to
    response = requests.get(uri, headers=headers)

    matches = response.json()['matches']

    return matches

def save_api_data():
    pass

def embed_data():
    pass

def save_embeddings():
    pass

def search_embeddings():
    pass

def query_llm():
    pass

def main():
    retrieve_api_data()
    save_api_data()
    embed_data()
    save_embeddings()
    search_embeddings()
    query_llm()