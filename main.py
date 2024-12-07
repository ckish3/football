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

def create_document(match: dict) -> str:

    home_goals = match['score']['fullTime']['homeTeam']
    away_goals = match['score']['fullTime']['awayTeam']

    home_team_name = match['homeTeam']['name']
    away_team_name = match['awayTeam']['name']

    document = ''
    document += match['competition']['name'] +'/n'
    document += 'In ' + match['area']['name'] + ','
    document += ' in the ' + match['season']['startDate'][0:4] + '-' + match['season']['endDate'][0:4] + ' season,'
    document += ' occurring on ' + match['utcDate'][0:10] 
    document += ' for matchday ' + str(match['matchday']) + ','
    document += ' and played at ' + home_team_name + '. '
    document += 'The match was between ' + home_team_name + ' vs ' + away_team_name + '. '
    document += 'The score was ' + home_goals + ' - ' + away_goals + ', '
    document += 'where ' + home_team_name + ' scored ' + home_goals + ' goals and ' + away_team_name + ' scored ' + away_goals + ' goals. '

    if home_goals > away_goals:
        document += home_team_name + ' won the match.'
    elif home_goals < away_goals:
        document += away_team_name + ' won the match.'
    else:
        document += 'The match was a draw.' #Currently, this assumes that there aren't tie-brakers like penalty kicks

    return document

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