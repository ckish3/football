import os
import requests
import json
from typing import List
import re
import datetime
import time
import logging

import voyager
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retrieve_api_data(date_from: str = None, date_to: str = None) -> List[dict]:
    """
    Retrieve data from the football data api

    Args:
        date_from (str): The date from which to retrieve data
        date_to (str): The date to which to retrieve data

    Returns:
        List[dict]: A list of matches
    """

    try:
        with open('data.json', 'r') as f:
            all_matches = json.load(f)
            logger.info('Retrieved API data from file')
    except:
        logger.info('Retrieving data from the football data api')
        
        headers = { 'X-Auth-Token': os.environ['FOOTBALL_DATA_TOKEN'] }
        
        uri = 'https://api.football-data.org/v4/matches'

        all_matches = []
        number_of_calls = 0

        if date_from is None:
            date_from = '2022-11-01'
        if date_to is None:
            date_to = datetime.date.today().isoformat()

        end_date = date_to
        while datetime.date.fromisoformat(date_from) <= datetime.date.fromisoformat(end_date) - datetime.timedelta(days=10): #start_date <= end_date:
            date_to = (datetime.date.fromisoformat(date_from) + datetime.timedelta(days=10)).isoformat()
            uri_dated = uri + '?dateFrom=' + date_from + '&dateTo=' + date_to

            #print(uri_dated)
            response = requests.get(uri_dated, headers=headers)
            number_of_calls += 1

            try:
                matches = response.json()['matches']

                all_matches += matches        
            except:
                print(f'Number of API calls: {number_of_calls}')
                raise Exception(f'Could not retrieve data from the API: \n{response.text}')

            date_from = (datetime.date.fromisoformat(date_from) + datetime.timedelta(days=10)).isoformat()
            time.sleep(6) #To avoid hitting the API rate limit

        with open('data.json', 'w') as f:
            json.dump(all_matches, f)
    return all_matches

def create_document(match: dict) -> str:
    """ 
    Converts a match (a dict) response from the APIto a document that is more comprehensible
    and should be easier to match against likely questions

    Args:
        match (dict): A match response from the API

    Returns:
        str: A document
    """
    home_goals = str(match['score']['fullTime']['home'])
    away_goals = str(match['score']['fullTime']['away'])

    home_team_name = match['homeTeam']['name']
    away_team_name = match['awayTeam']['name']

    document = ''
    document += match['competition']['name'] + '. '
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

def save_api_data(all_matches: List[dict]) -> dict:
    """ 
    Save the data to a file

    Args:
        all_matches (List[dict]): A list of matches

    Returns:
        dict: The match documents, where the key is the ID and the value is the document
    """
    id = 0
    match_documents = {}
    for match in all_matches:
        match_documents[id] = create_document(match)
        id += 1
        
    return match_documents


def embed_data(match_documents: dict, hf_token: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> dict:
    """ 
    Embeds the match documents using a Hugging Face model

    Args:
        match_documents (dict): The match documents, where the key is the ID and the value is the document
        hf_token (str): The Hugging Face token
        model_name (str): The name of the Hugging Face model name to use for embedding

    Returns:
        dict: The embeddings, where the key is the ID and the value is the embedding
    """

    embeddings = {}

    texts = []
    ids = []
    for id in match_documents:
        texts.append(match_documents[id])
        ids.append(id)

    response_data = embed_texts(texts, hf_token, model_name)
    
    for i in range(len(ids)):
        embeddings[ids[i]] = response_data[i]

    return embeddings


def embed_texts(texts: List[str], hf_token: str, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> List[list]:
    """ 
    Embeds a list of texts using a Hugging Face model

    Args:
        texts (List[str]): A list of texts
        hf_token (str): The Hugging Face token
        model_name (str): The name of the Hugging Face model name to use for embedding

    Returns:
        List[list]: A list of embeddings, one for each string in texts
    """
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    response.raise_for_status()  # This will raise an exception for error status codes
    
    response_data = response.json()

    return response_data

    
def save_embeddings(embeddings_dict: dict) -> voyager.Index:
    """ 
    Save the embeddings to a file

    Args:
        embeddings_dict (dict): The embeddings, where the key is the ID and the value is the embedding

    Returns:
        voyager.Index: The search space of embedded documents
    """
    for id, embedding in embeddings_dict.items():
        num_dimensions = len(embedding)
        break

    index = voyager.Index(voyager.Space.Cosine, num_dimensions=num_dimensions)
    for id, embedding in embeddings_dict.items():
        index.add_item(embedding, id)

    return index


def embed_query(query: str, hf_token: str, model_name: str) -> list:
    texts = [query]
    return embed_texts(texts, hf_token, model_name)[0]


def search_embeddings(query_embedding: list, index: voyager.Index, number_of_results: int) -> List[int]:
    """ 
    Search the embeddings for the most similar documents

    Args:
        query_embedding (list): The embedding of the query
        index (voyager.Index): The search space of embedded documents
        number_of_results (int): The number of results to return

    Returns:
        List[int]: The IDs of the most similar documents (matche reports)
    """
    
    neighbor_ids, distances = index.query(query_embedding, k=number_of_results)

    return neighbor_ids


def create_document_context(match_documents: dict, ids: List[int]) -> str:
    """ 
    Creates a context from a list of match documents

    Args:
        match_documents (dict): The match documents, where the key is the ID and the value is the document
        ids (List[int]): A list of IDs of match documents to include in the context

    Returns:
        str: The context
    """

    context = ''
    for id in ids:
        context += match_documents[id] + '\n'

    return context


def query_llm(query: str, context: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: str, model_name: str) -> str:
    """ 
    Queries the LLM to answer the query

    Args:
        query (str): The question the user posed to the LLM
        context (str): The context (the match reports) to get the answer from
        tokenizer (AutoTokenizer): The tokenizer
        model (AutoModelForCausalLM): The model
        device (str): The device to use for the model ("cpu" or "cuda")
        model_name (str): The name of the model

    Returns:
        str: The answer to the query
    """

    prompt = f"""Context: {context}\n\n
    Now here is the question you need to answer. 
    Question: {query}"""

    messages=[
            {"role": "system", "content": """Using the information contained in the context, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
If the answer cannot be deduced from the context, do not give an answer."""},
            {"role": "user", "content": prompt}
        ]

    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    return parse_response(tokenizer.decode(outputs[0]), model_name)


def parse_response(text: str, llm_model_name: str) -> str:
    """Parses a response from the model

    Args:
        text: Response from the model.

    Returns:
        str: The model's response
    """
    if llm_model_name == "HuggingFaceTB/SmolLM2-1.7B-Instruct":
        pattern = r"<|im_start|>assistant\n(.*?)<|im_end|>"
    else:
        raise Exception(f'Parsing for model "{llm_model_name}" not implemented yet')

    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        matches = [m for m in matches if len(m) > 0]
        if len(matches) > 0:
            answer = matches[0]
        else:
            raise Exception(f'Could not parse the response: \n{text}\n for model "{llm_model_name}"')
    else:
        raise Exception(f'Could not parse the response: \n{text}\n for model "{llm_model_name}"')
    return answer



def test_create_document():
    input = {'area': {'id': 2081, 'name': 'France', 'code': 'FRA', 'flag': 'https://crests.football-data.org/773.svg'}, 'competition': {'id': 2015, 'name': 'Ligue 1', 'code': 'FL1', 'type': 'LEAGUE', 'emblem': 'https://crests.football-data.org/FL1.png'}, 'season': {'id': 2290, 'startDate': '2024-08-18', 'endDate': '2025-05-18', 'currentMatchday': 14, 'winner': None}, 'id': 498034, 'utcDate': '2024-11-01T18:00:00Z', 'status': 'FINISHED', 'matchday': 10, 'stage': 'REGULAR_SEASON', 'group': None, 'lastUpdated': '2024-12-07T00:20:53Z', 'homeTeam': {'id': 548, 'name': 'AS Monaco FC', 'shortName': 'Monaco', 'tla': 'ASM', 'crest': 'https://crests.football-data.org/548.png'}, 'awayTeam': {'id': 532, 'name': 'Angers SCO', 'shortName': 'Angers SCO', 'tla': 'ANG', 'crest': 'https://crests.football-data.org/532.png'}, 'score': {'winner': 'AWAY_TEAM', 'duration': 'REGULAR', 'fullTime': {'home': 0, 'away': 1}, 'halfTime': {'home': 0, 'away': 1}}, 'odds': {'msg': 'Activate Odds-Package in User-Panel to retrieve odds.'}, 'referees': [{'id': 57501, 'name': 'Abdelatif Kherradji', 'type': 'REFEREE', 'nationality': 'France'}]}

    expected = ''
    expected += 'Ligue 1. '
    expected += 'In France,'
    expected += ' in the 2024-2025 season,'
    expected += ' occurring on 2024-11-01 '
    expected += 'for matchday 10,'
    expected += ' and played at AS Monaco FC. '
    expected += 'The match was between AS Monaco FC vs Angers SCO. '
    expected += 'The score was 0 - 1, '
    expected += 'where AS Monaco FC scored 0 goals and Angers SCO scored 1 goals. '
    expected += 'Angers SCO won the match.'

    output = create_document(input)

    if output == expected:
        print('test_create_document passed')
    else:
        print('test_create_document failed')


def main():
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    number_of_search_results = 3

    checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    matches = retrieve_api_data()
    match_documents = save_api_data(matches)

    hf_token = os.environ['HF_WRITE_TOKEN']

    embeddings = embed_data(match_documents, hf_token, embedding_model_name)
    
    index = save_embeddings(embeddings)

    query = 'What was the score of the match between Monaco and Angers SCO in the 2024-2025 season?'

    query_embedding = embed_query(query, hf_token, embedding_model_name)
    search_results = search_embeddings(query_embedding, index, number_of_search_results)
    
    context = create_document_context(match_documents, search_results)
    logger.info('')
    logger.info('Retrieved documents:')
    logger.info(context)
    answer = query_llm(query, context, tokenizer, model, device, checkpoint)

    print(answer)

#    test_create_document()


if __name__ == '__main__':
    main()