from main import create_document, embed_data
import pytest
from unittest.mock import patch
import json

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

    assert output == expected

def test_embed_data_success():
    # Test data
    match_documents = {
        "1": "This is a test document",
        "2": "Another test document"
    }
    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    # Mock the API response
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = mock_embeddings
        mock_post.return_value.status_code = 200
        
        # Call the function
        result = embed_data(match_documents, "dummy_token")
        
        # Verify the API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "Bearer dummy_token" in call_args[1]['headers']['Authorization']
        assert json.loads(json.dumps(call_args[1]['json'])) == {
            "inputs": ["This is a test document", "Another test document"],
            "options": {"wait_for_model": True}
        }
        
        # Verify the results
        assert result == {"1": mock_embeddings[0], "2": mock_embeddings[1]}

def test_embed_data_api_error():
    match_documents = {"1": "Test document"}
    
    # Mock API error response
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 401
        mock_post.return_value.raise_for_status.side_effect = Exception("API Error")
        
        # Verify the function handles the error
        with pytest.raises(Exception):
            embed_data(match_documents, "invalid_token")