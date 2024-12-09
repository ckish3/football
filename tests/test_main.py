from main import create_document, embed_data, retrieve_api_data, save_api_data

#TODO: Not all code has tests written for it yet

import pytest
from unittest.mock import patch, mock_open
import json
import datetime
import os

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


def test_retrieve_api_data_from_cache():
    # Mock data for testing
    mock_matches = [
        {"id": 1, "homeTeam": {"name": "Team A"}, "awayTeam": {"name": "Team B"}},
        {"id": 2, "homeTeam": {"name": "Team C"}, "awayTeam": {"name": "Team D"}}
    ]
    
    # Mock the file open and json load operations
    mock_file = mock_open(read_data=json.dumps(mock_matches))
    with patch('builtins.open', mock_file):
        result = retrieve_api_data()
        
        # Verify results
        assert len(result) == 2
        assert result[0]['id'] == 1
        assert result[1]['id'] == 2
        mock_file.assert_called_once_with('data.json', 'r')

def test_retrieve_api_data_from_api():
    # Mock data for API responses
    mock_matches_1 = {"matches": [{"id": 1}, {"id": 2}]}
    
    # Mock response for API call
    mock_response = type('MockResponse', (), {
        'json': lambda self: mock_matches_1,
        'text': 'Success'
    })()
    
    # Mock date to control the loop
    mock_date = type('MockDate', (), {
        'fromisoformat': lambda x: type('MockDateObj', (), {
            'isoformat': lambda: x,
            '__sub__': lambda self, other: datetime.timedelta(days=10),
            '__le__': lambda self, other: False  # Exit the loop after first iteration
        })()
    })()
    
    # Set up the environment variable
    with patch.dict(os.environ, {'FOOTBALL_DATA_TOKEN': 'test_token'}):
        # Mock file operations to fail on read but succeed on write
        mock_file = mock_open()
        with patch('builtins.open', mock_file) as mock_file_open:
            # Make first open call raise FileNotFoundError
            mock_file_open.side_effect = [FileNotFoundError, mock_open().return_value]
            
            # Mock datetime.date
            with patch('datetime.date', mock_date):
                # Mock the API calls
                with patch('requests.get') as mock_get:
                    mock_get.return_value = mock_response
                    
                    # Mock sleep to speed up tests
                    with patch('time.sleep'):
                        result = retrieve_api_data('2024-01-01', '2024-01-10')
                        
                        # Verify results
                        assert len(result) == 2
                        assert result[0]['id'] == 1
                        assert result[1]['id'] == 2
                        
                        # Verify API was called with correct parameters
                        mock_get.assert_called()
                        call_args = mock_get.call_args
                        assert 'test_token' in call_args[1]['headers']['X-Auth-Token']
                        
                        # Verify data was saved to cache
                        mock_file_open.assert_called_with('data.json', 'w')

def test_retrieve_api_data_error():
    # Mock API error response that will fail json parsing
    error_response = type('MockResponse', (), {
        'json': lambda self: {"matches": []},  # Return empty matches to trigger the exception
        'text': 'API Error'
    })()
    
    # Mock date to control the loop
    mock_date = type('MockDate', (), {
        'fromisoformat': lambda x: type('MockDateObj', (), {
            'isoformat': lambda: x,
            '__sub__': lambda self, other: datetime.timedelta(days=10),
            '__le__': lambda self, other: True  # Keep the loop going
        })()
    })()
    
    # Set up the environment variable
    with patch.dict(os.environ, {'FOOTBALL_DATA_TOKEN': 'test_token'}):
        # Mock file operations to fail
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = FileNotFoundError
            
            # Mock datetime.date
            with patch('datetime.date', mock_date):
                # Mock the API call to return an error
                with patch('requests.get') as mock_get:
                    mock_get.return_value = error_response
                    
                    # Mock sleep to speed up tests
                    with patch('time.sleep'):
                        # Verify the function raises an exception
                        with pytest.raises(Exception) as exc_info:
                            retrieve_api_data('2024-01-01', '2024-01-10')
                        
                        # The error message should contain our API error text
                        assert "Could not retrieve data from the API" in str(exc_info.value)

def test_save_api_data():
    # Test data - two matches with different properties
    test_matches = [
        {
            'area': {'name': 'England'},
            'competition': {'name': 'Premier League'},
            'season': {'startDate': '2024-08-18', 'endDate': '2025-05-18'},
            'utcDate': '2024-11-01',
            'matchday': 10,
            'homeTeam': {'name': 'Arsenal'},
            'awayTeam': {'name': 'Chelsea'},
            'score': {
                'winner': 'HOME_TEAM',
                'fullTime': {'home': 2, 'away': 0}
            }
        },
        {
            'area': {'name': 'Spain'},
            'competition': {'name': 'La Liga'},
            'season': {'startDate': '2024-08-18', 'endDate': '2025-05-18'},
            'utcDate': '2024-11-02',
            'matchday': 12,
            'homeTeam': {'name': 'Barcelona'},
            'awayTeam': {'name': 'Real Madrid'},
            'score': {
                'winner': 'AWAY_TEAM',
                'fullTime': {'home': 1, 'away': 3}
            }
        }
    ]
    
    # Call the function
    result = save_api_data(test_matches)
    
    # Verify the results
    assert len(result) == 2
    
    # Check that IDs are assigned sequentially
    assert 0 in result
    assert 1 in result
    
    # Verify first match document content
    assert 'Premier League' in result[0]
    assert 'England' in result[0]
    assert 'Arsenal' in result[0]
    assert 'Chelsea' in result[0]
    assert '2 - 0' in result[0]
    assert 'Arsenal won' in result[0]
    
    # Verify second match document content
    assert 'La Liga' in result[1]
    assert 'Spain' in result[1]
    assert 'Barcelona' in result[1]
    assert 'Real Madrid' in result[1]
    assert '1 - 3' in result[1]
    assert 'Real Madrid won' in result[1]

def test_save_api_data_empty():
    # Test with empty list
    result = save_api_data([])
    
    # Verify empty result
    assert len(result) == 0
    assert isinstance(result, dict)