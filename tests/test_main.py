from main import create_document

def test_create_document():
    input = {'area': {'id': 2081, 'name': 'France', 'code': 'FRA', 'flag': 'https://crests.football-data.org/773.svg'}, 'competition': {'id': 2015, 'name': 'Ligue 1', 'code': 'FL1', 'type': 'LEAGUE', 'emblem': 'https://crests.football-data.org/FL1.png'}, 'season': {'id': 2290, 'startDate': '2024-08-18', 'endDate': '2025-05-18', 'currentMatchday': 14, 'winner': None}, 'id': 498034, 'utcDate': '2024-11-01T18:00:00Z', 'status': 'FINISHED', 'matchday': 10, 'stage': 'REGULAR_SEASON', 'group': None, 'lastUpdated': '2024-12-07T00:20:53Z', 'homeTeam': {'id': 548, 'name': 'AS Monaco FC', 'shortName': 'Monaco', 'tla': 'ASM', 'crest': 'https://crests.football-data.org/548.png'}, 'awayTeam': {'id': 532, 'name': 'Angers SCO', 'shortName': 'Angers SCO', 'tla': 'ANG', 'crest': 'https://crests.football-data.org/532.png'}, 'score': {'winner': 'AWAY_TEAM', 'duration': 'REGULAR', 'fullTime': {'home': 0, 'away': 1}, 'halfTime': {'home': 0, 'away': 1}}, 'odds': {'msg': 'Activate Odds-Package in User-Panel to retrieve odds.'}, 'referees': [{'id': 57501, 'name': 'Abdelatif Kherradji', 'type': 'REFEREE', 'nationality': 'France'}]}

    expected = ''
    expected += 'Ligue 1/n'
    expected += 'In France,'
    expected += ' in the 2024-2025 season,'
    expected += ' occurring on 2024-11-01 '
    expected += ' for matchday 10,'
    expected += ' and played at Monaco FC. '
    expected += 'The match was between Monaco FC vs Angers SCO. '
    expected += 'The score was 0 - 1, '
    expected += 'where Monaco FC scored 0 goals and Angers SCO scored 1 goals. '
    expected += 'Angers SCO won the match.'

    output = create_document(input)

    assert output == expected