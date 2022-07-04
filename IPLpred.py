import pandas as pd


def predictInnings(inningFile, predictionFile, modelFile=''):
    innings = pd.read_csv(inningFile)
    if (modelFile != ''):
        model = pd.read_pickle(modelFile)

    innings.drop(columns=['batsman', 'non_striker', 'bowler', 'fielder', 'player_dismissed'], axis=1, inplace=True)

    innings['batting_team'].replace('Rising Pune Supergiants', 'Rising Pune Supergiant', inplace=True)
    innings['bowling_team'].replace('Rising Pune Supergiants', 'Rising Pune Supergiant', inplace=True)

    innings['dismissal_kind'].fillna(value=0, inplace=True)
    innings['wicket'] = (innings['dismissal_kind'] != 0).astype(int)
    innings['wicket'] = innings.groupby('match_id')['wicket'].cumsum()
    innings['balls_done'] = 6 * (innings['over'] - 1) + innings['ball']
    innings.drop(columns=['dismissal_kind', 'over', 'ball'], inplace=True)
    innings.set_index('match_id', inplace=True)
    innings.drop(
        columns=['wide_runs', 'bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs', 'batsman_runs', 'extra_runs'],
        inplace=True)
    innings['total_runs'] = innings.groupby('match_id')['total_runs'].cumsum()

    innings = pd.get_dummies(innings, columns=['batting_team', 'bowling_team'])

    innings['batting_team_Kochi Tuskers Kerala'] = 0

    predictions = model.predict(innings)
    ans = pd.DataFrame({'match_id': innings.index, 'prediction': predictions})
    ans = ans.groupby('match_id')['prediction'].agg('mean').reset_index()
    ans['prediction'] = ans['prediction'].round(0)

    ans.to_csv(predictionFile, index=False)

predictInnings('IPL_test.csv', 'pred.csv', 'IPL.pkl')