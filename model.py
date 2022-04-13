from operator import mul
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from datetime import date, datetime
from tqdm import tqdm

try:
    from constants import TEAM_TO_TEAM_ABBR
except:
    print('Abbreviations not found')


def splitForPlayer(df, old_df, player = 'Kyrie Irving', team = 'Brooklyn'):

    team_abbrev = [value for key, value in TEAM_TO_TEAM_ABBR.items() if team.lower() in key.lower()][0]
    player_mins = pd.read_csv('teams_mins/'+team_abbrev+'.csv')
    game_count = 0
    dates_played_in = []
    itt = 0
    for player_name in player_mins.columns:
        if player.lower() in player_name.lower():
            break
        itt += 1

    for row in player_mins.values:
        player_mins_game = row[itt]
        date = row[1]

        if player_mins_game > 0:
            dates_played_in.append(date)
            game_count +=1

    team_name = team
    team_data = df[[team_name, 'point_difference']]
    df = df.drop([team_name], axis=1)
    ind = 0
    with_player = []
    without_player = []
    for row in team_data.values:
        played = row[0]
        point_diff = row[1]
        old_df_row = old_df.iloc[ind]
        date_game = old_df_row['Date']
        if date_game in dates_played_in:
            with_player.append(played)
            without_player.append(0)
        else:
            with_player.append(0)
            without_player.append(played)

        ind += 1
    columnn_name_with = team+  ' (w/ ' + player + ')'
    columnn_name_without = team + ' (w/o ' + player + ')'
    df[columnn_name_with] = with_player
    df[columnn_name_without] = without_player
    return df

def model_data_matrix(df):

    df_visitor = pd.get_dummies(df['Away Team'], dtype=np.int64)
    df_home = pd.get_dummies(df['Home Team'], dtype=np.int64)
    df_model_data = df_home.sub(df_visitor)
    teams = df['Home Team'].unique()
    data = {
        'Teams': teams
    }

    df_model_data['point_difference'] = df['Point Differential']

    player = 'Bam Adebayo'
    team = 'Miami'
    df_model_data = splitForPlayer(df_model_data, df, player, team)
    #^select a player and team to split that team into game with/without the player
    #this will measure how much better/worse the team is with/without the player

    return df_model_data

def ridge_model(df):
    lr = Ridge(alpha=0.001)
    X = df.drop(['point_difference'], axis=1)
    y = df['point_difference']
    lr.fit(X, y)
    df_ratings = pd.DataFrame(data={'team': X.columns, 'rating': lr.coef_})
    df_ratings_sorted = df_ratings.sort_values('rating', ascending=False)
    return df_ratings_sorted


def cross_validation2(df_test, df_model_data, edge_threshold, df):
    model_ats_accuracy = []
    model_wl_accuracy = []
    test_count = 0
    df_test_point_diff = df_test['point_difference']
    df_test = df_test.drop(['point_difference'], axis=1)
    columns = list(df_test)
    for index, row in df_test.iterrows():
        home_team = ""
        away_team = ""
        df_with_spread = df.loc[df['Game Number'] == index+1]
        for column in columns:
            cell = row[column]
            if (cell==1):
                home_team = column
            elif (cell==-1):
                away_team = column
        home_team_rating = df_model_data.loc[df_model_data['team'] == home_team].iloc[0]['rating']
        away_team_rating = df_model_data.loc[df_model_data['team'] == away_team].iloc[0]['rating']
        home_team_implied_spread = (float(home_team_rating) - float(away_team_rating)) * -1
        home_spread = df_with_spread['Home Spread'].iloc[0]
        home_covered = df_with_spread['Home Covered'].iloc[0]
        home_wL = df_with_spread['Home W/L'].iloc[0]
        bet_home_team_to_cover = False

        if home_spread < 0:
            if (home_team_implied_spread < home_spread):
                    bet_home_team_to_cover = True
                    #if the model thinks a team should be favored by more than they really are,
                    #bet them


            edge = (home_team_implied_spread - home_spread) * -1.0
            #edge that model finds over actual spread
            if edge > edge_threshold:
                #only placing bets where the model has an edge over the real line
                test_count +=1
                if home_covered == 'Y' and bet_home_team_to_cover == True:
                    model_ats_accuracy.append(True)
                elif home_covered == 'N' and bet_home_team_to_cover == False:
                    model_ats_accuracy.append(True)
                elif home_covered == 'p':
                    #dumby situation to not count pushes against accuracy
                    test_count = test_count
                else:
                    model_ats_accuracy.append(False)

                if home_wL == 'W' and home_team_implied_spread <= 0:
                    model_wl_accuracy.append(True)
                elif home_wL == 'L' and home_team_implied_spread >= 0:
                    model_wl_accuracy.append(True)
                else:
                    model_wl_accuracy.append(False)
        else:
            if (home_team_implied_spread < home_spread):
                    bet_home_team_to_cover = True
                    #if the model thinks a team should be bigger underdogs than they really are,
                    #bet them
            edge = (home_team_implied_spread - home_spread)
            if edge > edge_threshold:
                #only placing bets where the model has an edge over the real line
                test_count +=1
                if home_covered == 'N' and bet_home_team_to_cover == False:
                    model_ats_accuracy.append(True)
                elif home_covered == 'Y' and bet_home_team_to_cover == True:
                    model_ats_accuracy.append(True)
                elif home_covered == 'p':
                    #dumby situation to not count pushes against accuracy
                    test_count = test_count
                else:
                    model_ats_accuracy.append(False)
                if home_wL == 'W' and home_team_implied_spread <= 0:
                    model_wl_accuracy.append(True)
                elif home_wL == 'L' and home_team_implied_spread >= 0:
                    model_wl_accuracy.append(True)
                else:
                    model_wl_accuracy.append(False)

    number_correct = model_ats_accuracy.count(True)
    number_incorrect = model_ats_accuracy.count(False)
    percent_correct = number_correct / (number_correct + number_incorrect)
    percent_correct_ats = percent_correct

    number_correct = model_wl_accuracy.count(True)
    number_incorrect = model_wl_accuracy.count(False)
    percent_correct = number_correct / (number_correct + number_incorrect)

    return percent_correct_ats

def test_model(df, itt, edge):
    run_count = 0
    total_perc_correct = 0

    for x in range(0,itt):
        #run itt itterations of each model
        df['Point Differential'] = df['Home Score'] - df['Away Score']

        df_model_data = model_data_matrix(df)
        #df with 1's for home teams, -1's for away teams, and point differential
        df_train, df_test = train_test_split(df_model_data, test_size=0.25)

        df_duplicate_rows = pd.DataFrame()
        #to append duplicate rows for weighing more recent games more heavily
        row_num = df_train.shape[0]
        for index, row in df_train.iterrows():
            if(index > (row_num*0.8)):
                df_duplicate_rows= df_duplicate_rows.append([row]*7)
            elif (index > (row_num*0.6) and index <= (row_num*0.8)):
                #*7
                df_duplicate_rows= df_duplicate_rows.append([row]*5)
            elif (index > (row_num*0.4) and index <= (row_num*0.6)):
                df_duplicate_rows= df_duplicate_rows.append([row]*3)
                #*5
            elif (index > (row_num*0.2) and index <= (row_num*0.4)):
                df_duplicate_rows= df_duplicate_rows.append([row]*2)
                #*3

        df_train = df_train.append(df_duplicate_rows)
        df_sorted = ridge_model(df_train).reset_index(drop=True)
        edge = 1

        ats_correct = cross_validation2(df_test, df_sorted, edge, df)
        total_perc_correct+=ats_correct
        run_count+=1

    avg_perc_correct = total_perc_correct/run_count
    return avg_perc_correct

def model(df):

    df['Point Differential'] = df['Home Score'] - df['Away Score']

    df_model_data = model_data_matrix(df)
    #dataframe with 1's for home teams, -1's for away teams, and point differential
    df_duplicate_rows = pd.DataFrame()
    #duplicate more recent games for weighing them more heavily

    row_num = df_model_data.shape[0]

    for index, row in df_model_data.iterrows():

        if(index > (row_num*0.8)):
            df_duplicate_rows= df_duplicate_rows.append([row]*7)
        elif (index > (row_num*0.6) and index <= (row_num*0.8)):
            #*7
            df_duplicate_rows= df_duplicate_rows.append([row]*5)
        elif (index > (row_num*0.4) and index <= (row_num*0.6)):
            df_duplicate_rows= df_duplicate_rows.append([row]*3)
            #*5
        elif (index > (row_num*0.2) and index <= (row_num*0.4)):
            df_duplicate_rows= df_duplicate_rows.append([row]*2)
            #*3

    df_model_data = df_model_data.append(df_duplicate_rows)
    df_sorted = ridge_model(df_model_data).reset_index(drop=True)
    return df_sorted

def main():
    file = 'nbaData.csv'
    df = pd.read_csv(file)
    test_num = 10
    test_edge = 1
    df_test = test_model(df, test_num, test_edge)
    print("Test average accuracy based on {} test runs: {}".format(test_num, df_test))
    df_model_results = model(df)
    print("Model: {}".format(df_model_results))
    #df_model_results holds coefficients representing how much better a team is than league average
    #for example: if Boston is 9.5 and Detroit is -5.5 then the model thinks the spread should be Boston -15
    #obviously many factors go into a spread and more hyperparameters will be added over time to finetune


if __name__ == "__main__":
    main()
