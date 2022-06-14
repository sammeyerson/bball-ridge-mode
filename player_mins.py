import csv
from datetime import date, datetime
import pandas as pd
import numpy as np
import lxml.html as lh
import urllib.request
import requests
from unidecode import unidecode


try:
    from constants import TEAM_TO_TEAM_ABBR
except:
    print('Abbreviations not found')

def get_boxscore_urls(df, urls = None):
    #df = pd.read_csv('nbaData.csv')
    if(urls == None):
        urls = []
    url_base = 'https://www.basketball-reference.com/boxscores/'
    for row in df.values:
        date = row[1].replace('\\','').replace('-','')
        home_team = row[2].upper()
        home_team = [team for team in TEAM_TO_TEAM_ABBR if home_team in team][0]
        home_team = TEAM_TO_TEAM_ABBR.get(home_team)
        urls.append(url_base + date + '0' + home_team + '.html')
    return urls

def get_player_mins(df):
    urls = get_boxscore_urls(df)
    for url in urls:
        page = requests.get(url)
        doc = lh.fromstring(page.content)
        rows = doc.xpath("//table/tbody")[0]
        teams = doc.xpath("//h1")[0].text_content().split(" at ")

        away_team = teams[0].upper()
        away_team = TEAM_TO_TEAM_ABBR.get(away_team)
        home_team = teams[1].split(" Box ")[0].upper()
        home_team = TEAM_TO_TEAM_ABBR.get(home_team)
        teams = [home_team, away_team]

        file_name = 'teams/'+ away_team[len(away_team)-3:] + '.csv'
        team_file = pd.read_csv(file_name, index_col=0)

        row_num = team_file.shape[0]
        date = url.split("/")[-1][0:8]
        date = date[:4] + "-" + date[4:6] + "-"+date[6:]
        team_file.at[row_num, 'Date'] = date
        for row in rows:
            player = unidecode(row[0].text_content())
            mins_played = row[1].text_content()
            if(player!="Reserves"):
                if "Did" in mins_played or "Not" in mins_played or "Suspended" in mins_played:
                    #pretty sure these are all cases of player out on bballref^
                    mins_played = "0"
                else:
                    mins_played = mins_played.split(":")
                    seconds = float(mins_played[1])/60
                    mins_played = round(float(mins_played[0])+seconds, 2)
                team_file.at[row_num,player] = mins_played
        team_file = team_file.fillna(0)
        team_file.to_csv(file_name)

        rows = doc.xpath("//table/tbody")[-1]

        file_name = 'teams/'+ home_team[len(home_team)-3:] + '.csv'
        team_file = pd.read_csv(file_name, index_col=0)
        row_num = team_file.shape[0]
        date = url.split("/")[-1][0:8]
        date = date[:4] + "-" + date[4:6] + "-"+date[6:]
        team_file.at[row_num, 'Date'] = date
        for row in rows:
            player = unidecode(row[0].text_content())
            mins_played = row[1].text_content()
            if(player!="Reserves"):
                if "Did" in mins_played or "Not" in mins_played or "Suspended" in mins_played:
                    #pretty sure these are all cases of player out on bballref^
                    mins_played = "0"
                else:
                    mins_played = mins_played.split(":")
                    seconds = float(mins_played[1])/60
                    mins_played = round(float(mins_played[0])+seconds, 2)
                team_file.at[row_num,player] = mins_played
        team_file = team_file.fillna(0)
        team_file.to_csv(file_name)
    return 0

def get_team_rosters():
    #get all team rosters to csv file for each team
    team_abbreviations = list(TEAM_TO_TEAM_ABBR.values())


    year = '2022'

    for team in team_abbreviations:
        fileName = "teams/"+ team +".csv"
        url ='https://www.basketball-reference.com/teams/'+team+'/' + year + '.html'
        page = requests.get(url)
        doc = lh.fromstring(page.content)
        rows = doc.xpath("//table/tbody")[1]
        columns = ['','Date']
        for row in rows:
            player = unidecode(row[1].text_content())
            columns.append(player)
        with open(fileName, 'w') as file:
            wr = csv.writer(file)
            wr.writerow(columns)
    return 0

def main():
    df = pd.read_csv('nbaData.csv')
    get_team_rosters()
    get_player_mins(df)
    return 0


if __name__ == "__main__":
    main()
