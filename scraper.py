import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
#import necessary classes

years = list(range(2024, 2022, -1)) #Scrape the past 2 seasons 
all_games = []
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats" #url to start on


for year in years: 
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text,  features="lxml")
    standings = soup.find_all('table', class_ = 'stats_table')[0] #get the first table since it has the standings with the team links
    time.sleep(15) #Try to avoid getting blocked from scraping

    team_links = [l.get("href") for l in standings.find_all('a')]  #Get the href proptery from each link and 
    team_links = [l for l in team_links if '/squads/' in l]  #only want links with "squad" in it. filter out all other links
    team_urls = [f"https://fbref.com{l}" for l in team_links] #Add prefix to all the links to make the link absolute 

    #Get the links for the previous seasons
    prev_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{prev_season}"
    
    #Scrape game logs for each team
    for team in team_urls:
        team_name = team.split("/")[-1].replace("-Stats","").replace("-", " ") #Clean team name
        data = requests.get(team)

        matches_df = pd.read_html(data.text, match="Scores & Fixtures")[0] #Find Scores & Fixtures table

        shooting_soup = BeautifulSoup(data.text, features="lxml")
        shooting_links = [l.get("href") for l in shooting_soup.find_all('a')]
        shooting_links = [l for l in shooting_links if l and 'all_comps/shooting/' in l] #Get the shooting link from all the links on the page
        shooting_urls = requests.get(f"https://fbref.com{shooting_links[0]}") #Add prefix to all the links to make the link absolute 
        shooting_df = pd.read_html(shooting_urls.text, match = "Shooting")[0]
        shooting_df.columns = shooting_df.columns.droplevel()

        #Skip over a team that doesn't have shooting stats
        try: 
            team_df = matches_df.merge(shooting_df[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError: 
            continue 

        team_df = team_df[team_df["Comp"] == "Premier League"] #only want games from prem not other competitions
        #Add data to df that is missing from original table
        team_df["Season"] = year
        team_df["Team"] = team_name
        all_games.append(team_df)
        time.sleep(15) #Try to avoid getting blocked from scraping

final_df = pd.concat(all_games)
final_df.to_csv("prem_matches.csv")


#inspired by dataquest tutorial 
