# NBA Moneyball

This repo contains the code for an NBA moneyball inspired project. Moneyball is a semi-documentary about the 2002 Oakland Athletic's MLB team GM Billy Beane who attempts to create an undervalued team with the help of assistant GM Peter Brand using sabermetrics. Essentially, we are trying to get the best bang for buck using statistics and in our case, machine learning. We transfer this concept to the NBA to try to find the best teams that are within a salary cap.

## Requirements

This dashboard is hosted on Streamlit's community cloud and does not require you to run the code locally. You can access the site through this link: https://github.com/cdy3870/NBA_moneyball.

## Libraries, Frameworks, APIs, Cloud Services
1. Libraries and Frameworks
- Pandas
- Streamlit
- Scikit-Learn
- Plotly
2. Data Source
- Aggregated Data (Basketball Reference: https://www.basketball-reference.com/)

## How it works and services involved
1. After aggregating data such as advanced statistics, team wins and starting rosters for the 20-21 season, and salaries for the 20-21 season, we can begin to generate our teams.
2. Training
- First we filter based on a certain number of games and minutes played so that there is no bias for the statistics
- Group based on team and generate n combinations of teams an average out their advanced statistics as the independent variables 
- The dependent variable (major assumption here) is the same number of wins that the particular team received during the 20-21 season
3. The testing data is m combinations of teams where the advanced statistics are also averaged out
4. We obtain the highest predicted number of wins on the testing data and perform additional analysis such as
- How do the average statistics of the suggested teams compare to real, successful starting 5s in the 20-21 season? We'd expect the average statistics to be comparable or better, which is shown to be true in the visualizations.
- How many players in the suggested team were part of the top 10 teams in the 20-21 season? What about the bottom 10 teams? We'd expect there to be high number of players in the top 10 teams and low number of players from the bottom 10 teams, which is shown to be true in the visualizations. 
