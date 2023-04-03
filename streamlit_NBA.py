import pandas as pd
import itertools
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# warnings.filterwarnings('ignore')
# pd.set_option('display.max_columns', 70)

st.set_page_config(layout="wide")
st.title("NBA Moneyball")


min_games_played = 41
min_min_played = 25
desired_yearly_sal_min = 50000000
desired_yearly_sal_max = 70000000

combos_considered = 30000
num_players = 5
position_matters = True
teams_considered = 30000

teams_dict = {"MIA": "Miami Heat", "NOP": "New Orleans Pelicans", "MEM": "Memphis Grizzlies", "MIL": "Milwaukee Bucks", "POR": "Portland Trail Blazers", "WAS": "Washington Wizards", "PHO": "Phoenix Suns", "CHO": "Charlotte Hornets", "SAC": "Sacramento Kings", "NYK": "New York Knicks", "DEN": "Denver Nuggets", "LAC": "Los Angeles Clippers", "GSW": "Golden State Warriors", "OKC": "Oklahoma City Thunder", "TOR": "Toronto Raptors", "DET": "Detroit Pistons", "UTA": "Utah Jazz", "IND": "Indiana Pacers", "DAL": "Dallas Mavericks", "BRK": "Brooklyn Nets", "BOS": "Boston Celtics", "HOU": "Houston Rockets", "LAL": "Los Angeles Lakers", "ATL": "Atlanta Hawks", "PHI": "Philadelphia 76ers", "SAS": "San Antonio Spurs", "MIN": "Minnesota Timberwolves", "CLE": "Cleveland Cavaliers", "CHI": "Chicago Bulls", "ORL": "Orlando Magic"}


def get_data():
	adv_stats = pd.read_excel("adv_stats_with_dup.xlsx").drop("Rk", axis=1)
	full_salaries = pd.read_excel("salaries.xlsx").drop("Rk", axis=1).rename(columns = {"2021-22":"2021-22 Salary"})
	standings = pd.read_excel("standings.xlsx")
	salaries_18_19 = pd.read_excel("team_salaries-18-19.xlsx")
	salaries_19_20 = pd.read_excel("team_salaries-19-20.xlsx")
	salaries_20_21 = pd.read_excel("team_salaries-20-21.xlsx")
	salaries_21_22 = pd.read_excel("team_salaries-21-22.xlsx")

	salaries = full_salaries[["Player", "2021-22 Salary"]]

	team_salaries = pd.merge(salaries_18_19, salaries_19_20, on=["Team"])
	team_salaries = pd.merge(team_salaries, salaries_20_21, on=["Team"])
	team_salaries = pd.merge(team_salaries, salaries_21_22, on=["Team"])
	team_salaries.columns = ["Team", "18-19 Salary", "19-20 Salary", "20-21 Salary", "21-22 Salary"]
	print(team_salaries)

	combined_data = pd.merge(adv_stats, full_salaries[["Player", "2021-22 Salary"]], on=["Player"]).drop_duplicates()
	

	return standings, salaries, combined_data, team_salaries


def parse_data(combined_data, position_matters=False):
	parsed_data = combined_data[combined_data["Tm"] != "TOT"]
	# parsed_data = parsed_data[parsed_data["G_x"] >= min_games_played]
	# parsed_data = parsed_data[parsed_data["MP_x"] >= min_min_played].reset_index(drop=True)
	# parsed_data = parsed_data.drop("Pos_x", axis=1).drop("Age_x", axis=1).drop("G_x", axis=1).drop("MP_x", axis=1)
	# parsed_data = parsed_data.drop("Pos_y", axis=1).drop("Age_y", axis=1).drop("Tm_y", axis=1).drop("G_y", axis=1).drop("GS", axis=1).drop("MP_y", axis=1)

	parsed_data = parsed_data[parsed_data["G"] >= min_games_played]
	parsed_data = parsed_data[parsed_data["MP"] >= min_min_played].reset_index(drop=True)
	parsed_data = parsed_data.drop("G", axis=1).drop("MP", axis=1)
	# parsed_data = parsed_data.sort_values(by=['Pos']).reset_index(drop=True)

	players_considered = len(parsed_data)

	if position_matters:  
		temp = parsed_data.groupby('Pos')
		data_by_positions = [x for _, x in temp]

	return parsed_data

def get_five(data_len):
	random_set = []
	# if position_matters:
	#     for i in range(len(data_by_positions)):
	#         random_index = random.randint(0, len(data_by_positions[i]) - 1)

	#         while random_index in random_set:
	#             random_index = random.randint(0, len(data_by_positions[i]) - 1)
	#         new_index = data_by_positions[i].iloc[random_index].name
	#         random_set.append(new_index)
	# else:
	for i in range(num_players):
		random_index = random.randint(0, data_len - 1)
		while random_index in random_set:
			random_index = random.randint(0, data_len - 1)
		random_set.append(random_index)
	return random_set

def get_model(standings, parsed_data, params):
	# Train data
	teams = standings["Team"]
	team_wins = standings["Wins"]

	players_dict = {}
	for i in range(len(parsed_data)):
		team = parsed_data.iloc[i]["Tm"]
		if team not in players_dict:
			players_dict[team] = [i]
		else:
			players_dict[team].append(i)

	temp_data = parsed_data
	parsed_data = parsed_data.drop("Tm", axis=1)
	parsed_data = parsed_data.drop("Pos", axis=1)


	different_teams = []

	for key, value in players_dict.items():
		wins = standings[standings["Team"] == teams_dict[key]]["Wins"].values[0]
		
		different_teams.append((wins, list(itertools.combinations(value, num_players))))

	X = []
	y = [] 
	for i in range(len(different_teams)):
		team = different_teams[i][1]
		for j in range(len(team)):
			players = list(team[j])
			averages = parsed_data.iloc[players].mean(axis=0).tolist()[:-1]
			X.append(averages)
			y.append(different_teams[i][0])


	# Test data
	data_len = len(parsed_data)
	all_combos = [get_five(data_len) for i in range(teams_considered)]

	salaries = parsed_data["2021-22 Salary"].tolist()
	players = parsed_data["Player"]

	team_salaries = []

	for i in range(len(all_combos[:combos_considered])):
		total_sal = 0
		for j in range(num_players):
			total_sal += salaries[all_combos[i][j]]
			
		if total_sal <= params["max_values"] and total_sal >= params["min_values"]:  
			team_salaries.append((total_sal, all_combos[i]))

	extra_test_X = []
	extra_teams_X = []
	for sal in team_salaries:
		players = sal[1]
		averages = parsed_data.iloc[players].mean(axis=0).tolist()[:-1]
		extra_test_X.append(averages)
		extra_teams_X.append(players)


	train_X, true_test_X, train_y, true_test_y = train_test_split(X, y, test_size=0.2)
	scaler = StandardScaler()
	scaler.fit(train_X)
	scaler.transform(train_X)
	# print(train_X)
	model = LinearRegression().fit(train_X, train_y)


	return model, extra_test_X, extra_teams_X, temp_data, parsed_data
		

def get_pred_info(y_pred, extra_teams_X, temp_data, parsed_data, top_n=1, get_worst=False):
	if get_worst:
		index = np.argmin(y_pred)
		players = extra_teams_X[index]

		num_wins = y_pred[index]
		stats = pd.merge(temp_data[["Player", "Tm", "Pos"]], parsed_data.iloc[players], on=["Player"])
#         stats = pd.merge(parsed_data.iloc[players], combined_data[["Player", "Pos"]], on=["Player"])
#         stats = parsed_data.iloc[players]
		averages = pd.DataFrame({"Averages": stats.mean(axis=0).round(1)}).T

		total_salary = int(stats.sum(axis=0)["2021-22 Salary"].tolist())
		player_names = stats["Player"].tolist()
		return stats, averages, player_names, total_salary, int(np.round(num_wins, 0))
	
	else:
		indices = np.argpartition(y_pred, -top_n)[-top_n:]

		sorted_indices = indices[np.argsort(y_pred[indices])]

		info = []

		for i in range(len(sorted_indices) - 1, -1, -1):
			index = sorted_indices[i]
			players = extra_teams_X[index]

			num_wins = y_pred[index]
			stats = pd.merge(temp_data[["Player", "Tm", "Pos"]], parsed_data.iloc[players], on=["Player"])
#             stats = parsed_data.iloc[players]
			averages = pd.DataFrame({"Averages": stats.mean(axis=0).round(1)}).T

			total_salary = int(stats.sum(axis=0)["2021-22 Salary"].tolist())
			player_names = stats["Player"].tolist()
			info.append([stats, averages, player_names, total_salary, int(np.round(num_wins, 0))])
		return info 


def get_standings_fig(standings):

	fig = px.bar(standings, x="Team", y=["Wins", "Losses"], title="2021-2022 Standings")
	fig.update_xaxes(tickangle=-45)

	return fig

def get_salaries_fig(standings, team_salaries, option):
	top_10_teams = standings["Team"][:10].values
	top_10_teams = [x.split(" ")[0] for x in top_10_teams]
	df = team_salaries[team_salaries["Team"].isin(top_10_teams)]
	if option == "All (19-22)":
		fig = px.line(df, x="Team", y=["21-22 Salary", "20-21 Salary", "19-20 Salary", "18-19 Salary"], title="Total Salary For T-10 Teams")
	else:
		fig = px.line(df, x="Team", y=option + " Salary", title="Total Salary For T-10 Teams")


	fig.update_xaxes(tickangle=-45)

	return fig

def get_stats_fig(parsed_data, predicted_avgs, high_or_low="Higher"):
	suns_starters = ["Devin Booker", "Chris Paul", "Mikal Bridges", "Deandre Ayton", "Jae Crowder"]
	warrs_starters = ["Steph Curry", "Klay Thompson", "Draymond Green", "Otto Porter Jr.", "Andrew Wiggings"]
	heat_starters = ["Jimmy Butler", "Bam Adebayo", "Duncan Robinson", "PJ Tucker", "Kyle Lowry"]

	subset = parsed_data[parsed_data["Player"].isin(suns_starters)]
	subset_avgs_suns = pd.DataFrame(subset.mean(axis=0).round(1))

	subset = parsed_data[parsed_data["Player"].isin(heat_starters)]
	subset_avgs_heat = pd.DataFrame(subset.mean(axis=0).round(1))

	subset = parsed_data[parsed_data["Player"].isin(warrs_starters)]
	subset_avgs_warrs = pd.DataFrame(subset.mean(axis=0).round(1))

	if high_or_low == "Higher":
		fig = make_subplots(rows=3, cols=3, subplot_titles=["PER", "TS%", "ORB%", "DRB%", "TRB%", "BLK%", "STL%"])
		row = 1
		col = 1

		for column in ["PER", "TS%", "ORB%", "DRB%", "TRB%", "BLK%", "STL%"]:
			x = ["Suns", 'Heat', 'Warriors', "Predicted"]
			y = [subset_avgs_suns.loc[column].values[0], subset_avgs_heat.loc[column].values[0], subset_avgs_warrs.loc[column].values[0], predicted_avgs[column].values[0]]
			# axes[row, col].bar(x, y, color=['orange', 'red', 'blue', 'black'])
			# axes[row, col].set_title(column)
			print((x, y))

			fig.append_trace(go.Bar(x=x, y=y, marker_color=['orange', 'red', 'blue', 'black']), row=row, col=col)

			col += 1

			if col == 4:
				col = 1
				row += 1
	else:
		fig = make_subplots(rows=2, cols=1, subplot_titles=["TOV%", "2021-22 Salary"])
		row = 1
		col = 1

		for column in ["TOV%", "2021-22 Salary"]:
			x = ["Suns", 'Heat', 'Warriors', "Predicted"]
			y = [subset_avgs_suns.loc[column].values[0], subset_avgs_heat.loc[column].values[0], subset_avgs_warrs.loc[column].values[0], predicted_avgs[column].values[0]]
			fig.append_trace(go.Bar(x=x, y=y, marker_color=['orange', 'red', 'blue', 'black']), row=row, col=col)
			row += 1
			print(row)
			print(col)
			print(y)
			print(column)

	fig.update_layout(height=600, width=600, title_text="Team Averages")
	fig.update(layout_showlegend=False)


	return fig

def get_overlap_fig(info, top_10_teams, bot_10_teams):
	fig = make_subplots(rows=2, cols=3, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}], [{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])
	row = 1
	col = 1

	for i in range(len(info)):
		stats, averages, player_names, total_salary, num_wins = info[i]
		player_teams = set(map(lambda s : teams_dict[s], stats["Tm"]))
		top_overlap = len(player_teams.intersection(top_10_teams))
		bot_overlap = len(player_teams.intersection(bot_10_teams))

		if top_overlap == 0: 
			sizes = [bot_overlap * 10, 100 - (bot_overlap * 10)]        
			labels=["In Bottom-10", "In Mid-Tier"]

		elif bot_overlap == 0:
			sizes = [top_overlap * 10, 100 - (top_overlap * 10)]
			labels = ["In Top-10", "In Mid-Tier"]

		else:
			sizes = [top_overlap * 10, bot_overlap * 10, 100 - (bot_overlap * 10 + top_overlap * 10)]
			labels = ["In Top-10", "In Bottom-10", "In Mid-Tier"]



		fig.add_trace(go.Pie(
			values=sizes,
			labels=labels,
			domain=dict(x=[0, 0.5])), 
			row=row, col=col)

		col += 1
		
		if col == 4:
			col = 1
			row += 1  

	fig.update_layout(title_text="Players Currently in Top, Middle, or Bottom Teams from 2021-22 Season")

	return fig

@st.cache_data
def cache_get_data():
	standings, salaries, combined_data, team_salaries = get_data()

	return standings, salaries, combined_data, team_salaries

@st.cache_data
def cache_get_model(standings, salaries, combined_data, team_salaries, params):
	parsed_data = parse_data(combined_data)

	model, extra_test_X, extra_teams_X, temp_data, new_parsed_data = get_model(standings, parsed_data, params)
	extra_pred_y = model.predict(extra_test_X)
	
	return parsed_data, model, extra_test_X, extra_teams_X, temp_data, new_parsed_data, extra_pred_y

def main():
	metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
	metric_col1.markdown('##### Author: Calvin Yu\n')
	standings, salaries, combined_data, team_salaries = cache_get_data()

	print(standings)

	st.markdown("#### Overview")
	st.plotly_chart(get_standings_fig(standings), use_container_width=True)
	year = st.selectbox("Year", ("All (19-22)", "21-22", "20-21", "19-20", "18-19"))
	st.plotly_chart(get_salaries_fig(standings, team_salaries, year), use_container_width=True)


	with st.sidebar:
		"Configure the details of your starting 5"
		min_values = st.slider(
				'Select desired yearly salary minimum',
				min_value=20000000, max_value=60000000, step=100000)

		max_values = st.slider(
				'Select desired yearly salary maximum',
				min_value=60000000, max_value=100000000, step=100000)

		min_minutes = st.slider(
				'Select minimum minutes played',
				min_value=0, max_value=48, value=24)

		min_games = st.slider(
				'Select minimum games played',
				min_value=0, max_value=82, value=41)

		# st.write('Values:', values)

		params = {"min_values":min_values,
				"max_values":max_values,
				"min_minutes":min_minutes,
				"min_games":min_games}

	st.markdown("#### Generated Teams")

	parsed_data, model, extra_test_X, extra_teams_X, temp_data, new_parsed_data, extra_pred_y = cache_get_model(standings, salaries, combined_data, team_salaries, params)

	info = get_pred_info(extra_pred_y, extra_teams_X, temp_data, new_parsed_data, top_n=5)
	params = {"Players Considered": len(parsed_data), "Position Matters": False, "Size of Team": num_players, "Min Games Played": params["min_games"], "Min Minutes Played": params["min_minutes"], "Salary Range (Min)": "${:,}".format(params["min_values"]), "Salary Range (Max)": "${:,}".format(params["max_values"])} 
	params_df = pd.DataFrame({"Parameters": params})
	params_df.T

	rank = st.selectbox("Rank", (1, 2, 3, 4, 5))

	stats, averages, player_names, total_salary, num_wins = info[rank - 1]
	stats

	my_expander = st.expander(label='Individual Team Stats')
	with my_expander:
		high_or_low = st.radio("Higher or Lower Value is Better", ("Higher", "Lower"))		
		st.plotly_chart(get_stats_fig(parsed_data, averages, high_or_low=high_or_low), use_container_width=True)

	my_expander_2 = st.expander(label='All 5 Team Stats')
	top_10_teams = standings["Team"][:10].values
	bot_10_teams = standings["Team"][-10:]
	with my_expander_2:
		st.plotly_chart(get_overlap_fig(info, top_10_teams, bot_10_teams))


if __name__ == "__main__":
	main()