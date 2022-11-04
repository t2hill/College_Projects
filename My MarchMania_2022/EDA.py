import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#####   Read all the data files and make dataFrames   #############################################
teams = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage1/MTeams.csv')
teams = teams[teams.LastD1Season == 2022]

rscResults = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
rsdResults = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv')
tourney_seeds = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneySeeds.csv')
tcResults = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
mOrdinals = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage1/MMasseyOrdinals.csv')
###############################################################################


### EXPLORATORY DATA ANALYSIS #################################################


##########
def rsdResults_Heatmap(df, width, height):
    data = df.corr()
    fig = plt.figure(figsize=(width, height))
    fig = sn.heatmap(data = data,  cmap="Blues", linewidths=0.1, linecolor="white", annot=True, annot_kws={"size": 8})
    for t in fig.texts:
        if float(t.get_text()) >= 0.5:
            t.set_text(t.get_text())  # if the value is greater than 0.6 then I set the text
        else:
            t.set_text("")  # if not it sets an empty text

    plt.yticks(rotation=0)
    plt.show()
##########
# rsdResults_relevant = rsdResults.filter(['WScore','LScore','WFGM','WFGA','WAst','WPF','WFTA','WOR',
#                                         'LFGM','LFGA','LAst','LPF','LFTA','LOR'], axis=1)
# rsdResults_Heatmap(rsdResults_relevant,10, 10)   # Heatmap using the correlation of rsdResults


##########
def calc_teamsWL():
    teams_fmt = teams.drop(columns=['FirstD1Season', 'LastD1Season'])
        # Regular Season
    rscResults['counter'] = 1
    wins = rscResults.groupby('WTeamID')['counter'].count()
    wins.rename('Season_WTotal', inplace=True)
    loses = rscResults.groupby('LTeamID')['counter'].count()
    loses.rename('Season_LTotal', inplace=True)

    teams_fmt = pd.merge(teams_fmt, wins, left_on='TeamID', right_on='WTeamID')
    teams_fmt = pd.merge(teams_fmt, loses, left_on='TeamID', right_on='LTeamID')

    teams_fmt['Season_WPercentage'] = round(100 * (teams_fmt['Season_WTotal'] / (teams_fmt['Season_WTotal'] + teams_fmt['Season_LTotal'])), 2)

        # Tourney
    tcResults['counter'] = 1
    wins = tcResults.groupby('WTeamID')['counter'].count()
    wins.rename('Tourney_WTotal', inplace=True)
    loses = tcResults.groupby('LTeamID')['counter'].count()
    loses.rename('Tourney_LTotal', inplace=True)

    teams_fmt = pd.merge(teams_fmt, wins, left_on='TeamID', right_on='WTeamID')
    teams_fmt = pd.merge(teams_fmt, loses, left_on='TeamID', right_on='LTeamID')

    teams_fmt['Tourney_WPercentage'] = round(100 * (teams_fmt['Tourney_WTotal'] / (teams_fmt['Tourney_WTotal'] + teams_fmt['Tourney_LTotal'])), 2)

    teams_fmt = teams_fmt.drop(columns=['TeamID'])
    return teams_fmt
##########
# print(calc_teamsWL().head(5).to_string())

##########
def teamNum_increase_perYear(width, height):
    yr_count = pd.DataFrame({'year' : np.arange(1985, 2022)})
    teams_perYear = teams

    for year in yr_count['year']:
        teams_perYear['teams_in'] = 0
        teams_perYear.loc[(teams_perYear['FirstD1Season'] <= year) & (teams_perYear['LastD1Season'] >= year), 'teams_in'] = 1
        total_teams = teams_perYear['teams_in'].sum()
        yr_count.loc[yr_count.year == year, 'numOf_teams'] = total_teams

    yr_count = yr_count.set_index('year')
    yr_count['numOf_teams'].plot(figsize=(width, height))
    plt.title("Number of Teams Per Year", fontsize= 16)
    plt.show()
##########
# teamNum_increase_perYear(10, 10)

##########
def timesInTourney():
    tourney_seeds_fmt = tourney_seeds
    tourney_seeds_fmt['counter'] = 1
    times_in = tourney_seeds_fmt.groupby('TeamID')['counter'].count()
    times_in.rename('TimesInTourney', inplace=True)
    tourney_seeds_fmt = pd.merge(teams, times_in, left_on='TeamID', right_on='TeamID')
    tourney_seeds_fmt = tourney_seeds_fmt.drop(columns=['TeamID', 'FirstD1Season', 'LastD1Season'])
    return tourney_seeds_fmt
##########
# print(timesInTourney().sort_values('TimesInTourney', ascending= False).to_string())

##########
def winPercents_perSeason():
    W_A = rsdResults[rsdResults['WLoc'] == 'A'].groupby(['Season', 'WTeamID'])[
        'WTeamID'].count().to_frame().rename(columns={"WTeamID": "Win_A"})
    W_N = rsdResults[rsdResults['WLoc'] == 'N'].groupby(['Season', 'WTeamID'])[
        'WTeamID'].count().to_frame().rename(columns={"WTeamID": "Win_N"})
    W_H = rsdResults[rsdResults['WLoc'] == 'H'].groupby(['Season', 'WTeamID'])['WTeamID'].count().to_frame().rename(
        columns={"WTeamID": "Win_H"})
    win = W_A.join(W_H, how='outer').join(W_N, how='outer').fillna(0)

    L_A = rsdResults[rsdResults['WLoc'] == 'A'].groupby(['Season', 'LTeamID'])[
        'LTeamID'].count().to_frame().rename(columns={"LTeamID": "Lost_A"})
    L_N = rsdResults[rsdResults['WLoc'] == 'N'].groupby(['Season', 'LTeamID'])[
        'LTeamID'].count().to_frame().rename(columns={"LTeamID": "Lost_N"})
    L_H = rsdResults[rsdResults['WLoc'] == 'H'].groupby(['Season', 'LTeamID'])['LTeamID'].count().to_frame().rename(
        columns={"LTeamID": "Lost_H"})
    lose = L_A.join(L_H, how='outer').join(L_N, how='outer').fillna(0)

    win.index = win.index.rename(['Season', 'TeamID'])
    lose.index = lose.index.rename(['Season', 'TeamID'])
    winLoss = win.join(lose, how='outer').reset_index()
    winLoss['H_WinPercent'] = round(100 * (winLoss['Win_H'] / (winLoss['Win_H'] + winLoss['Lost_H'])), 2)
    winLoss['A_WinPercent'] = round(100 * (winLoss['Win_A'] / (winLoss['Win_A'] + winLoss['Lost_A'])), 2)
    winLoss['N_WinPercent'] = round(100 * (winLoss['Win_N'] / (winLoss['Win_N'] + winLoss['Lost_N'])), 2)
    winLoss['W_Games'] = winLoss['Win_A'] + winLoss['Win_N'] + winLoss['Win_H']
    winLoss['L_Games'] = winLoss['Lost_A'] + winLoss['Lost_N'] + winLoss['Lost_H']
    winLoss['Season_WPercent'] = round(100 * winLoss['W_Games'] / (winLoss['W_Games'] + winLoss['L_Games']), 2)
    # winLoss = winLoss.drop(
    #     columns=['Win_A', 'Win_H', 'Win_N', 'Lost_H', 'Lost_A', 'Lost_N', 'W_Games', 'L_Games'])
    winLoss = winLoss.dropna().reset_index(drop=True)
    return winLoss
##########


########## HOME COURT ADVANTAGE
def HomeCourtADV():
    winPercents = winPercents_perSeason()
    wins_df = winPercents.drop(columns=['Season', 'TeamID', 'Season_WPercent'])
    Hwins = round(wins_df['Win_H'].mean(), 2)
    Awins = round(wins_df['Win_A'].mean(), 2)
    Nwins = round(wins_df['Win_N'].mean(), 2)
    total = Hwins + Awins + Nwins
    Hprct = round(100 * (Hwins / total), 2)
    Aprct = round(100 * (Awins / total), 2)
    Nprct = round(100 * (Nwins / total), 2)

    # plot
    y = np.array([Hprct, Aprct, Nprct])
    mylabels = ["Home", "Away", "Neutral"]
    colors = ['#66b3ff', '#ff9999', '#CDB7F6']

    plt.pie(y, labels=mylabels, autopct='%1.2f%%', colors=colors, startangle=90)
    # draw circle
    centre_circle = plt.Circle((0, 0), 0.77, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.tight_layout()
    plt.suptitle('Win Locations')
    plt.show()
##########


##########
def trainingStatsCorrelation():
    train = pd.read_csv('train.csv')
    train = train.drop(columns={'Season', 'Team1ID', 'Team2ID'})
    train_corr = train.corr()
    return train_corr
##########


########## Shows that most games came down to only a few points
def pointDiffPlot(df):
    df['PointDiff'] = df['WScore'] - df['LScore']
    plt.figure(figsize=(10, 6))
    sn.countplot(df[df["PointDiff"] < 50]["PointDiff"])
    plt.xticks(np.arange(4, 51, 5), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Point Differential', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.show()
##########
# pointDiffPlot(tcResults)  # games still come close at the end but not as frequent
# pointDiffPlot(rscResults)  # most games came down to only a few points


########## We can see that there is a long of change in teams rank as the season continues
def rankChangeThuSeason():
    df = mOrdinals[mOrdinals['Season'] < 2022]
    df_begin = df[df['RankingDayNum'] <= 40]
    df_end = df[df['RankingDayNum'] >= 128]

    df_begin = df_begin.groupby(['Season', 'TeamID'])['OrdinalRank'].agg(np.mean).reset_index()
    df_end = df_end.groupby(['Season', 'TeamID'])['OrdinalRank'].agg(np.mean).reset_index()

    df = pd.merge(df_begin, df_end, on=['Season', 'TeamID'], how='left')
    df['Rank Change'] = round(abs(df['OrdinalRank_x'] - df['OrdinalRank_y']))
    df = df.dropna().reset_index(drop=True)
    df['Rank Change'] = df['Rank Change'].apply(np.int64)

    plt.figure(figsize=(10, 6))
    sn.countplot(df[df['Rank Change'] < 100]['Rank Change'])
    sn.color_palette("crest", as_cmap=True)
    plt.xticks(np.arange(0, 101, 10), fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel("Number of Teams", fontsize=18)
    plt.xlabel('Rank Change', fontsize=18)
    plt.show()
##########
