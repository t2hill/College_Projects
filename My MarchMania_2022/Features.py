import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


########## Adds custom features to the datasets
def addFeatures(dResults):
    dResults['WPOSS'] = round(dResults['WFGA']
                              - dResults['WOR']
                              + dResults['WTO']
                              + (0.44 * dResults['WFTA']), 3)
    dResults['LPOSS'] = round(dResults['LFGA']
                              - dResults['LOR']
                              + dResults['LTO']
                              + (0.44 * dResults['LFTA']), 3)
    dResults['WORating'] = round((dResults['WScore'] / dResults['WPOSS']) * 100, 1)
    dResults['LORating'] = round((dResults['LScore'] / dResults['LPOSS']) * 100, 1)
    dResults['WDRating'] = dResults['LORating']
    dResults['LDRating'] = dResults['WDRating']
    dResults['WFG%'] = round(dResults['WFGM'] / dResults['WFGA'], 3)
    dResults['LFG%'] = round(dResults['LFGM'] / dResults['LFGA'], 3)
    dResults['WFT%'] = round(dResults['WFTM'] / dResults['WFTA'], 3)
    dResults['LFT%'] = round(dResults['LFTM'] / dResults['LFTA'], 3)
    dResults['WReb'] = dResults['WOR'] + dResults['WDR']
    dResults['LReb'] = dResults['LOR'] + dResults['LDR']
    dResults['WDef'] = dResults['WStl'] + dResults['WBlk']
    dResults['LDef'] = dResults['LStl'] + dResults['LBlk']
    dResults = dResults.drop(columns=['WFGM', 'WFGA', 'WFGM3', 'WFGA3',
                                      'WOR', 'WDR', 'WFTM', 'WFTA', 'WStl', 'WBlk',
                                      'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
                                      'LOR', 'LDR', 'LFTM', 'LFTA', 'LStl', 'LBlk'])

    return dResults
##########


########## Generates averages for each teams stats per season
def statsPerTeam(dResults):
    dResults = addFeatures(dResults)
    stats_col = ['Score', 'FG%', 'Reb', 'FT%', 'Ast', 'TO',
                 'Def', 'PF', 'POSS', 'ORating', 'DRating']
    box_statsW = dResults[['Season', 'WTeamID'] + ['W' + col for col in stats_col]]
    box_statsL = dResults[['Season', 'LTeamID'] + ['L' + col for col in stats_col]]
    box_statsW = box_statsW.rename(columns={'WTeamID': 'TeamID'})
    box_statsW = box_statsW.rename(columns={('W' + col): col for col in stats_col})
    box_statsL = box_statsL.rename(columns={'LTeamID': 'TeamID'})
    box_statsL = box_statsL.rename(columns={('L' + col): col for col in stats_col})
    box_stats = pd.merge(box_statsW, box_statsL, on=['Season', 'TeamID'] + stats_col, how='outer')
    box_stats = box_stats.groupby(['Season', 'TeamID'])[stats_col].agg(np.mean).reset_index()
    box_stats = round(box_stats, 3)
    box_stats = box_stats.rename(columns={'Score': 'PPG'})
    # print(box_stats.head().to_string())
    return box_stats
##########


########## initializes the train set
def getTraining(dtResults):
    dtResults = addFeatures(dtResults)
    dtResults2 = dtResults.copy()

    dtResults = dtResults.rename(columns={'WTeamID': 'Team1ID', 'LTeamID': 'Team2ID',
                                          'WScore': 'T1Score', 'LScore': 'T2Score'})
    dtResults2 = dtResults2.rename(columns={'WTeamID': 'Team2ID', 'LTeamID': 'Team1ID',
                                            'WScore': 'T1Score', 'LScore': 'T2Score'})

    features = ['Season', 'Team1ID', 'Team2ID', 'T1Win']
    dtResults['T1Win'] = 1.0
    dtResults2['T1Win'] = 0.0
    train_df = pd.merge(dtResults, dtResults2, on=features, how="outer")
    train_df = train_df[features]
    # print(train_df)
    return train_df
##########


########## initializes the test set
def getTesting(stageSubmission):
    seasons = []
    team1ID = []
    team2ID = []
    for i in range(stageSubmission.shape[0]):
        szn = int((stageSubmission['ID'].iloc[i])[0:4])
        t1 = int((stageSubmission['ID'].iloc[i])[5:9])
        t2 = int((stageSubmission['ID'].iloc[i])[10:14])
        seasons.append(szn)
        team1ID.append(t1)
        team2ID.append(t2)
    stageSubmission['Season'] = seasons
    stageSubmission['Team1ID'] = team1ID
    stageSubmission['Team2ID'] = team2ID
    return stageSubmission
##########


########## combines the season and the tournament stats
def addTeamStats(dtResults, dsResults, df_toAddto):
    seasonTeamStats = statsPerTeam(dsResults)
    tourneyTeamStats = statsPerTeam(dtResults)

    stat_col = ['PPG', 'FG%', 'FT%', 'Reb', 'Ast', 'TO',
                'Def', 'PF', 'POSS', 'ORating', 'DRating']

    stats_N1 = tourneyTeamStats.copy()
    stats_N2 = tourneyTeamStats.copy()
    stats_S1 = seasonTeamStats.copy()
    stats_S2 = seasonTeamStats.copy()

    stats_N1.columns = ['Season', 'Team1ID'] + ['T1' + stat for stat in stat_col]
    stats_N2.columns = ['Season', 'Team2ID'] + ['T2' + stat for stat in stat_col]
    stats_S1.columns = ['Season', 'Team1ID'] + ['T1' + stat for stat in stat_col]
    stats_S2.columns = ['Season', 'Team2ID'] + ['T2' + stat for stat in stat_col]

    temp_N = pd.merge(df_toAddto, stats_N1, on=['Season', 'Team1ID'], how='left')
    temp_S = pd.merge(df_toAddto, stats_S1, on=['Season', 'Team1ID'], how='left')

    tourneyTeamStats = pd.merge(temp_N, stats_N2, on=['Season', 'Team2ID'], how='left')
    seasonTeamStats = pd.merge(temp_S, stats_S2, on=['Season', 'Team2ID'], how='left')

    if 'T1Win' in df_toAddto:
        teamStats = pd.merge(tourneyTeamStats, seasonTeamStats,
                             on=['Season', 'Team1ID', 'Team2ID', 'T1Win'] +
                                ['T1' + stat for stat in stat_col] +
                                ['T2' + stat for stat in stat_col], how='outer')
        teamStats = teamStats.groupby(['Season', 'Team1ID', 'Team2ID', 'T1Win'])[
            ['T1' + stat for stat in stat_col] +
            ['T2' + stat for stat in stat_col]].agg(np.mean).reset_index()
        teamStats = round(teamStats, 3)
        return teamStats
    else:
        teamStats = pd.merge(tourneyTeamStats, seasonTeamStats,
                             on=['Season', 'Team1ID', 'Team2ID'] +
                                ['T1' + stat for stat in stat_col] +
                                ['T2' + stat for stat in stat_col], how='outer')
        teamStats = teamStats.groupby(['Season', 'Team1ID', 'Team2ID'])[
            ['T1' + stat for stat in stat_col] +
            ['T2' + stat for stat in stat_col]].agg(np.mean).reset_index()
        teamStats = round(teamStats, 3)
        return teamStats
##########


########## returns the tournament seed for both teams playing
def addSeedRankings(seeds):
    seeds['Seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
    seeds_T1 = seeds[['Season', 'TeamID', 'Seed']].copy()
    seeds_T2 = seeds[['Season', 'TeamID', 'Seed']].copy()
    seeds_T1.columns = ['Season', 'Team1ID', 'T1_Seed']
    seeds_T2.columns = ['Season', 'Team2ID', 'T2_Seed']

    return seeds_T1, seeds_T2
##########


########## returns the average season rank for both teams playing
def addOrdinals(ranks):
    ranks = ranks[ranks['RankingDayNum'] >= 118]
    ranks = ranks.drop(columns={'RankingDayNum', 'SystemName'})
    ranks = ranks.groupby(['Season', 'TeamID'])['OrdinalRank'].agg(np.mean).reset_index()
    ranks['OrdinalRank'] = round(ranks['OrdinalRank'], 3)
    ranks_T1 = ranks.copy()
    ranks_T2 = ranks.copy()
    ranks_T1.columns = ['Season', 'Team1ID', 'T1_Rank']
    ranks_T2.columns = ['Season', 'Team2ID', 'T2_Rank']

    return ranks_T1, ranks_T2
########## changed to only take the rankings from the last 2 weeks before the tournament


########## generates a training set
def createTrain(dtResults, dsResults, seeds, ranks, startYear, endYear):
    seasonList = list(range(startYear, endYear + 1))
    if seasonList.__contains__(2020):
        seasonList.remove(2020)
    dtResults = dtResults.loc[dtResults['Season'].isin(seasonList)].copy().reset_index(drop=True)
    dsResults = dsResults.loc[dsResults['Season'].isin(seasonList)].copy().reset_index(drop=True)
    seeds = seeds.loc[seeds['Season'].isin(seasonList)].copy().reset_index(drop=True)
    ranks = ranks.loc[ranks['Season'].isin(seasonList)].copy().reset_index(drop=True)

    df_train = getTraining(dtResults)
    df_train = addTeamStats(dtResults, dsResults, df_train)

    seeds_T1, seeds_T2 = addSeedRankings(seeds)
    df_train = pd.merge(df_train, seeds_T1, on=['Season', 'Team1ID'], how='left')
    df_train = pd.merge(df_train, seeds_T2, on=['Season', 'Team2ID'], how='left')
    df_train['SeedDiff'] = df_train['T1_Seed'] - df_train['T2_Seed']
    for i in range(df_train['SeedDiff'].shape[0]):
        # A team with SeedDiff >= 10 has only happened once
        if df_train['SeedDiff'].iloc[i] >= 10:
            df_train['SeedDiff'].iloc[i] = 15
        if df_train['SeedDiff'].iloc[i] <= -10:
            df_train['SeedDiff'].iloc[i] = -15

    ranks_T1, ranks_T2 = addOrdinals(ranks)
    df_train = pd.merge(df_train, ranks_T1, on=['Season', 'Team1ID'], how='left')
    df_train = pd.merge(df_train, ranks_T2, on=['Season', 'Team2ID'], how='left')
    df_train['RankDiff'] = df_train['T1_Rank'] - df_train['T2_Rank']

    return df_train
##########


########## generates a test set
def createTest(dtResults, dsResults, seeds, ranks, startYear, endYear, sampleSubmission):
    df_test = getTesting(sampleSubmission)

    seasonList = list(range(startYear, endYear + 1))
    if seasonList.__contains__(2020):
        seasonList.remove(2020)

    dtResults = dtResults.loc[dtResults['Season'].isin(seasonList)].copy()
    dsResults = dsResults.loc[dsResults['Season'].isin(seasonList)].copy()
    seeds = seeds.loc[seeds['Season'].isin(seasonList)].copy()
    ranks = ranks.loc[ranks['Season'].isin(seasonList)].copy()

    df_test = addTeamStats(dtResults, dsResults, df_test)

    seeds_T1, seeds_T2 = addSeedRankings(seeds)
    df_test = pd.merge(df_test, seeds_T1, on=['Season', 'Team1ID'], how='left')
    df_test = pd.merge(df_test, seeds_T2, on=['Season', 'Team2ID'], how='left')
    df_test['SeedDiff'] = df_test['T1_Seed'] - df_test['T2_Seed']
    for i in range(df_test['SeedDiff'].shape[0]):
        # A team with SeedDiff >= 10 has only happened once
        if df_test['SeedDiff'].iloc[i] >= 10:
            df_test['SeedDiff'].iloc[i] = 15
        if df_test['SeedDiff'].iloc[i] <= -10:
            df_test['SeedDiff'].iloc[i] = -15

    ranks_T1, ranks_T2 = addOrdinals(ranks)
    df_test = pd.merge(df_test, ranks_T1, on=['Season', 'Team1ID'], how='left')
    df_test = pd.merge(df_test, ranks_T2, on=['Season', 'Team2ID'], how='left')
    df_test['RankDiff'] = df_test['T1_Rank'] - df_test['T2_Rank']

    return df_test
##########


########## Load Stage 1 Data
def loadStage1Data():
    ''''
    MRegularSeasonCompactResults.csv

    Parameters
    ------------------------
    Season : (int64) The year the season occurs (2015-2016 season = 2016)
    DayNum : (int64) The day a game is played on (0-132)
    WTeamID : (int64) ID of the team that won
    WScore : (int64) Number of points the winning team scored
    LTeamID : (int64) ID of the team that lost
    LScore : (int64) Number of points the losing team scored
    WLoc : (object) Location of winning team (H = home, A = away, N = neutral)
    NumOT : (int64) The number of overtime periods (game with no overtime = 0)
    WFGM : (int64) field goals made
    WFGA : (int64) field goals attempted
    WFGM3 : (int64) three pointers made
    WFGA3 : (int64) three pointers attempted
    WFTM : (int64) free throws made
    WFTA : (int64) free throws attempted
    WOR : (int64) offensive rebounds
    WDR : (int64) defensive rebounds
    WAst : (int64) assists
    WTO : (int64) turnovers committed
    WStl : (int64) steals
    WBlk : (int64) blocks
    WPF : (int64) personal fouls committed
    ...
    ...
    The same stats are given for the losing team replacing the W with an L
    ------------------------
    '''
    sdResults = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv')
    '''
    MNCAATourneyDetailedResults.csv
    
    Parameters
    ------------------------
    Season : (int64) The year the season occurs (2015-2016 season = 2016)
    DayNum : (int64) The day a game is played on (0-132)
    WTeamID : (int64) ID of the team that won
    WScore : (int64) Number of points the winning team scored
    LTeamID : (int64) ID of the team that lost
    LScore : (int64) Number of points the losing team scored
    WLoc : (object) Location of winning team (H = home, A = away, N = neutral)
    NumOT : (int64) The number of overtime periods (game with no overtime = 0)
    WFGM : (int64) field goals made 
    WFGA : (int64) field goals attempted 
    WFGM3 : (int64) three pointers made 
    WFGA3 : (int64) three pointers attempted 
    WFTM : (int64) free throws made 
    WFTA : (int64) free throws attempted 
    WOR : (int64) offensive rebounds 
    WDR : (int64) defensive rebounds 
    WAst : (int64) assists 
    WTO : (int64) turnovers committed 
    WStl : (int64) steals 
    WBlk : (int64) blocks 
    WPF : (int64) personal fouls committed 
    ...
    ...
    The same stats are given for the losing team replacing the W with an L
    '''
    tdResults = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')
    '''
    MNCAATourneySeeds.csv
    
    Parameters
    ------------------------
    Season : (int64) The year the season occurs (2015-2016 season = 2016)
    Seed : (object) The first letter is the division and the number is the teams ranking
    TeamID : (int64) ID of a team 
    ------------------------
    '''
    tSeeds = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneySeeds.csv')
    '''
    MMasseyOrdinals.csv
    
    Parameters
    ------------------------
    Season : (int64) The year the season occurs (2015-2016 season = 2016)
    RankingDayNum : (int64) Day number the ranking was placed on
    SystemName : (object) Name of the system that ranked the team
    TeamID : (int64) ID of the given team
    OrdinalRank : (int64) The rank given to a team
    ------------------------
    '''
    sRanks = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage1/MMasseyOrdinals.csv')
    '''
    MSampleSubmissionStage1.csv
    
    Parameters
    ------------------------
    ID : (object) String stating the year, Team1ID, and Team2ID for each possible outcome
    Pred : (int64) The odds of Team1 winning the game
    ------------------------
    '''
    stage1SampleSubmission = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage1/MSampleSubmissionStage1.csv')
    return sdResults, tdResults, tSeeds, sRanks, stage1SampleSubmission
##########


########## Load Stage 2 Data
def loadStage2Data():
    ''''
    MRegularSeasonCompactResults.csv

    Parameters
    ------------------------
    Season : (int64) The year the season occurs (2015-2016 season = 2016)
    DayNum : (int64) The day a game is played on (0-132)
    WTeamID : (int64) ID of the team that won
    WScore : (int64) Number of points the winning team scored
    LTeamID : (int64) ID of the team that lost
    LScore : (int64) Number of points the losing team scored
    WLoc : (object) Location of winning team (H = home, A = away, N = neutral)
    NumOT : (int64) The number of overtime periods (game with no overtime = 0)
    WFGM : (int64) field goals made
    WFGA : (int64) field goals attempted
    WFGM3 : (int64) three pointers made
    WFGA3 : (int64) three pointers attempted
    WFTM : (int64) free throws made
    WFTA : (int64) free throws attempted
    WOR : (int64) offensive rebounds
    WDR : (int64) defensive rebounds
    WAst : (int64) assists
    WTO : (int64) turnovers committed
    WStl : (int64) steals
    WBlk : (int64) blocks
    WPF : (int64) personal fouls committed
    ...
    ...
    The same stats are given for the losing team replacing the W with an L
    ------------------------
    '''
    sdResults = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv')
    '''
        MNCAATourneyDetailedResults.csv

        Parameters
        ------------------------
        Season : (int64) The year the season occurs (2015-2016 season = 2016)
        DayNum : (int64) The day a game is played on (0-132)
        WTeamID : (int64) ID of the team that won
        WScore : (int64) Number of points the winning team scored
        LTeamID : (int64) ID of the team that lost
        LScore : (int64) Number of points the losing team scored
        WLoc : (object) Location of winning team (H = home, A = away, N = neutral)
        NumOT : (int64) The number of overtime periods (game with no overtime = 0)
        WFGM : (int64) field goals made 
        WFGA : (int64) field goals attempted 
        WFGM3 : (int64) three pointers made 
        WFGA3 : (int64) three pointers attempted 
        WFTM : (int64) free throws made 
        WFTA : (int64) free throws attempted 
        WOR : (int64) offensive rebounds 
        WDR : (int64) defensive rebounds 
        WAst : (int64) assists 
        WTO : (int64) turnovers committed 
        WStl : (int64) steals 
        WBlk : (int64) blocks 
        WPF : (int64) personal fouls committed 
        ...
        ...
        The same stats are given for the losing team replacing the W with an L
        '''
    tdResults = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv')
    '''
        MNCAATourneySeeds.csv

        Parameters
        ------------------------
        Season : (int64) The year the season occurs (2015-2016 season = 2016)
        Seed : (object) The first letter is the division and the number is the teams ranking
        TeamID : (int64) ID of a team 
        ------------------------
        '''
    tSeeds = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage2/MNCAATourneySeeds.csv')
    '''
    MMasseyOrdinals_thruDay128.csv

    Parameters
    ------------------------
    Season : (int64) The year the season occurs (2015-2016 season = 2016)
    RankingDayNum : (int64) Day number the ranking was placed on
    SystemName : (object) Name of the system that ranked the team
    TeamID : (int64) ID of the given team
    OrdinalRank : (int64) The rank given to a team
    ------------------------
    '''
    sRanks = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage2/MMasseyOrdinals_thruDay128.csv')
    '''
    MSampleSubmissionStage2.csv

    Parameters
    ------------------------
    ID : (object) String stating the year, Team1ID, and Team2ID for each possible outcome
    Pred : (int64) The odds of Team1 winning the game
    ------------------------
    '''
    stage2SampleSubmission = pd.read_csv('../mens-march-mania-2022/MDataFiles_Stage2/MSampleSubmissionStage2.csv')
    return tdResults, sdResults, tSeeds, sRanks, stage2SampleSubmission
##########


########## Generate and write training and test to csv
def generateTrainTest(stageNumber, year):
    if stageNumber == 1:
        tdResults, sdResults, tSeeds, sRanks, sampleSubmission = loadStage1Data()
    if stageNumber == 2:
        tdResults, sdResults, tSeeds, sRanks, sampleSubmission = loadStage2Data()
    else:
        print('This Stage does not exist, try again')

    # Removing Data from the first week and a half of the season since most teams don't play
    # to their actual skill level and some teams have easier games earlier in the season
    # From personal knowledge about sports, assumption
    sdResults = sdResults[sdResults['DayNum'] >= 20]

    train = createTrain(tdResults, sdResults, tSeeds, sRanks, 2003, year)
    test = createTest(tdResults, sdResults, tSeeds, sRanks, year, year,
                      sampleSubmission)

    # after looking at other submissions, these factors had minimal importance
    # and Dean Lewis book, Basketball on Paper
    col_toDrop = ['Reb', 'TO', 'PF', 'POSS']
    train = train.drop(columns={'T1' + col for col in col_toDrop})
    train = train.drop(columns={'T2' + col for col in col_toDrop})
    test = test.drop(columns={'T1' + col for col in col_toDrop})
    test = test.drop(columns={'T2' + col for col in col_toDrop})

    print(train)
    print(test)
    train.to_csv('train.csv', index=None)
    test.to_csv('test.csv', index=None)
##########

generateTrainTest(2, 2022)
