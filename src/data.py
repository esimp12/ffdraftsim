"""Module for pulling and processing data."""

import typing as T
import json

import nfl_data_py as nfl
import pandas as pd


def get_current_season_players(season: int, positions: T.Optional[T.List[str]] = None) -> T.List[str]:
    """Get the list of players for a given season.
    
    Args:
        season - The season to get players for.
        positions - The positions to include.

    Returns:
        A list of player names for the given season.
    """
    if positions is None:
        positions = ["QB", "RB", "WR", "TE"]
    df = nfl.import_weekly_data(years=[season], columns=["player_display_name", "position"])
    df = df[df["position"].isin(positions)]
    return df["player_display_name"].unique().tolist()


def get_historical_data(players: T.List[str], seasons: T.List[str]) -> pd.DataFrame:
    """Get historical PPR data for the given players and seasons.
    
    Args:
        players - The list of players to get data for.
        seasons - The list of seasons to get data for.
    
    Returns:
        A DataFrame containing the PPR data for the given players and seasons.
    """
    cols = ["player_display_name", "season", "week", "fantasy_points_ppr", "position"]
    df = nfl.import_weekly_data(years=seasons, columns=cols)
    df = df[df["player_display_name"].isin(players)]
    preprocessed_df = pd.DataFrame(columns=cols)
    for name, grp_df in df.groupby("player_display_name"):
        preprocessed_df = pd.concat(
            [
                preprocessed_df,
                fill_missing_weeks(name, grp_df, seasons),
            ],
            ignore_index=True,
        )
    return preprocessed_df


def fill_missing_weeks(name: str, df: pd.DataFrame, seasons: T.List[int]) -> pd.DataFrame:
    """Fill in missing weeks for a player in a given DataFrame.

    Missing weeks are filled in with 0.0 fantasy points.
    
    Args:
        name - The player's name.
        df - The DataFrame containing the player's data.
        seasons - The list of seasons to fill in.

    Returns:
        A DataFrame for a player with missing weeks filled in.
    """
    player_df = df[df["player_display_name"] == name].reset_index(drop=True)
    pos = player_df["position"].iloc[0]
    for season in seasons:
        if season not in player_df["season"].unique():
            continue

        last_week = 19 if season >= 2021 else 18
        player_df = player_df[player_df["week"] <= last_week] # disregard playoff data
        player_season_df = player_df[player_df["season"] == season]
        weeks_not_played = set(i for i in range(1, last_week)) - set(player_season_df["week"].tolist())
        weeks_not_played_df = pd.DataFrame(data={
            "player_display_name": [name for _ in weeks_not_played],
            "season": [season for _ in weeks_not_played],
            "week": [week for week in weeks_not_played],
            "fantasy_points_ppr": [0.0 for _ in weeks_not_played],
            "position": [pos for _ in weeks_not_played],
        })
        player_df = pd.concat(
            [
                player_df,
                weeks_not_played_df,
            ],
            ignore_index=True,
        ).sort_values(["season", "week"]).reset_index(drop=True)
    return player_df


def get_adp_data(filename: str, positions: T.Optional[T.List[str]] = None) -> pd.DataFrame:
    """Get ADP data for the current season.
    
    Returns:
        A DataFrame containing the ADP data for the current season.
    """
    if positions is None:
        positions = ["QB", "RB", "WR", "TE"]
    with open(filename, "r") as f:
        data = json.load(f)
    players = data["players"]
    df = pd.DataFrame(columns=["player_display_name", "adp", "position", "stdev"])
    player_names = []
    adps = []
    player_positions = []
    stdevs = []
    for player in players:
        pos = player["position"]
        if pos not in positions:
            continue
        player_names.append(player["name"])
        adps.append(player["adp"])
        player_positions.append(pos)
        stdevs.append(player["stdev"])
    df["player_display_name"] = player_names
    df["adp"] = adps
    df["position"] = player_positions
    df["stdev"] = stdevs
    return df