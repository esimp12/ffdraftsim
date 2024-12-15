import time

import click
import pandas as pd
import numpy as np

from data import get_current_season_players, get_historical_data, get_adp_data
from league import create_random_teams, create_league
from draft import Draft, ADPDraftStrategy, UrgencyScoreDraftStrategy


@click.group()
def cli():
    pass


@cli.command()
def frontier(N: int = 10000, draft_season: int = 2023):
    # Load data
    player_universe = get_current_season_players(draft_season)
    df = pd.read_csv("../data/ppr_data_2013_2022.csv")
    
    # Generate N random teams and compute the expected return, risk, and Sharpe ratio
    teams = create_random_teams(N, player_universe, df)
    stats_df = pd.DataFrame(
        columns=["expected_return", "risk", "sharpe_ratio", "team_players"],
    )
    stats_df["team_players"] = [", ".join([p.name for p in team.players]) for team in teams]
    stats_df["expected_return"] = [team.expected_return for team in teams]
    stats_df["risk"] = [team.risk for team in teams]
    stats_df["sharpe_ratio"] = [team.sharpe_ratio for team in teams]
    stats_df.to_csv(f"../data/frontier_{draft_season}.csv", index=False)


@click.option("-d", "--draft-season", default=2023, help="The season to draft for.")
@click.option("-s", "--num-seasons", default=10, help="The number of historical seasons to include.")
@cli.command()
def get_ppr_data(
    draft_season: int = 2023,
    num_seasons: int = 10,
):
    seasons = [draft_season - i for i in range(1, num_seasons+1)]
    players = get_current_season_players(draft_season)
    df = get_historical_data(players, seasons)
    first_season = seasons[-1]
    last_season = seasons[0]
    df.to_csv(f"../data/ppr_data_{first_season}_{last_season}.csv", index=False)


@click.option("-p", "--user-pick", default=1, help="The pick number of the user.")
@click.option("-d", "--draft-season", default=2023, help="The season to draft for.")
@click.option("-s", "--num-seasons", default=10, help="The number of historical seasons to include.")
@cli.command()
def draft(user_pick: int = 1, draft_season: int = 2023, num_seasons: int = 10):
    historical_df = pd.read_csv(f"../data/ppr_data_{draft_season - num_seasons}_{draft_season-1}.csv")
    season_df = pd.read_csv(f"../data/ppr_data_{draft_season}_{draft_season}.csv")
    adp_df = get_adp_data(f"../data/adp_{draft_season}.json")

    players = season_df["player_display_name"].unique().tolist()
    draft = Draft(pool=set(players), data_df=historical_df)
    user_strategy = UrgencyScoreDraftStrategy(adp_df)
    ai_strategy = ADPDraftStrategy(adp_df, randomness=0.1)
    teams = draft.simulate(user_strategy, ai_strategy, user_pick)

    # Print the teams
    print("Draft Results:")
    for i, team in enumerate(teams):
        print(f"Team {i+1} ({'User' if i == user_pick-1 else 'AI'}):")
        print(", ".join([p.name for p in team.players]))
        print(f"Expected Return: {team.expected_return}")
        print(f"Risk: {team.risk}")
        print(f"Sharpe Ratio: {team.sharpe_ratio}")
        print()

    # Simulate the season
    league = create_league(draft_season, user_pick, teams)
    season_results = league.simulate_season(season_df)
    print("Season Results:")
    for i, stats in season_results.items():
        print(f"Team {i} ({'User' if i == user_pick else 'AI'}):")
        print(f"Wins: {stats['wins']}")
        print(f"Losses: {stats['losses']}")
        print(f"Points: {stats['total_points']}")
        print(f"Avg. Points Per Game: {stats['average_points']}")
        print(f"Stdev. Points: {stats['std']}")
        print()


@click.option("-n", "--num_trials", default=2, help="The number of trials to run.")
@click.option("-d", "--draft-season", default=2023, help="The season to draft for.")
@click.option("-s", "--num-seasons", default=10, help="The number of historical seasons to include.")
@click.option("-u", "--strategy", default="urgency", help="The user's draft strategy.")
@click.option("-r", "--adp-randomness", default=0.1, help="The randomness in the AI's draft strategy.")
@cli.command()
def simulate(
    num_trials: int = 2, 
    draft_season: int = 2023,
    num_seasons: int = 10,
    strategy: str = "urgency",
    adp_randomness: float = 0.1,
):
    historical_df = pd.read_csv(f"../data/ppr_data_{draft_season - num_seasons}_{draft_season-1}.csv")
    season_df = pd.read_csv(f"../data/ppr_data_{draft_season}_{draft_season}.csv")
    adp_df = get_adp_data(f"../data/adp_{draft_season}.json")

    players = season_df["player_display_name"].unique().tolist()

    user_strategy = (
        UrgencyScoreDraftStrategy(adp_df) 
        if strategy == "urgency" 
        else ADPDraftStrategy(adp_df, randomness=adp_randomness)
    )
    ai_strategy = ADPDraftStrategy(adp_df, randomness=adp_randomness)

    # Track results per trial for user team
    results_df = pd.DataFrame(
        data={
            "trial": [],
            "user_pick": [],
            "team_players": [],
            "expected_return": [],
            "risk": [],
            "sharpe_ratio": [],
            "wins": [],
            "losses": [],
            "total_points": [],
            "average_points": [],
            "std": [],
        },
    )
    for i in range(num_trials):
        print(f"Trial {i+1}/{num_trials}...")
        start = time.time()
        draft = Draft(pool=set(players), data_df=historical_df)

        user_pick = np.random.randint(1, draft.num_teams)
        schedule_offset = np.random.randint(0, draft.num_teams - 1)
        
        # Draft teams
        print(f"Drafting teams (User Pick = {user_pick})...")
        teams = draft.simulate(user_strategy, ai_strategy, user_pick)
        user_team = teams[user_pick-1]

        # Simulate the season
        print("Simulating season...")
        league = create_league(draft_season, user_pick, teams, schedule_offset=schedule_offset)
        season_results = league.simulate_season(season_df)

        # Track results
        user_results = season_results[user_pick]
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    data={
                        "trial": [i],
                        "user_pick": [user_pick],
                        "team_players": [", ".join([p.name for p in user_team.players])],
                        "expected_return": [user_team.expected_return],
                        "risk": [user_team.risk],
                        "sharpe_ratio": [user_team.sharpe_ratio],
                        "wins": [user_results["wins"]],
                        "losses": [user_results["losses"]],
                        "total_points": [user_results["total_points"]],
                        "average_points": [user_results["average_points"]],
                        "std": [user_results["std"]],
                    },
                ),
            ],
            ignore_index=True,   
        )
        print(f"Trial {i+1}/{num_trials} - {time.time() - start:.2f}s.")

    results_df.to_csv(f"../data/simulation_results_{strategy}_{draft_season}.csv", index=False)


if __name__ == "__main__":
    cli()