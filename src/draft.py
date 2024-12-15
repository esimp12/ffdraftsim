import typing as T

import pandas as pd
import numpy as np
from scipy.stats import geom

from league import Player, Team, create_player


class DraftStrategy:

    def draft_player(self, pool: T.Set[Player], team: Team, data_df: pd.DataFrame, next_pick: int) -> Player:
        """Draft a player from the pool.
        
        Args:
            pool - The pool of players to draft from.
            team - The team that is drafting the player.
            data_df - The DataFrame containing the player data.
            next_pick - The number of the next pick for the given team.
        
        Returns:
            The drafted player.
        """
        raise NotImplementedError
    

class ADPDraftStrategy(DraftStrategy):

    def __init__(self, adp_df: pd.DataFrame, randomness: float = 0.0):
        self.adp_df = adp_df
        self.temperature = 1 - randomness

    def draft_player(
        self, 
        pool: T.Set[str],
        team: Team,
        data_df: pd.DataFrame,
        next_pick: int,
    ) -> Player:
        """Draft a player from the pool using the ADP strategy.
        
        Args:
            pool - The pool of players to draft from.
            team - The team that is drafting the player.
            data_df - The DataFrame containing the player data.
            next_pick - The number of the next pick for the given team.
        
        Returns:
            The drafted player.
        """
        # Find available player slots for team
        available_slots = team.available_slots
        
        # Filter adps by available slots
        adp_df = self.adp_df[self.adp_df["position"].isin(available_slots)]

        # Filter adps by available players
        adp_df = adp_df[adp_df["player_display_name"].isin(pool)]

        # Filter adps by players with data
        players_with_data = data_df["player_display_name"].unique().tolist()
        adp_df = adp_df[adp_df["player_display_name"].isin(players_with_data)]

        # Sort by ADP
        adp_df = adp_df.sort_values("adp", ascending=True).reset_index(drop=True)
        max_offset = len(adp_df)-1
        min_idx = adp_df["adp"].idxmin()

        # Add "noisy" index to the ADP
        min_idx += min(max_offset, sample_geometric(self.temperature))
        
        # Find the player with the lowest adp
        player_row = adp_df.loc[min_idx]

        player = create_player(
            name=player_row["player_display_name"],
            df=data_df,
        )
        return player
    

def sample_geometric(p):
    if p == 1:
        return 0
    U = np.random.uniform(0, 1)
    X = np.ceil(np.log(1-U) / np.log(1-p))
    return X-1


class UrgencyScoreDraftStrategy(DraftStrategy):

    def __init__(self, adp_df: pd.DataFrame):
        self.adp_df = adp_df

    def draft_player(
        self,
        pool: T.Set[str],
        team: Team,
        data_df: pd.DataFrame,
        next_pick: int,
    ) -> Player:
        """Draft a player from the pool using the urgency score strategy.
        
        Args:
            pool - The pool of players to draft from.
            team - The team that is drafting the player.
            data_df - The DataFrame containing the player data.
            next_pick - The number of the next pick for the given team.
        
        Returns:
            The drafted player.
        """
        # Only get players from pool that have data
        df = data_df[data_df["player_display_name"].isin(pool)]

        # Only get players that are available to be drafted for team
        available_slots = team.available_slots
        df = df[df["position"].isin(available_slots)]

        available_players = df["player_display_name"].unique().tolist()

        # Find marginal sharpe ratio contribution to team for each player
        sharpe_ratios = []
        for player_name in available_players:
            player = create_player(
                name=player_name,
                df=df,
            )
            team.add_player(player)
            sharpe_ratios.append((player, team.sharpe_ratio))
            team.remove_player_by_name(player_name)
        
        # Find position scarcity for each position type
        position_sharpe_ratios = {pos: [] for pos in available_slots}
        for player, sharpe_ratio in sharpe_ratios:
            position_sharpe_ratios[player.position].append(sharpe_ratio)
        position_scarcity = {
            pos: np.std(srs) for pos, srs in position_sharpe_ratios.items()
        }

        # Find urgency score for each player
        urgency_scores = []
        for player, sharpe_ratio in sharpe_ratios:
            # Find the availability probability of the player
            adp_df = self.adp_df[self.adp_df["player_display_name"] == player.name]
            if adp_df.empty:
                adp = self.adp_df["adp"].max()
            else:
                adp = adp_df["adp"].iloc[0]
            p = 1 / adp
            # Probability that the player is not picked before the next pick
            availabitlity_probability = 1 - geom.cdf(next_pick, p)

            urgency_score = sharpe_ratio * position_scarcity[player.position] * (1 - availabitlity_probability)
            urgency_scores.append((player, urgency_score))

        player, _ = max(urgency_scores, key=lambda x: x[1])
        return player


class Draft:
    num_teams: int = 12
    num_rounds: int = 7

    def __init__(self, pool: T.Set[str], data_df: pd.DataFrame):
        self.pool = pool
        self.data_df = data_df

    @property
    def num_picks(self) -> int:
        """Get the total number of picks in the draft.
        
        Returns:
            The total number of picks in the draft.
        """
        return self.num_teams * self.num_rounds

    def simulate(
        self, 
        user_strategy: DraftStrategy,
        ai_strategy: DraftStrategy,
        user_pick: int,
    ) -> T.List[Team]:
        """Create a league of teams according to the given draft strategies.
        
        Args:
            user_strategy - The user strategy to use for drafting players.
            ai_strategy - The AI strategy to use for drafting players.
            user_pick - The pick number of the user.
        
        Returns:
            A list of drafted Teams.
        """
        # Create empty teams, keys are the picks, key == user_pick is the user's team
        teams = {
            i: Team() for i in range(1, self.num_teams + 1)
        }

        pick_round = 1
        for pick in range(1, self.num_picks+1):
            # Snake draft
            reverse = pick_round % 2 == 0

            # Find the team that is currently picking
            if not reverse:
                current_team = (pick - 1) % self.num_teams + 1
            else:
                current_team = (self.num_teams - (pick % self.num_teams)) % self.num_teams + 1

            is_user_pick = current_team == user_pick
            team = teams[current_team]
            next_pick = self._get_next_pick(current_team, pick_round)

            # Draft a player
            if is_user_pick:
                player = user_strategy.draft_player(self.pool, team, self.data_df, next_pick)
            else:
                player = ai_strategy.draft_player(self.pool, team, self.data_df, next_pick)
            
            # Add the player to the team and remove them from the pool
            team.add_player(player)
            self.pool.remove(player.name)

            # Update the current round
            if pick % self.num_teams == 0:
                pick_round += 1

        return list(teams.values())

    def _get_next_pick(self, team: int, round: int) -> int:
        next_round = round + 1
        reverse = next_round % 2 == 0

        possible_picks = list(range(round * self.num_teams + 1, next_round * self.num_teams + 1))
        if reverse:
            possible_picks = possible_picks[::-1]
        
        return possible_picks[(team - 1) % self.num_teams]