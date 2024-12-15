"""Module for logic and creation of leagues."""

from dataclasses import dataclass, field
import itertools as it
import typing as T

import numpy as np
import pandas as pd


@dataclass
class Game:
    """A single game for a player.
    
    Attributes:
        season - The season the game was played in.
        week - The week the game was played in.
        ppr_score - The PPR score the player received in the game.
    """
    season: int
    week: int
    ppr_score: float


@dataclass
class Player:
    """A NFL player in the league.
    
    Attributes:
        name - The name of the player.
        position - The position the player plays.
        games - The games the player has played all time.
    """
    name: str
    position: str
    games: T.List[Game] = field(default_factory=list)

    def get_games_by_season(self, season: int) -> T.List[Game]:
        """Return all games for the player in a given season.
        
        Args:
            season - The season to filter games by.
        
        Returns:
            A list of games for the player in the given season.
        """
        games = [game for game in self.games if game.season == season]
        # order by week
        return sorted(games, key=lambda x: x.week, reverse=False)

    @property
    def seasons_played(self) -> T.List[int]:
        """Return all seasons the player has played in.
        
        Returns:
            A list of seasons the player has played in.
        """
        seen = set()
        for game in self.games:
            seen.add(game.season)
        return list(seen)

    @property
    def scores(self) -> T.List[float]:
        """Return all PPR scores for the player.
        
        Returns:
            A list of PPR scores for the player.
        """
        return [game.ppr_score for game in self.games]

    @property
    def expected_return(self) -> float:
        """Return the expected return of the player.
        
        Returns:
            The mean PPR score of the player.
        """
        return np.mean(self.scores)
    
    @property
    def variance(self) -> float:
        """Return the variance of the player.

        Returns:
            The variance of the PPR scores of the player.
        """
        return np.var(self.scores)
    
    @property
    def risk(self) -> float:
        """Return the risk of the player.

        Returns:
            The standard deviation of the PPR scores of the player.
        """
        return np.sqrt(self.variance)


class Team:
    """A fantasy football team.

    Attributes:
        players - The players on the team.
    """

    _player_limits: T.Dict[str, int] = {
        "QB": 1,
        "RB": 2,
        "WR": 2,
        "TE": 1,
        "FLEX": 1,
        "TOTAL": 7,
    }

    def __init__(self):
        self._players: T.List[Player] = []
        # track via attribute versus player property as 'flex' is not a real position
        self._num_current_flex: int = 0 

    @property
    def available_slots(self) -> T.List[str]:
        """Return the available slots on the team.
        
        Returns:
            A list of available slots on the team.
        """
        qbs = self._get_num_players("QB")
        rbs = self._get_num_players("RB")
        wrs = self._get_num_players("WR")
        tes = self._get_num_players("TE")
        flex = self._num_current_flex

        slots = set()
        if qbs < self._player_limits["QB"]:
            slots.add("QB")
        if rbs < self._player_limits["RB"]:
            slots.add("RB")
        if wrs < self._player_limits["WR"]:
            slots.add("WR")
        if tes < self._player_limits["TE"]:
            slots.add("TE")
        if flex < self._player_limits["FLEX"]:
            slots.update(["RB", "WR", "TE"])
        return list(slots)

    @property
    def players(self) -> T.List[Player]:
        """Return the players on the team.
        
        Returns:
            A list of players on the team.
        """
        return self._players

    @property
    def expected_return(self) -> float:
        """Return the expected return of the team.
        
        Returns:
            The sum of the expected return of each player on the team.
        """
        return np.sum([p.expected_return for p in self.players])
    
    @property
    def variance(self) -> float:
        """Return the variance of the team.
        
        Returns:
            The sum of the variance of each player and the sum of the covariances
            between each pair of players.
        """

        player_variances = np.sum([p.variance for p in self.players])

        # Account for correlation between players
        player_covariances = 0.0
        for i in range(len(self.players)):
            for j in range(len(self.players)):
                if i == j:
                    continue
                player_x = self.players[i]
                player_y = self.players[j]
                player_covariances += covariance_estimate(player_x, player_y)
        return player_variances
    
    @property
    def risk(self) -> float:
        """Return the risk of the team.

        Returns:
            The standard deviation of the team's variance.
        """
        return np.sqrt(self.variance)
    
    @property
    def sharpe_ratio(self) -> float:
        """Return the Sharpe ratio of the team.
        
        Returns:
            The expected return divided by the risk.
        """
        return self.expected_return / self.risk

    def get_game_score(self, season: int, week: int, df: pd.DataFrame) -> float:
        """Return the game score of the team for a given season and week.
        
        Args:
            season - The season to get the game score for.
            week - The week to get the game score for.
            df - The DataFrame containing player data.
        
        Returns:
            The sum of the PPR scores of the players on the team for the given season and week.
        """
        points = 0.0
        for player in self.players:
            # NOTE: We can't use the player's games property as that's based on
            # historical games and we need the scores for the current drafting season
            score = get_player_game_score(player.name, season, week, df)
            points += score
        return points

    def add_player(self, player: Player):
        """Add a player to the team.

        Args:
            player - The player to add to the team.
        """
        slot = self._find_player_slot(player)
        if slot not in self._player_limits:
            return
        self._players.append(player)

    def remove_player_by_name(self, player_name: str):
        """Remove a player from the team.

        Args:
            player_name - The name of the player to remove from the team.
        """
        wrs_before = self._get_num_players("WR")
        rbs_before = self._get_num_players("RB")
        tes_before = self._get_num_players("TE")

        self._players = [p for p in self.players if p.name != player_name]
        
        # Update flex count if player was flex
        wrs_after = self._get_num_players("WR")
        rbs_after = self._get_num_players("RB")
        tes_after = self._get_num_players("TE")
        
        wrs_removed = wrs_before - wrs_after
        rbs_removed = rbs_before - rbs_after
        tes_removed = tes_before - tes_after

        if wrs_removed > 0 and wrs_after >= self._player_limits["WR"]:
            self._num_current_flex -= 1
        elif rbs_removed > 0 and rbs_after >= self._player_limits["RB"]:
            self._num_current_flex -= 1
        elif tes_removed > 0 and tes_after >= self._player_limits["TE"]:
            self._num_current_flex -= 1

    def _get_num_players(self, position: str) -> int:
        return len([p for p in self.players if p.position == position])

    def _find_player_slot(self, player: Player) -> str:
        # Too many players
        if len(self.players) >= self._player_limits["TOTAL"]:
            return "FULL"
        
        num_current_players = self._get_num_players(player.position)
        player_limit = self._player_limits[player.position]
        flex_limit = self._player_limits["FLEX"]

        # QB's cant be flex
        if num_current_players < player_limit:
            return player.position
        
        # Check if adding as flex player    
        if player.position != "QB":
            if self._num_current_flex < flex_limit:
                self._num_current_flex += 1
                return "FLEX"

        return "FULL"


@dataclass
class League:
    """A fantasy football league.

    Attributes:
        season - The season the league is being played in.
        teams - The teams in the league.
        user_team - The index of the user's team in the league teams.
        schedule - The schedule of the league. The keys of the dict
            represent the week of the league and the values are a list
            of tuples representing the matchups for that week.
    """
    season: int
    teams: T.List[Team]
    user_team: int
    schedule: T.Dict[int, T.List[T.Tuple[int, int]]] = field(default_factory=dict)

    @property
    def num_teams(self) -> int:
        """Return the number of teams in the league.
        
        Returns:
            The number of teams in the league.
        """
        return len(self.teams)

    def simulate_season(self, df: pd.DataFrame) -> T.Dict[int, T.Dict[str, T.Any]]:
        """Simulate the season for the league.

        Computes the results of the league for each week of the given season and
        tallies the total wins and losses for each team at the end of the season.
        Records metrics such as average points scored and standard deviation of points
        scored for each team.

        Args:
            df - The DataFrame containing player data.

        Returns:
            A dictionary containing the results of the league for the entire season.
        """
        # Simulate each week of the season
        weekly_results = {}
        for week, matchups in self.schedule.items():
            weekly_results[week] = [{
                "team": i+1,
                "w/l": 0,
                "points": 0.0,
            } for i in range(self.num_teams)]
            
            # Simulate each matchup of the week
            for matchup in matchups:
                a, b = matchup
                a = a - 1
                b = b - 1
                team_a = self.teams[a]
                team_b = self.teams[b]

                score_a = team_a.get_game_score(self.season, week+1, df)
                score_b = team_b.get_game_score(self.season, week+1, df)
                if score_a > score_b:
                    weekly_results[week][a]["w/l"] = 1
                    weekly_results[week][a]["points"] = score_a
                    weekly_results[week][b]["points"] = score_b
                    weekly_results[week][b]["w/l"] = 0
                else:
                    weekly_results[week][b]["w/l"] = 1
                    weekly_results[week][b]["points"] = score_b
                    weekly_results[week][a]["points"] = score_a
                    weekly_results[week][a]["w/l"] = 0

        # Compute season results
        season_results = {}
        for team in range(self.num_teams):
            season_results[team+1] = {
                "wins": 0,
                "losses": 0,
                "games_played": 0,
                "points": [],
                "total_points": 0.0,
                "average_points": 0.0,
                "std": 0.0,
            }
        for week, results in weekly_results.items():
            for result in results:
                team = result["team"]
                season_results[team]["wins"] += result["w/l"]
                season_results[team]["losses"] += 1 - result["w/l"]
                season_results[team]["points"].append(result["points"])
        for team, data in season_results.items():
            data["games_played"] = data["wins"] + data["losses"]
            data["total_points"] = np.sum(data["points"])
            data["average_points"] = np.mean(data["points"])
            data["std"] = np.std(data["points"])
        
        return season_results


def covariance_estimate(player_x: Player, player_y: Player) -> float:
    """Return the covariance estimate between two players.
    
    This estimates the covariance as follows: 
    
    1. Find all possible season combinations between the two players.

    2. Compute the covariance between the two players for each season combination.

    3. Return the weighted average of the covariance estimates.

    Args:
        player_x - The first player.
        player_y - The second player.
    
    Returns:
        The covariance estimate between the two players.
    """ 
    total = 0.0  
    n = 0
    seasons_x = player_x.seasons_played
    seasons_y = player_y.seasons_played
    for season_x, season_y in it.product(seasons_x, seasons_y):
        # Get games for each player in the given season
        games_x = player_x.get_games_by_season(season_x)
        games_y = player_y.get_games_by_season(season_y)

        # Note it is possible for two seasons to have different number of
        # games played, and so we correct for this by sampling the smaller
        # number of games of the two from both seasons
        num_games_x = len(games_x)
        num_games_y = len(games_y)
        min_games = min(num_games_x, num_games_y)
        max_games = max(num_games_x, num_games_y)
        sampled_games = np.random.permutation(max_games)[:min_games].tolist()
        if num_games_x <= num_games_y:
            games_y = [games_y[i] for i in sampled_games]
        else:
            games_x = [games_x[i] for i in sampled_games]

        # Compute covariance between the two players
        scores_x = [game.ppr_score for game in games_x]
        scores_y = [game.ppr_score for game in games_y]
        total += np.cov(scores_x, scores_y)[0, 1] # covariance is a 2x2 matrix
        n += 1
    return total / n


def get_player_game_score(player_name: str, season: int, week: int, df: pd.DataFrame) -> float:
    """Return the PPR score of a player in a given week of a season.
    
    Args:
        player_name - The name of the player.
        season - The season the game was played in.
        week - The week the game was played in.
        df - The DataFrame containing player data.
    
    Returns:
        The PPR score of the player in the given week of the season.
    """
    df = df[(df["player_display_name"] == player_name) & (df["season"] == season) & (df["week"] == week)]
    if df.empty:
        return None
    return df["fantasy_points_ppr"].iloc[0]


def create_random_teams(N: int, player_universe: T.List[str], df: pd.DataFrame) -> T.List[Team]:
    """Create N random teams only using players from the given universe.
    
    Args:
        N - The number of teams to create.
        player_universe - The list of players to choose from.
        df - The DataFrame containing player data.
    
    Returns:
        A list of N random teams.
    """
    qbs = df[df["position"] == "QB"]["player_display_name"].unique().tolist()
    rbs = df[df["position"] == "RB"]["player_display_name"].unique().tolist()
    wrs = df[df["position"] == "WR"]["player_display_name"].unique().tolist()
    tes = df[df["position"] == "TE"]["player_display_name"].unique().tolist()

    available_qbs = [name for name in player_universe if name in qbs]
    available_rbs = [name for name in player_universe if name in rbs]
    available_wrs = [name for name in player_universe if name in wrs]
    available_tes = [name for name in player_universe if name in tes]
    available_flex = rbs + wrs + tes
    
    teams = []
    for _ in range(N):
        team = Team()
        chosen_flex = []
        qb = np.random.choice(available_qbs)
        rb = np.random.choice(available_rbs, 2, replace=False).tolist()
        wr = np.random.choice(available_wrs, 2, replace=False).tolist()
        te = np.random.choice(available_tes)
        chosen_flex.extend(rb)
        chosen_flex.extend(wr)
        chosen_flex.append(te)
        flex = np.random.choice(available_flex)
        while flex in chosen_flex:
            flex = np.random.choice(available_flex)
        team.add_player(create_player(qb, df))
        team.add_player(create_player(rb[0], df))
        team.add_player(create_player(rb[1], df))
        team.add_player(create_player(wr[0], df))
        team.add_player(create_player(wr[1], df))
        team.add_player(create_player(te, df))
        team.add_player(create_player(flex, df))
        teams.append(team)
    return teams


def create_team(player_names: T.List[str], df: pd.DataFrame) -> Team:
    """Create a team from the given player names.

    Args:
        player_names - The names of the players to add to the team.
        df - The DataFrame containing player data.

    Returns:
        A team with the given players if valid.
    """
    team = Team()
    for name in player_names:
        player = create_player(name, df)
        if player is None:
            continue
        team.add_player(player)
    return team


def create_player(name: str, df: pd.DataFrame) -> Player:
    """Create a player from the given name.

    Args:
        name - The name of the player.
        df - The DataFrame containing player data.

    Returns:
        A player with the given name if valid.
    """
    df = df[df["player_display_name"] == name]
    if df.empty:
        return None
    games = []
    for _, row in df.iterrows():
        game = Game(
            season=row["season"],
            week=row["week"],
            ppr_score=row["fantasy_points_ppr"],
        )
        games.append(game)
    pos = df["position"].iloc[0]
    return Player(name=name, position=pos, games=games)


def create_league(
    season: int,
    user_team: int,
    teams: T.List[Team],
    num_weeks: int = 18,
    schedule_offset: int = 0,
) -> League:
    """Create a league with the given number of teams and weeks with
    a round robin schedule.

    Args:
        season - The season the league is being played in.
        user_team - The index of the user's team in the league teams. Note this is
            just the user's draft position in the league.
        teams - The teams in the league.
        num_weeks - The number of weeks in the league.
        schedule_offset - The offset to apply to the schedule. This is
            the initial number of cyclic permutation rotations to apply to the
            round robin scheduling. Helps to reduce bias in league scheduling.
        
    Returns:
        A league with N teams.
    """
    num_teams = len(teams)
    schedule = create_schedule(num_teams, num_weeks, schedule_offset)
    league = League(
        season=season,
        teams=teams,
        user_team=user_team,
        schedule=schedule,
    )
    return league


def create_schedule(num_teams: int, num_weeks: int, offset: int = 0) -> T.Dict[int, T.List[T.Tuple[int, int]]]:
    """Create a schedule for a league with the given number of teams.
    
    Args:
        num_teams - The number of teams in the league.
    
    Returns:
        A list of tuples representing the schedule.
    """
    fixed_team = 1
    even_teams = range(2, num_teams + 1, 2)
    odd_teams = range(num_teams - 1, 2, -2)

    # initial round robin schedule
    teams = list(even_teams) + list(odd_teams)

    # rotate right the teams an initial offset (for randomization)
    teams = teams[-offset:] + teams[:-offset]

    # rotate right the teams each week to create the schedule
    schedule = {}
    midpoint = int(num_teams / 2)
    for week in range(num_weeks):
        # generate schedule for the week
        right = teams[:midpoint]
        left = [fixed_team] + teams[midpoint:][::-1]
        schedule[week] = [(i, j) for i, j in zip(left, right)]
        # rotate right
        teams = [teams[-1]] + teams[:-1]
    return schedule