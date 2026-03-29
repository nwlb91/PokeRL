"""Parse Showdown team paste format into structured data.

Extracts species and move lists from team files so that full moveset
information can be encoded into battle observations (even for bench
pokemon that haven't revealed moves yet).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from poke_env.battle.move import Move


@dataclass
class ParsedPokemon:
    """Parsed Pokemon from a Showdown team paste."""
    species: str
    moves: List[str] = field(default_factory=list)  # move IDs (lowercase, no spaces)


@dataclass
class ParsedTeam:
    """Parsed team with species and movesets."""
    pokemon: List[ParsedPokemon] = field(default_factory=list)

    def get_move_objects(self, gen: int) -> List[List[Move]]:
        """Convert parsed move IDs to poke-env Move objects.

        Returns:
            List of 6 lists, each containing up to 4 Move objects.
        """
        result = []
        for mon in self.pokemon:
            moves = []
            for move_id in mon.moves[:4]:
                try:
                    moves.append(Move(move_id, gen=gen))
                except Exception:
                    pass  # skip moves poke-env doesn't recognize
            result.append(moves)
        return result


def _to_move_id(move_name: str) -> str:
    """Convert a Showdown move name to a poke-env move ID.

    Examples:
        'Dragon Dance' -> 'dragondance'
        'U-turn' -> 'uturn'
        'Freeze-Dry' -> 'freezedry'
        'Earth Power' -> 'earthpower'
    """
    return re.sub(r"[^a-z0-9]", "", move_name.lower())


def _to_species_id(species_line: str) -> str:
    """Extract a normalized species identifier from the first line of a set.

    The first line has format: 'Nickname (Species) @ Item' or 'Species @ Item'.
    """
    # Strip item
    line = species_line.split("@")[0].strip()
    # Check for nickname: 'Nickname (Species)' pattern
    paren_match = re.search(r"\(([^)]+)\)", line)
    if paren_match:
        species = paren_match.group(1).strip()
    else:
        species = line.strip()
    # Remove gender suffix like ' (M)' or ' (F)'
    species = re.sub(r"\s*\([MF]\)\s*$", "", species)
    return re.sub(r"[^a-z0-9]", "", species.lower())


def parse_team(team_str: str) -> ParsedTeam:
    """Parse a Showdown paste format team string.

    Args:
        team_str: Full team text in Showdown paste format.

    Returns:
        ParsedTeam with up to 6 pokemon and their moves.
    """
    team = ParsedTeam()
    # Split into individual Pokemon blocks (separated by blank lines)
    blocks = re.split(r"\n\s*\n", team_str.strip())

    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue

        species = _to_species_id(lines[0])
        moves = []
        for line in lines[1:]:
            if line.startswith("- "):
                move_name = line[2:].strip()
                moves.append(_to_move_id(move_name))

        team.pokemon.append(ParsedPokemon(species=species, moves=moves))

    return team
