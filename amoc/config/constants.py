MAX_DISTANCE_FROM_ACTIVE_NODES = 2
MAX_NEW_CONCEPTS = 3
MAX_NEW_PROPERTIES = 3
CONTEXT_LENGTH = 1
# edge_visibility defines edge lifespan within the active graph
# Edges decay over time and become inactive when visibility reaches zero
# This is essential for building the active per-sentence graph
EDGE_VISIBILITY = 2
NR_RELEVANT_EDGES = 15
DEBUG = False


STORY_TEXT = "A young knight rode through the forest. The knight was unfamiliar with the country. Suddenly, a dragon appeared. The dragon was kidnapping a beautiful princess. The knight wanted to free the princess. The knight wanted to marry the princess. The knight hurried after the dragon. The knight and the dragon fought for life and death. Soon, the knight's armor was completely scorched. At last, the knight killed the dragon. The knight freed the princess. The princess was very thankful to the knight. The princess married the knight."

AGE_REGIMES = {
    "primary_school": (5, 10),
    "secondary_school": (11, 14),
    "high_school": (15, 18),
    "university_freshman": (17, 18),
}

BLUE_NODES = [
    "knight",
    "princess",
    "dragon",
    "forest",
    "armor",
    "beautiful",
    "scorched",
]
