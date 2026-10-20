"""Maps an inventory table (in data/inventories.db / data/sjts.db) to the list
of concept/dimension names it covers. Concept names here are passed straight
through psychometric_utils.concept_to_dimension() and matched against the
dimension column, so for mpi120 they're just the OCEAN trait names as they
appear in that column (see `select distinct dimension from mpi120`).
"""

inventory_to_dimensions = {
    "mpi120": ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"],
}
