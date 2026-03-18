# Admission Modules

**edge_admission.py**  
Responsible for creating edges and deciding whether they can be added to the graph.  
It also checks for duplicates, ensures connectivity when needed, and records accepted edges.

**node_admission.py**  
Controls how nodes enter the graph.  
It checks whether a node already exists and whether it can be attached to the current structure.

**node_validation.py**  
Validates nodes before they are added.  
Checks include lemma length, allowed characters, story grounding, and provenance.

**text_normalizer.py**  
Standardizes the text used for nodes and edges so they follow a consistent format.

**triplet_deduplicator.py**  
Identifies and removes duplicate triplets.

**triplet_validator.py**  
Checks that triplets are both syntactically correct and semantically meaningful.