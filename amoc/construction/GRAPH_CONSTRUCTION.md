# Graph Construction

## sentence_builder.py — `SentenceGraphBuilder`

Builds the graph sentence by sentence.  

It extracts relationships from the text (both explicit and inferred), validates the resulting triplets with the LLM, normalizes node and edge labels, and links new nodes to nodes carried over from previous sentences.  

It handles both the initialization of the graph on the first sentence and the processing of all subsequent sentences.

## relationship_builder.py — `RelationshipGraphBuilder`

Processes relationships inferred by the LLM and adds them to the graph.
  
It validates the triplets, enforces limits on the number of concepts and properties per node, creates new nodes when needed, and ensures that explicit nodes are not pruned.