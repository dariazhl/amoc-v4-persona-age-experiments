# Runtime

## per_sentence.py — `PerSentenceGraph`, `PerSentenceGraphBuilder`

Creates a frozen snapshot of the graph for each sentence.  
These snapshots represent the active state of the graph at that moment and cannot be modified afterward.

The module identifies explicit, carryover, and active nodes and edges by running a BFS from explicit nodes within distance limits.

## sentence_runtime.py — `SentenceRuntime`

Handles the runtime state for each sentence.  
This includes pronoun resolution, lemma extraction, resetting sentence-level state, and building the per-sentence graph view.

It acts as the bridge between raw text processing and graph operations.

## state_manager.py — `ProjectionStateManager`

Tracks how nodes activate and deactivate across sentences.  
It records activation matrices used by the landscape model, manages inactive nodes, and prepares data used for plotting.