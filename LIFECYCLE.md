# AMoC Lifecycle

Entry point: `AMoCv4.analyze()` in `pipeline/orchestrator.py`.

## Setup

1. SpaCy splits the story into sentences.
2. Pronouns get resolved using the full story context.
3. Graph and counters are cleared.

## Per-Sentence Loop

### 1. Process the sentence
- **First sentence**: `handle_first_sentence_wrapper()` extracts explicit nodes, asks the LLM for relationships, and builds the initial graph.
- **Later sentences**: `handle_nonfirst_sentence_wrapper()` does the same but also attaches new concepts to carryover nodes via the LLM.

### 2. Post-processing 
`run_post_processing()` runs three things:
- **Semantic decay** — the LLM scores each active edge 0-3 for narrative relevance. Low scores reduce visibility.
- **Pruning** — edges below threshold get removed, node limits are enforced.
- **Cleanup** — dangling edges and nodes with no connections get removed.

### 3. Build per-sentence view
`build_per_sentence_view_wrapper()` builds a frozen `PerSentenceGraph` snapshot through `PerSentenceGraphBuilder`:

1. **Set explicit nodes** — nodes directly mentioned in the current sentence.
2. **Compute carryover nodes** — BFS starts from each explicit node and walks outward along active edges (edges where `active=True` and `visibility_score > 0`), up to `max_distance` hops. Nodes that are reachable but have no active edges themselves get excluded.
3. **Build the view** — `view_nodes` = explicit nodes ∪ carryover nodes. `view_edges` = all globally active edges where both endpoints are in `view_nodes`.
4. **Freeze** — the result is a `PerSentenceGraph` with `explicit_nodes`, `carryover_nodes`, `active_nodes`, `active_edges`.

### 4. Connectivity stabilization 

Implements rollback mechanism and general connecitivity stabilization in case of fragmentation.

The graphs grow organically and it stays connected radially to a main hub. However, fragments may occur after decay / pruning. Also, if a sentence does not have explicit nodes, then the plot must revert to the previous state.

It could be remove to reduce complexity if deemed that fragmentation after pruning is allowed. See CONNECTIVITY.md. 

`stabilize_connectivity_wrapper()` kicks in if the active subgraph is disconnected. It tries three things in order:
- Reactivate old cumulative edges along shortest paths
- Ask the LLM to generate bridging edges
- Add generic "relates_to" edges as a last resort

If none of that works, the sentence gets rolled back to the previous graph state.


### 5. Update projection state
`update_post_projection_state_wrapper()` figures out which nodes are newly inferred, which just got deactivated, and sorts them into plotting lists (explicit, carryover, inactive).

### 6. Output
- `plot_paper_graph_style_wrapper()` draws the graph + triplet overlay panel (if plotting is on).
- `capture_state_only_wrapper()` saves the snapshot for batch plotting (if collecting states).
- `capture_sentence_triplets_wrapper()` records this sentence's triplets.
- `finalize_run_outputs_wrapper()` returns the final triplets, per-sentence triplets, and activation matrices.

## Overview

```
story
  -> spaCy sentences -> pronoun resolution
    -> for each sentence:
        reset -> extract nodes -> LLM inference -> validate triplets
          -> add edges -> decay + prune -> build view
            -> [connectivity repair] -> update state -> plot/capture
  -> final triplets + matrices
```
