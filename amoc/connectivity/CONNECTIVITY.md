# Connectivity for Rollback and Fragmentation

This is a module that could eventually be removed because it adds complexity to the workflow.

The graph typically maintains organic connectivity during normal growth. However, fragmentation may happen because of pruning and decay. Also, if a plot has 0 explicit nodes, it must rollback to previous state. This is where connectivity is needed.  

To disable connectivity, comment out this piece of code: 

```bash
  # Run repair pipeline before checking for dangling nodes
  if self._per_sentence_view is not None:
      self._connectivity_ops.run_repair_pipeline(
          per_sentence_view=self._per_sentence_view,
          prev_sentences=prev_sentences,
          current_sentence_text=self._current_sentence_text,
          normalize_edge_label_fn=self._normalize_edge_label,
          create_forced_edges_fn=self.create_forced_connectivity_edges_wrapper,
          persona=self.persona,
      )

      # Rebuild the view to reflect any edges added by repair
      self._per_sentence_view = self.build_per_sentence_view_wrapper(
          explicit_nodes=list(explicit_nodes),
          sentence_index=self._current_sentence_index,
      )
```

## repair.py — `ConnectivityRepair`

Handles basic graph connectivity.  
It finds disconnected parts of the graph and reconnects them by reactivating existing edges or computing the shortest paths between components.


## stabilizer.py — `ConnectivityStabilizer`

Manages connectivity during processing.  
For each sentence, it runs a repair pipeline:

1. Reactivate existing edges  
2. Generate new edges using the LLM  
3. If needed, add a fallback edge (`relates_to`)


## Wiring

- `repair.py` is attached to `Graph` in `graph.py` as  
  `self._stability_ops = ConnectivityRepair(self)`.

- `stabilizer.py` is attached to the orchestrator in `wiring.py` as  
  `core._connectivity_ops = ConnectivityStabilizer(...)`.

- During each sentence, `stabilize_connectivity_wrapper()` runs  
  `run_connectivity_pipeline()` and `repair_dangling_nodes()`.

- At the end of the run, if the graph is still disconnected, the stabilizer runs one more time.