# Pipeline

## orchestrator.py — `AMoCv4`

The main controller of the pipeline.  
It processes a story sentence by sentence and coordinates the key steps: graph construction, LLM calls, connectivity repair, decay, and output generation.


## engine.py — `AgeAwareAMoCEngine`

High-level wrapper that runs the AMoC pipeline for a single persona–story pair.  
It builds the persona description including age, initializes the orchestrator, and returns the results as final, sentence-level, and cumulative triplets.


## runner.py — `process_persona_csv`

Command-line entry point for batch processing.  
It reads a CSV file of personas, runs the AMoC pipeline for each one, saves results incrementally, supports checkpointing and resuming, and generates reverse graph plots.


## decay.py — `Decay`

Handles semantic edge decay and pruning in the graph.  
It uses LLM-based narrative relevance to decide which edges fade, get removed, or are reactivated.  
It also computes node distances and records activation matrices used by the landscape model.


## wiring.py — `wire_core_dependencies`

Sets up the system’s internal dependencies.  
This function creates all operational components and connects them to the orchestrator.  
