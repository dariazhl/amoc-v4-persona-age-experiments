# AMoC v4 — Persona Age Experiments

## How to run

### 1. Navigate to the project directory

```bash
cd /export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments
```

### 2. Pull latest changes

```bash
git pull
git status
```

### 3. Submit a job

Use scripts Qwen3-30B-A3B-Instruct-2507 or Llama-3.3-70B:

```bash
slurm_scripts/ 
```

By default, I have been using  Llama-3.3-70B on a handful of personas with:
```bash
slurm_scripts/run_llama70b_4gpu_small_example.sh
```

For a custom text (e.g. grade 4-5 texts):

```bash
sbatch slurm_scripts/run_llama70b_4gpu_small_example.sh tusa_text/grade_4_5.txt
```

For the original knight story (default):

```bash
sbatch slurm_scripts/run_llama70b_4gpu_small_example.sh
```

### 4. Check job status

Logs go to `exports/`. To inspect a running or completed job:

```bash
scontrol show jobid -dd <JOBNR>
```

### 5. Find the outputs

Results are saved to:

```
/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/small_example_output_llama/run_<JOBNR>/
```

Folder structure:
- **graphs/** — graphs 
-- reverse --> nodes in frozen positions, plotted backwards (**main point of interest**)
-- cumulative -> node position may shift 
- **matrix/** — activation matrices
- **triplets/** — extracted triplets (per-sentence and cumulative)

### 6. Export graphs using rsync
```bash
rsync -avz \
ana_daria.zahaleanu@fep.grid.pub.ro:/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/small_example_output_llama/run_129277/graphs/reverse_plots/ \
/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/results/...
```

```bash
rsync -avz \
ana_daria.zahaleanu@fep.grid.pub.ro:/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/small_example_output_llama/run_129277/graphs/cumulative_graph/ \
/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/results/...
```