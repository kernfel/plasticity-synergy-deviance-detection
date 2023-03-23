# Short-term neuronal and synaptic plasticity act in synergy for deviance detection in spiking networks
Code for "Short-term neuronal and synaptic plasticity act in synergy for deviance detection in spiking networks", Kern &amp; Chao 2023

## Entry points
All code expects an environment with the dependencies listed in `requirements.txt`.

### Running simulations
Set up `conf/params.py` and `conf/<a config module>.py` first. The former describes the model and paradigm, while the latter controls details of the simulation. See `conf/isi5_500.py` for the example used in our paper.

Then, from the base directory, run `python isi.py conf/<a config module>.py` to run the requested simulations and save the raw data to disk.

### Raw data processing
The raw data is processed in various, very memory-intensive ways. These are scripted through the `process_*.py` scripts, which must all be run to display all figures.
The convenience script `process_all.py` does all of them in the necessary order, but may take a long time to complete.
Invoke each processing script with `python process_[...].py conf/<a config module>.py`.
Note that the configuration can represent a subset of the data present in order to do a partial analysis, e.g. with `N_networks` set to a lower number.
If you've run multiple templates or ISIs, process_all can select from these; to do so, run it as `python process_all.py <conf> <ISI-in-ms> <template-idx>`.
Absent such selection, the scripts will process the first ISI mentioned in the configuration, and the first template, only.
Processing multiple ISIs or templates simultaneously is currently not supported.

### Figures
Finally, run each of the `Fig_*.ipynb*` notebooks. Figure PDFs will be placed in `paper-1/`.
