from .params import params

# Number of networks to include:
N_networks = 30

# Number of sequence templates to run:
N_templates = 1

# Number of sequence templates that save dynamics data:
N_templates_with_dynamics = 1

# Toggle STD - always an iterable of 0 and/or 1
STDs = (0,1)

# Toggle TA - always an iterable of 0 and/or 1
TAs = (0,1)

# Sequence inter-trial intervals to run (ms):
ISIs = (500,)

# Output file locations
fbase = 'data/isi5-500/isi5_'
raw_fbase = fbase + 'net{net}_isi{isi}_STD{STD}_TA{TA}_templ{templ}'
fname = raw_fbase + '.h5'
netfile = fbase + 'net{net}.h5'
digestfile = fbase + '{kind}.h5'

# Iterable of stimulus names
stimuli = {key: j for j, key in enumerate('ABCDE')}

# Iterable of stimulus pairings (each pair is run as both std/dev and dev/std)
pairings=(('A','B'), ('C','E'))

# Network validation: minimum number of neurons spiking in response to each stimulus
minimum_active_fraction = .5
