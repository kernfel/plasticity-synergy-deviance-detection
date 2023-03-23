import process_nspikes as nspikes
import process_contrasts as contrasts
import process_suppression as suppr
import process_stats as stats
import process_distance as distance


def process_to_disk(cfg, isi, templ):
    for module in (nspikes, contrasts, suppr, distance, stats):
        module.process_to_disk(cfg, isi, templ)


if __name__ == '__main__':
    import sys
    import importlib

    if len(sys.argv) > 1:
        conf = '.'.join(sys.argv[1].split('.')[0].split('/'))
    else:
        conf = 'conf.isi5_500'
    
    if len(sys.argv) > 2:
        isi = int(sys.argv[2])
    else:
        isi = None
    
    if len(sys.argv) > 3:
        templ = int(sys.argv[3])
    else:
        templ = 0

    cfg = importlib.import_module(conf)
    process_to_disk(cfg, isi, templ)
