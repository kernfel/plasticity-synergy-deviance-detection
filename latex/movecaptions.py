import sys

states = {k:v for v,k in enumerate(['none', 'pre_caption', 'post_caption', 'caption_title', 'caption', 'dump_buffer'])}

def recon_figure(infile, outfile):
    state = states['none']
    caption_tag = r'\caption{}'
    indent = ''
    buffer = []
    for i, line in enumerate(infile):
        if state == states['none']:
            if line.startswith(r'\begin{figure'):
                state = states['pre_caption']
        elif state == states['pre_caption']:
            if line.startswith(r'\end{figure'):
                print(f'Warning: Figure without empty caption tag near line {i}.', file=sys.stderr)
                state = states['none']
            if line.lstrip().startswith(caption_tag):
                state = states['post_caption']
                col = line.find(caption_tag)
                line = line[:col + len(caption_tag)-1] + '\n'
                indent = line[:col]
        elif state == states['post_caption']:
            if line.startswith(r'\end{figure'):
                state = states['caption_title']
            buffer.append(line)
            line = ''
        elif state == states['caption_title']:
            if line.startswith('\n'):
                state = states['dump_buffer']
            else:
                state = states['caption']
                if not line.lstrip().startswith(r'\textbf{Fig.'):
                    print(f'Warning: Expected postfixed figure caption with title on line {i}.', file=sys.stderr)
                else:
                    line = indent + indent + line[line.find('}')+1:].lstrip()
        elif state == states['caption']:
            if line.startswith('\n'):
                state = states['dump_buffer']
            else:
                line = indent + indent + line
        
        if state == states['dump_buffer']:
            state = states['none']
            outfile.write(indent + '}\n')
            for buffered_line in buffer:
                outfile.write(buffered_line)
            buffer = []
        
        outfile.write(line)


if __name__ == '__main__':
    infile = sys.argv[1]
    with open(infile, 'r') as i:
        if len(sys.argv) > 2:
            outfile = sys.argv[2]
            with open(outfile, 'w') as o:
                recon_figure(i, o)
        else:
            recon_figure(i, sys.stdout)
