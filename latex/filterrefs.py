import sys
import re


if __name__ == '__main__':
    texfile = sys.argv[1]
    bibfile = sys.argv[2]
    if not texfile.endswith('.tex') or not bibfile.endswith('.bib'):
        print('Use: filterrefs.py document.tex bibliography.bib')
        exit()

    keys = []
    pattern = r'\\citep?\{([^\}]+)\}'
    with open(texfile) as tex:
        doc = tex.read()
    for match in re.finditer(pattern, doc, re.MULTILINE):
        for key in match.group(1).split(','):
            key = key.strip()
            if key not in keys:
                keys.append(key)

    minbib = []
    visited_keys = []
    drop = 0
    with open(bibfile) as bib:
        for line in bib.readlines():
            if line.startswith('@') and '{' in line:
                key = line[line.find('{') + 1:].rstrip().rstrip(',')
                if key not in keys:
                    drop = -1
                elif key not in visited_keys:
                    visited_keys.append(key)
            
            if not drop:
                minbib.append(line)
            
            if drop:
                if line.startswith('}'):
                    drop = 1
                elif drop > 0:
                    drop -= 1
    
    with open(bibfile, 'w') as bib:
        bib.writelines(minbib)
    
    undefined_refs = set(keys) - set(visited_keys)
    if undefined_refs:
        print('\nUndefined references in tex:\n')
        for key in undefined_refs:
            print(key)
