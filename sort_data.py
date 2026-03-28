import pandas as pd
import re

def clean(csv_path, gff_path, dead_path, output_path):
    manual_map = {
        'hsa-miR-526a, hsa-miR-520c-5p, hsa-miR-518d-5p': (19, 54000000)
    }

    manual_drop_list = [
        'hsa-miR-3713', 'hsa-miR-1973', 'hsa-miR-3591-3p', 
        'hsa-miR-4456', 'hsa-miR-378g', 'hsa-miR-3591-5p'
    ]

    #load dead list
    dead_set = set()
    with open(dead_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[1].lower()
                dead_set.add(name)
                base_name = re.sub(r'-\d+$', '', name)
                if base_name != name:
                    dead_set.add(base_name)


    for d in manual_drop_list:
        dead_set.add(d.lower())

  

    coord_map = {}
    with open(gff_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9: continue
            chrom = parts[0].replace('chr', '')
            start = int(parts[3])
            
            if chrom == 'X': chrom_num = 23
            elif chrom == 'Y': chrom_num = 24
            elif chrom in ['M', 'MT']: chrom_num = 25
            else:
                try: chrom_num = int(chrom)
                except: chrom_num = 99

            attr_dict = {}
            for attr in parts[8].split(';'):
                if '=' in attr:
                    k, v = attr.split('=')
                    attr_dict[k] = v
            
            if 'Name' in attr_dict: coord_map[attr_dict['Name']] = (chrom_num, start)
            if 'Alias' in attr_dict:
                for alias in attr_dict['Alias'].split(','):
                    coord_map[alias] = (chrom_num, start)


    df = pd.read_csv(csv_path, index_col=0)
    features = [c for c in df.columns if c.lower() not in ['label', 'target', 'class']]
    
    mapped_cols = []   
    kept_unmapped = [] 
    dropped_dead = []  

    for col in features:
        if col in manual_map:
            c, s = manual_map[col]
            mapped_cols.append({'name': col, 'chr': c, 'start': s})
            print(f'Manually mapped "{col}" to Chr {c}')
            continue

        if col in coord_map:
            c, s = coord_map[col]
            mapped_cols.append({'name': col, 'chr': c, 'start': s})
            continue

        if ',' in col:
            first_part = col.split(',')[0].strip()
            if first_part in coord_map:
                c, s = coord_map[first_part]
                mapped_cols.append({'name': col, 'chr': c, 'start': s})
                continue

        base_name_map = col.replace('-5p', '').replace('-3p', '')
        lower_name_map = col.replace('miR', 'mir')
        found_loc = None
        if base_name_map in coord_map: found_loc = coord_map[base_name_map]
        elif lower_name_map in coord_map: found_loc = coord_map[lower_name_map]
            
        if found_loc:
            mapped_cols.append({'name': col, 'chr': found_loc[0], 'start': found_loc[1]})
            continue

        sub_names = [x.strip() for x in col.split(',')]
        is_dead = False
        for name in sub_names:
            clean_name = name.lower().replace('mir', 'mir')
            if clean_name in dead_set: is_dead = True
            elif clean_name.replace('-5p','').replace('-3p','') in dead_set: is_dead = True
        
        if is_dead:
            dropped_dead.append(col)
        else:
            kept_unmapped.append(col)


    df_mapped = pd.DataFrame(mapped_cols)
    if not df_mapped.empty:
        df_mapped = df_mapped.sort_values(by=['chr', 'start'])
        sorted_features = df_mapped['name'].tolist()
    else:
        sorted_features = []
        
    final_features = ['label'] + sorted_features + kept_unmapped
    
        
    df_final = df[final_features]
    # print(final_features)
    
    print('-' * 30)
    print('sumamry')
    print(f'Total feature number: {len(sorted_features + kept_unmapped)}')
    print(f'Total sample number: {len(df_final)}')
    print(f'mapped & sorted: {len(sorted_features)}')
    print(f'dropped:         {len(dropped_dead)}')
    print(f'remaining unmapped: {len(kept_unmapped)}')
    print('-' * 30)
    
    if len(kept_unmapped) > 0:
        print('Still unmapped:', kept_unmapped)

    df_final.to_csv(output_path, index=False)


clean('data/backgroundcorrected_idx.csv', 'data/hsa.gff3', 'data/miRNA.dead', 'data/sorted.csv')
print('sorting and cleaning done')