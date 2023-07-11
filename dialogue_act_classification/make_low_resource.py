import random

def count_samples(fname):
    with open(fname) as f:
        lines = f.readlines()
        #header = lines[0]
        label2rows = dict()
        for line in lines[1:]:
            label = line.split(',')[-1].replace('\n','')
            if label in label2rows:
                label2rows[label] += 1
            else:
                label2rows[label] = 1
    print(label2rows)

def reduce_samples(fname, threshold):
    with open(fname) as f:
        lines = f.readlines()
        #header = lines[0]
        label2rows = dict()
        for line in lines[1:]:
            label = line.split(',')[-1]
            if label in label2rows:
                label2rows[label].append(line)
            else:
                label2rows[label] = [line]
    for label in label2rows:
        label2rows[label] = label2rows[label][:threshold]
    return label2rows    

def write_new_samples(new_fname, label2rows):
    with open(new_fname,'w') as f:
        f.write('id,speakers,tokens,tags\n')
        all_rows = []
        for label in label2rows:
            all_rows.extend(label2rows[label])
        random.shuffle(all_rows)
        for row in all_rows:
            f.write(row)

dtype = 'dev' # 'train' or 'dev'
threshold = 10 # max number of samples per category; 30 for train and 10 for dev    
fname = 'csv_original/'+dtype+'.csv'
new_fname = 'csv_da_annotations/csv_low_resource/'+dtype+'.csv'    

print('Original distribution:')
count_samples(fname)
label2rows = reduce_samples(fname, threshold)
write_new_samples(new_fname, label2rows)
print('New distribution:')
count_samples(new_fname)
