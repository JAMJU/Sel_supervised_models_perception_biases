import argparse
import csv
import os
import tarfile
import pandas as pd 
import json
from phonemizer import phonemize
import ast



#Arguments line commands 

parser = argparse.ArgumentParser(description='Phonemization of the CommonVoice dataset.')

parser.add_argument('manifest_file', type = str,help="Name of the manifest csv file")

parser.add_argument("--phone_target_dir", type = str,  help="Directory to store the phonemization")
parser.add_argument("--txt_dir", type = str,  help="Directory where txt are stored")

# choose between 'espeak', 'espeak-mbrola', 'segments', or 'festival'

parser.add_argument("--backend", type = str, default ='espeak')


# fr or en-us  

parser.add_argument("--language", type = str, default ='en-us', help='fr-fr or en-us')
parser.add_argument("--begin", type = int, default =0, help='where to begin in the list')

args = parser.parse_args()

######


def phoni(x,bck,lg,txt_dir,phon_dir) : 
    
    file_path = x

    file_name = os.path.splitext(os.path.basename(file_path))[0]

    with open(os.path.join(txt_dir, file_name + '.txt'),'r') as f :

        a = phonemize(f,backend=bck,language=lg)

        res = a[0]

        with open(os.path.join(phon_dir, file_name + '_phonemized' +'.txt'),'w') as p :
            
            p.write(res)
            

########


def phontotal(csv,bck,lg,txt_dir,phon_dir, beg = 0):
    
    
    df = pd.read_csv(csv)
    print(df)
    
    samples = df['samples']
    
    taille = len(samples)
    print('samples to convert:', taille)
    
    i=0
    
    for ind in df.index :
        if i < beg:
            i += 1
            continue
        x = df['transcript_path'][ind]
        if not os.path.isfile(os.path.join(phon_dir, os.path.splitext(os.path.basename(x))[0] + '_phonemized' +'.txt')):
            phoni(x,bck,lg,txt_dir,phon_dir)
        
        print('Phonemization of {} / {}'.format(i,taille))
        
        i+=1
        

#######


def main():
    
    target_dir = args.phone_target_dir
    text_dir = args.txt_dir
    
    manifest_file = args.manifest_file
    
    backend = args.backend
    
    language = args.language
    begin = args.begin
    
    #tipe = manifest_file[manifest_file.find('_')+1:manifest_file.find('_m')]

    txt_dir = text_dir

    phon_dir = target_dir
    
    if os.path.exists(phon_dir):
        
        print('Find existing folder {}'.format(phon_dir))
        phontotal(manifest_file, backend, language, txt_dir, phon_dir, beg=begin)
        
    else :
        
        print("Creating phone directory")
        
        os.makedirs(phon_dir,exist_ok=True)
        
        phontotal(manifest_file,backend,language,txt_dir,phon_dir, beg = begin )
        

if __name__ == "__main__":
    main()
        
        
        
        
        
        

        
      
        
        