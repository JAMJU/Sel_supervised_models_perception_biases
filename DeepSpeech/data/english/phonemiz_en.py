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

parser.add_argument("--target-dir", type = str, default='CommonVoice_dataset/', help="Directory to store the dataset.")

# choose between 'espeak', 'espeak-mbrola', 'segments', or 'festival'

parser.add_argument("--backend", type = str, default ='espeak')


# fr or en-us  

parser.add_argument("--language", type = str, default ='en-us')

args = parser.parse_args()

######


def phoni(x,bck,lg,txt_dir,phon_dir) : 
    
    file_path = x

    file_name = os.path.splitext(os.path.basename(file_path))[0]

    with open(os.path.join(txt_dir, file_name + '.txt'),'r') as f :

        a = phonemize(f,backend=bck,language=lg)
        
        try : 

            res = a[0]
            
        except IndexError :
            
            print(res)
            
        else : 

            with open(os.path.join(phon_dir, file_name + '_phonemized' +'.txt'),'w') as p :
            
                p.write(res)
            

########


def phontotal(csv,bck,lg,txt_dir,phon_dir):
    
    
    df = pd.read_csv(csv)
    
    samples = df['samples']
    
    taille = len(samples)
    
    i=1
    
    for x in samples : 
        
        tit = ast.literal_eval(x)
        
        x = tit['transcript_path']
        
        phoni(x,bck,lg,txt_dir,phon_dir)
        
        print('Phonemization of {} / {}'.format(i,taille))
        
        i+=1
        

#######


def main():
    
    target_dir = args.target_dir
    
    manifest_file = args.manifest_file
    
    backend = args.backend
    
    language = args.language
    
    tipe = manifest_file[manifest_file.find('_')+1:manifest_file.find('_m')]
    
    tgt_dir = os.path.join(target_dir,tipe)

    txt_dir = os.path.join(tgt_dir, 'txt/')

    phon_dir = os.path.join(tgt_dir, 'phon/')
    
    if os.path.exists(phon_dir):
        
        print('Find existing folder {}'.format(phon_dir))
        
    else :
        
        print("Creating phone directory")
        
        os.makedirs(phon_dir,exist_ok=True)
        
        phontotal(manifest_file,backend,language,txt_dir,phon_dir)
        

if __name__ == "__main__":
    main()
        
        
        
        
        
        

        
      
        
        
