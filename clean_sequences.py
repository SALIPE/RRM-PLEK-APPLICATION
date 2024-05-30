import os
import subprocess


def clean_sequences(class_name,root,filename):

    command = f'seqkit rmdup -o {os.path.join(root, class_name+"_cleaned")} {os.path.join(root, filename)}' 
    print(command)
    subprocess.check_output(command, shell= True)
    