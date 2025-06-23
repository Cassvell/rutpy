# -*- coding: utf-8 -*-
"""

"""

## para ejecutar es python Plot_V1.py  arch1.m arch2.m yyyy-mm-dd HH:MM:SS
import paramiko

def get_files(initial_date, final_date, remote_path, localfilepath, select_fnames):
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='132.247.181.60', username='carlos', password='c4rl0s1s44c')
    
    
    sftp = ssh.open_sftp()
    sftp.chdir(remote_path)
    
    
    files = [] #lista de los archivos, ordenada del último al primero
    for entry in sorted(sftp.listdir_attr(),\
                        key=lambda k: k.st_mtime, reverse=True):
        files.append(entry.filename)   
    
    
    for entry in select_fnames:
        sftp.get(entry, localfilepath+entry) 
    
    return()

def list_names(daterange, string1, string2):
    select_fnames = []
    for i in daterange:
        tmp_name = string1+str(i)+string2
        select_fnames.append(tmp_name)
    return(select_fnames)
"""

"""

def get_file(remote_path, localfilepath, filename):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    available = True
    try:
        ssh.connect(hostname='132.247.181.60', username='carlos', password='c4rl0s1s44c')
        sftp = ssh.open_sftp()
        sftp.chdir(remote_path)

        # Check if the file exists on the server
        if filename in sftp.listdir():
            sftp.get(filename, localfilepath + filename)
            print(f"File '{filename}' downloaded successfully.")
        else:
            print(f"File '{filename}' not found on server.")
            available = False

    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        ssh.close()
    return available


def list_names(daterange, string1, string2):
    select_fnames = []
    for i in daterange:
        tmp_name = string1+str(i)+string2
        select_fnames.append(tmp_name)
    return(select_fnames)

       
