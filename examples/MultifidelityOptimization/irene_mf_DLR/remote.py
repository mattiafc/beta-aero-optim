from paramiko import SSHClient, AutoAddPolicy, ProxyCommand
import os
import time

hostname_riemann = "riemann.dalembert.upmc.fr"
username_riemann = "mattiafc"
key_file_riemann = "~/.ssh/riemann"

hostname_irene = "irene-fr.ccc.cea.fr"
username_irene = "ciarlatm"
password_irene = "password"

fileName = "high_infill_1"
remote_dir = "/ccc/scratch/cont003/gen13457/ciarlatm/Irene/irene_parallel_optimization/"

# Copia il file usando scp tramite ProxyJump
print(f"Copiando {fileName}.txt su Irene...")
scp_cmd = f"sshpass -p '{password_irene}' scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ProxyCommand='ssh -i {key_file_riemann} -W %h:%p {username_riemann}@{hostname_riemann}' {fileName}.txt {username_irene}@{hostname_irene}:{remote_dir}"
os.system(scp_cmd)

# Connessione SSH per eseguire comandi
proxy_cmd = f"ssh -i {key_file_riemann} {username_riemann}@{hostname_riemann} -W {hostname_irene}:22"
proxy = ProxyCommand(proxy_cmd)

ssh = SSHClient()
ssh.set_missing_host_key_policy(AutoAddPolicy())
ssh.connect(
    hostname=hostname_irene,
    username=username_irene,
    password=password_irene,
    sock=proxy
)

print("Eseguendo setup e generazione runs...")
stdin, stdout, stderr = ssh.exec_command(f"cd {remote_dir} && rm -rf scripts/ {fileName} && python3 generate_runs.py -f {fileName}.txt -s LES -n 102")
print(stdout.read().decode())

print("Lanciando job master...")
stdin, stdout, stderr = ssh.exec_command(f"cd {remote_dir} && ccc_msub scripts/master_LES.job")
output = stdout.read().decode()
print(output)

# Estrai job ID dall'output
job_id = None
for line in output.split('\n'):
    if 'request' in line.lower() or 'submitted' in line.lower():
        parts = line.split()
        for part in parts:
            if part.isdigit():
                job_id = part
                break

if job_id:
    print(f"Job ID: {job_id}")
    print("Attendo completamento job...")
    
    # Polling per verificare se il job è terminato
    while True:
        stdin, stdout, stderr = ssh.exec_command(f"ccc_mstat -u {username_irene}")
        job_status = stdout.read().decode()
        
        if job_id not in job_status:
            print("Job completato!")
            break
        
        print("Job ancora in esecuzione, attendo 60 secondi...")
        time.sleep(60)
    
    print(f"\nScaricando directory {fileName}...")
    local_dest = f"./{fileName}"
    os.makedirs(local_dest, exist_ok=True)
    scp_download_cmd = f"sshpass -p '{password_irene}' scp -r -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ProxyCommand='ssh -i {key_file_riemann} -W %h:%p {username_riemann}@{hostname_riemann}' {username_irene}@{hostname_irene}:{remote_dir}{fileName}/* {local_dest}/"
    os.system(scp_download_cmd)
    print(f"Download completato in {local_dest}")
else:
    print("Errore: impossibile estrarre Job ID")

ssh.close()
