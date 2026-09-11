import sys
import paramiko

HOST = "100.66.1.7"
USER = "liuli"
PASS = "123456"

local = sys.argv[1]
remote = sys.argv[2]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=20)
sftp = client.open_sftp()
sftp.put(local, remote)
sftp.close()
print("uploaded:", local, "->", remote)
client.close()