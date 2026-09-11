import sys
import paramiko

HOST = "100.66.1.7"
USER = "liuli"
PASS = "123456"

cmd = sys.argv[1] if len(sys.argv) > 1 else "echo ok"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=20,
               banner_timeout=20, auth_timeout=20)
stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
out = stdout.read().decode("utf-8", errors="replace")
err = stderr.read().decode("utf-8", errors="replace")
print(out, end="")
if err.strip():
    print("--- STDERR ---")
    print(err, end="")
client.close()