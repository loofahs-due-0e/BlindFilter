import os
import sys
from config import dest_ip

if len(sys.argv) != 2:
    print("usage : python3 demo_rcv.py [ round (0, 1, ...) ]")
    exit()

round = int(sys.argv[1])

len_inbox = 5

for i in range(len_inbox):
    os.system("curl {}/1/inbox/{} --output {}.ctr".format(
        dest_ip, i + 2 * len_inbox * round, i + 1))
    os.system("go run receiver_decrypt.go {}.ctr".format(i + 1))

print("-" * 120)
for i in range(len_inbox):
    os.system("curl {}/1/inbox/{} --output {}.ctr".format(
        dest_ip, i + 2 * len_inbox * round + len_inbox, i + 1 + len_inbox))
    os.system("go run receiver_decrypt.go {}.ctr".format(i + 1 + len_inbox))
