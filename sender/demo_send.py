import os
import random
import sys
from config import dest_ip

if len(sys.argv) != 2:
    print("usage : python3 demo_send.py [ plain | encrypt ]")
    exit()

len_inbox = 5

plain = True if sys.argv[1] == "plain" else False

if not plain:
    os.system("curl {}/1/pk --output 1.pk".format(dest_ip))

spam_index = [random.randrange(17170) + 1 for _ in range(len_inbox)]
ham_index = [random.randrange(16545) + 1 for _ in range(len_inbox)]

for i in ham_index:
    os.system("go run sender.go {} ham {}".format(
        "plain" if plain else "1.pk", i))
    os.system('curl -X POST -F ct=@ct {}/1/send'.format(dest_ip))

print("-" * 120)
for i in spam_index:
    os.system("go run sender.go {} spam {}".format(
        "plain" if plain else "1.pk", i))
    os.system('curl -X POST -F ct=@ct {}/1/send'.format(dest_ip))
