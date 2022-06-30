import os
from config import dest_ip

os.system('curl {}/flush'.format(dest_ip))

os.system("go run receiver_keygen.go")
os.system('curl -X POST -F "pk=@pk" {}/1/pk'.format(dest_ip))
os.system('curl -X POST -F "rek=@rek" {}/1/rek'.format(dest_ip))
os.system('curl -X POST -F "rok=@rok" {}/1/rok'.format(dest_ip))
