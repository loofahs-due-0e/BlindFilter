# BlindFilter
## abstract
* Naïve Bayesian spam filtering
* with Homomorphic Encryption (using [Lattigo](https://github.com/ldsec/lattigo) implementation of CKKS)
* with [WordPiece embedding](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/tokenizer.json) (using [huggingface/tokenizers](https://github.com/huggingface/tokenizers/releases/tag/python-v0.11.0))
* on `enron` dataset from [MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data)
* using [gin](https://github.com/gin-gonic/gin) web framework


## tested on
* client (both sender & receiver)
    * MacBook Air (M1, 2020)
    * macOS 12.1(21C52)
    * Docker Desktop 4.3.2 (72729)
* server
    * [AMD Ryzen 7 3700X](https://www.amd.com/en/product/8446)
    * 64 GB DDR4 memory
    * Ubuntu 18.04.4 LTS
    * Docker 19.03.8 (afacb8b7f0)


## how it works
1. receiver
    * generates keys
    * posts its public key-set on server
        * public key `pk`, relinearization key `rek`, rotation key `rok`
2. sender
    * writes down some letter
    * in demo, random sample from `enron` dataset
    * tokenizes the letter
    * in demo, use already-tokenized value from `enron` dataset
    * make it one-or-zero(occur in letter or not) per embedding
    * gets public key from server
    * encrypt one-or-zero embedding
    * post result on server
3. receiver
    * get result from server
    * decrypt using secret key

## how to run
### common
* build container

`cd <PATH_TO>/BlindFilter`

Modify timezone in `dockerfile:4`. [list of timezones](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)

`docker build --tag=spam_matcher_env:1.0 .`

* you may want to enter server's IP address at `receiver/demo_keygen.py, receiver/demo_rcv.py, sender/demo_send.py`

### server
1. run container

`docker run -it --rm -p 8080:8080 -v <PATH_TO>/BlindFilter/server:/home spam_matcher_env:1.0`

This command connects port 8080 of host to 8080 of container. Make sure that
   *  host port 8080 is open or available to use
   *  container port 8080 lines up with `server/main.go`

2. `cd /home`

3. `tar -zxvf probdiffs.tar.gz`

4. `go run main.go`

### client - sender
1. ready dataset (for sample input)

`cd <PATH_TO>/BlindFilter/sender`
* on linux, `tar -zxvf enron.tar.gz`
* on mac, `gtar -zxvf enron.tar.gz`
    * install GNU-tar using [brew](https://brew.sh)
    * `brew update && brew install gnu-tar`

2. run container

`docker run -it --rm -v <PATH_TO>/BlindFilter/sender:/home spam_matcher_env:1.0`

3. `cd /home`

### client - receiver
1. run container

`docker run -it --rm -v <PATH_TO>/BlindFilter/receiver:/home spam_matcher_env:1.0`

2. `cd /home`

### demo
with server & two client containers running,
1. receiver  `python3 demo_keygen.py`
2. sender    `python3 demo_send.py [ plain | encrypt ]`
   * send 5 ham and 5 spam mails in plaintext | encrypted form
3. receiver  `python3 demo_rcv.py [ num_round (0, 1, ..) ]`
   * receive mails index of [num_round * 10, (num_round+1) * 10)


## result
* accuracy
    * accuracy 0.9681 (std.dev 0.0012)
    * F1 score 0.9690 (std.dev 0.0012)
* latency
    * key post
        * public key (9.1 M) : 531.485318 ms
        * relinearization key (289 M) : 10.996501728 s
        * rotation key (289 M) : 10.755876206 s
        * by the way, secret key size is 4.6 M
    * key get
        * public key : 413.843871 ms
    * embedding post
        * select random 10
        * POST + HE calculation + some filesystem access time
        * encrypted message : 5.1 M each
        * [1.407455663, 1.397351234, 1.339487997, 1.345956759, 1.347923028, 1.3614335, 1.337556796, 1.336377291, 1.529589613, 1.362279956] s
        * by the way, plaintext calculation takes...
            * [8.044751, 7.324642, 7.674607, 7.137246, 6.84202, 6.863009, 8.295167, 9.045825, 8.491323, 7.812885] ms
    * message get
        * calculation result : 5.1 M each
        * [257.904997, 219.306127, 260.559319, 231.457287, 216.552031, 193.210169, 240.853641, 226.510264, 184.619124, 237.10274] ms
        * by the way, plaintext get takes...
            * [2446.67, 121.038, 125.558, 61.949, 126.488, 104.878, 70.078, 119.688, 55.529, 54.749] µs

## about dataset
[MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data) provides `enron_spam_data.csv`

We corrected some format error on that to make `enron_spam_data_prep.csv`, which consists of 33715 e-mails (16545 ham + 17170 spam)
