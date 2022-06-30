FROM ubuntu:focal

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/GMT

RUN apt update && apt install golang git curl python3 -y

RUN go get golang.org/x/crypto/blake2b &&\
    go get github.com/ldsec/lattigo &&\
    go get github.com/gin-gonic/gin &&\
    go get github.com/google/uuid

RUN cd /root/go/src/github.com/ldsec/lattigo &&\
    git checkout v2.3.0 &&\
    cd .. &&\
    mv lattigo v2 &&\
    mkdir lattigo &&\
    mv v2 lattigo
COPY params.go /root/go/src/github.com/ldsec/lattigo/v2/ckks/params.go

