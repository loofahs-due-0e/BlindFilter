package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

type Params struct {
	params      ckks.Parameters
	ringQ       *ring.Ring
	ringP       *ring.Ring
	prng        utils.PRNG
	encoder     ckks.Encoder
	kgen        rlwe.KeyGenerator
	sk          *rlwe.SecretKey
	pk          *rlwe.PublicKey
	rlk         *rlwe.RelinearizationKey
	encryptorPk ckks.Encryptor
	encryptorSk ckks.Encryptor
	decryptor   ckks.Decryptor
	evaluator   ckks.Evaluator
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func float64FromByte(b []byte) float64 {
	var f float64
	buf := bytes.NewReader(b)
	err := binary.Read(buf, binary.LittleEndian, &f)
	check(err)
	return f
}

const tokenByte int = 8 // 64-bit = 8-byte
const encryptSize int = 5_242_898

func main() {
	argsWithoutProg := os.Args[1:]

	ctBinary, err := ioutil.ReadFile(argsWithoutProg[0])
	check(err)
	if len(ctBinary) != encryptSize && len(ctBinary) != tokenByte {
		fmt.Println("invalid file size")
		return
	}

	var result float64

	if len(ctBinary) == tokenByte {
		result = float64FromByte(ctBinary)
	} else {
		var defaultParam ckks.ParametersLiteral
		defaultParam = ckks.PN16QP1761
		params, err := ckks.NewParametersFromLiteral(defaultParam)
		check(err)

		skBinary, err := ioutil.ReadFile("sk")
		check(err)

		resSecretKey := new(rlwe.SecretKey)
		err = resSecretKey.UnmarshalBinary(skBinary)
		check(err)

		ct := new(ckks.Ciphertext)
		err = ct.UnmarshalBinary(ctBinary)
		check(err)

		context := new(Params)
		context.sk = resSecretKey
		context.params = params

		context.encoder = ckks.NewEncoder(context.params)
		context.decryptor = ckks.NewDecryptor(context.params, context.sk)

		result_plain := context.decryptor.DecryptNew(ct)
		decrypted := context.encoder.Decode(result_plain, context.params.LogSlots())

		result = real(decrypted[0])
	}
	if result < 0. {
		fmt.Println("spam")
	} else {
		fmt.Println("ham")
	}
}
