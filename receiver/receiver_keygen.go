package main

import (
	"io/ioutil"

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

func genParams(defaultParam ckks.Parameters, hw int) (
	context *Params, err error) {
	//

	context = new(Params)
	context.params = defaultParam
	context.kgen = ckks.NewKeyGenerator(context.params)

	if hw == 0 {
		context.sk, context.pk = context.kgen.GenKeyPair()
	} else {
		context.sk, context.pk = context.kgen.GenKeyPairSparse(hw)
	}

	context.ringQ = defaultParam.RingQ()
	if context.params.PCount() != 0 {
		context.ringP = defaultParam.RingP()
		context.rlk = context.kgen.GenRelinearizationKey(context.sk, 2)
	}

	if context.prng, err = utils.NewPRNG(); err != nil {
		return nil, err
	}

	context.encoder = ckks.NewEncoder(context.params)
	context.encryptorPk = ckks.NewEncryptor(context.params, context.pk)
	context.encryptorSk = ckks.NewEncryptor(context.params, context.sk)
	context.decryptor = ckks.NewDecryptor(context.params, context.sk)
	context.evaluator = ckks.NewEvaluator(context.params, rlwe.EvaluationKey{Rlk: context.rlk})

	return context, nil
}

func main() {
	////////// Lattigo Setting //////////
	// settingStartTime := time.Now()

	var defaultParam ckks.ParametersLiteral
	defaultParam = ckks.PN16QP1761

	params, err := ckks.NewParametersFromLiteral(defaultParam)
	check(err)

	var context *Params
	context, err = genParams(params, 0)
	check(err)

	rots := []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}
	rotKey := context.kgen.GenRotationKeysForRotations(rots, true, context.sk)

	// save pk
	marshalledPk, err := context.pk.MarshalBinary()
	check(err)

	err = ioutil.WriteFile("./pk", marshalledPk, 0644)
	check(err)

	// store rotKey
	marshalledRk, err := rotKey.MarshalBinary()
	check(err)

	err = ioutil.WriteFile("./rok", marshalledRk, 0644)
	check(err)

	// store relin key
	marshalledRelinKey, err := rotKey.MarshalBinary()
	check(err)

	err = ioutil.WriteFile("./rek", marshalledRelinKey, 0644)
	check(err)

	// store secret key
	marshalledSecretKey, err := context.sk.MarshalBinary()
	check(err)

	err = ioutil.WriteFile("./sk", marshalledSecretKey, 0644)
	check(err)
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}
