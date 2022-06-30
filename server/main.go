package main

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

const logSlots int = 15
const tokenByte int64 = 4096 // 32_768 / 8
const encryptSize int64 = 5_242_898

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

type User struct {
	id    uint
	inbox []Mail
}

type Mail struct {
	uuid string
}

func parse_uint_id(c *gin.Context, query string) uint {
	requestID, err := strconv.ParseUint(c.Param(query), 10, 32)
	if err != nil || requestID > math.MaxUint32 || requestID == 0 {
		c.Status(http.StatusBadRequest)
		return 0
	}
	return uint(requestID)
}

func float64ToByte(f float64) []byte {
	var buf bytes.Buffer
	err := binary.Write(&buf, binary.LittleEndian, f)
	check(err)
	return buf.Bytes()
}

func constructPublicKey(id uint) *rlwe.PublicKey {
	// file --> rlwe.PublicKey

	pkBinary, err := ioutil.ReadFile(fmt.Sprintf("%d.pk", id))
	check(err)

	pkTest := new(rlwe.PublicKey)
	err = pkTest.UnmarshalBinary(pkBinary)
	check(err)

	return pkTest
}

func constructRotationKey(id uint) *rlwe.RotationKeySet {
	// file --> rlwe.RotationKeySet
	rtBinary, err := ioutil.ReadFile(fmt.Sprintf("%d.rok", id))
	check(err)

	resRotationKey := new(rlwe.RotationKeySet)
	err = resRotationKey.UnmarshalBinary(rtBinary)
	check(err)
	return resRotationKey
}

func constructRelinearizationKey(id uint) *rlwe.RelinearizationKey {
	// file --> rlwe.RelinarizationKey
	relinBinary, err := ioutil.ReadFile(fmt.Sprintf("%d.rek", id))
	check(err)

	resRelinKey := new(rlwe.RelinearizationKey)
	err = resRelinKey.UnmarshalBinary(relinBinary)
	check(err)
	return resRelinKey
}

func constructEmbedding(id uint, uuid string) *ckks.Ciphertext {
	// file --> ckks.Ciphertext
	ctBinary, err := ioutil.ReadFile(fmt.Sprintf("%d_%s.ct", id, uuid))
	check(err)

	ctTest := new(ckks.Ciphertext)
	err = ctTest.UnmarshalBinary(ctBinary)
	check(err)

	return ctTest
}

func find_user(userbase []User, id uint) int {
	for i, u := range userbase {
		if u.id == id {
			return i
		}
	}
	return -1
}

func user_id_404(c *gin.Context, id uint) {
	c.String(http.StatusNotFound, "user (%d) does not exist", id)
}

func Float64frombytes(bytes []byte) float64 {
	bits := binary.LittleEndian.Uint64(bytes)
	float := math.Float64frombits(bits)
	return float
}

func inject_noise(params []complex128, noise_margin []complex128) []complex128 {
	len_params := len(params)
	w := make([]complex128, len_params)
	b := make([]byte, 8)
	for i, p := range params {
		_, err := rand.Read(b)
		check(err)
		b[7] = 0x3F
		b[6] |= 0xF0

		w[i] = p +
			noise_margin[i] * complex(Float64frombytes(b) - 1.5, 0)
	}
	return w
}

func main() {
	// common settings
	files, err := filepath.Glob(filepath.Join("probdiffs", "*.csv"))
	check(err)

	len_params := len(files)
	thresholds := make([]float64, len_params)
	for i, fn := range files {
		i0 := strings.Index(fn, "_")
		i1 := strings.LastIndex(fn, ".")
		t, err := strconv.ParseFloat(fn[i0+1:i1], 64)
		check(err)
		thresholds[i] = t
	}

	probdiffs := make([][]complex128, len_params)
	noise_margins := make([][]complex128, len_params)
	for i, fn := range files {
		data, err := os.Open(fn)
		check(err)

		probdiff := make([]complex128, 1<<logSlots)
		noise_margin := make([]complex128, 1<<logSlots)
		rdr := csv.NewReader(bufio.NewReader(data))
		rows, err := rdr.ReadAll()
		check(err)

		lineNum := 0
		for j, _ := range rows {
			temp_probdiff, err := strconv.ParseFloat(rows[j][0], 64)
			check(err)
			probdiff[lineNum] = complex(temp_probdiff, 0)
			temp_noise, err := strconv.ParseFloat(rows[j][1], 64)
			check(err)
			noise_margin[lineNum] = complex(temp_noise, 0)
			lineNum += 1
		}
		probdiffs[i] = probdiff
		noise_margins[i] = noise_margin
	}

	// server code
	r := gin.Default()
	specific_id := r.Group("/:id")
	userbase := []User{}
	r.GET("/ping", func(c *gin.Context) {
		c.String(http.StatusOK, "pong")
	})
	r.GET("/flush", func(c *gin.Context) {
		userbase = []User{}
		c.String(http.StatusOK, "flushed")
	})

	specific_id.POST("/rek", func(c *gin.Context) {
		id := parse_uint_id(c, "id")
		// given pubkey --> save at `id.rek`
		file, err := c.FormFile("rek")
		if err != nil {
			c.String(http.StatusBadRequest, "no relin key")
			return
		}
		c.SaveUploadedFile(file, fmt.Sprintf("./%d.rek", id))

		// if not in `userbase`, enroll
		i := find_user(userbase, id)
		if i < 0 {
			new_user := User{id, []Mail{}}
			userbase = append(userbase, new_user)
		}
		c.String(http.StatusOK, "ok")
	})

	specific_id.POST("/rok", func(c *gin.Context) {
		id := parse_uint_id(c, "id")
		// given pubkey --> save at `id.rok`
		file, err := c.FormFile("rok")
		if err != nil {
			c.String(http.StatusBadRequest, "no rotation key")
			return
		}
		c.SaveUploadedFile(file, fmt.Sprintf("./%d.rok", id))

		// if not in `userbase`, enroll
		i := find_user(userbase, id)
		if i < 0 {
			new_user := User{id, []Mail{}}
			userbase = append(userbase, new_user)
		}
		c.String(http.StatusOK, "ok")
	})

	specific_id.POST("/pk", func(c *gin.Context) {
		id := parse_uint_id(c, "id")
		// given pubkey --> save at `id.pk`
		file, err := c.FormFile("pk")
		if err != nil {
			c.String(http.StatusBadRequest, "no pubkey")
			return
		}
		c.SaveUploadedFile(file, fmt.Sprintf("./%d.pk", id))

		// if not in `userbase`, enroll
		i := find_user(userbase, id)
		if i < 0 {
			new_user := User{id, []Mail{}}
			userbase = append(userbase, new_user)
		}
		c.String(http.StatusOK, "ok")
	})
	specific_id.GET("/pk", func(c *gin.Context) {
		// return file `id.pk`
		query_id := parse_uint_id(c, "id")
		i := find_user(userbase, query_id)

		if i >= 0 {
			c.File(fmt.Sprintf("./%d.pk", query_id))
		} else {
			user_id_404(c, query_id)
		}
	})
	specific_id.POST("/send", func(c *gin.Context) {
		// random index; up to 256 different parameter sets
		b := make([]byte, 1)
		_, err := rand.Read(b)
		check(err)
		index := int(b[0])

		probdiff := probdiffs[index]
		threshold := thresholds[index]
		noise_margin := noise_margins[index]

		query_id := parse_uint_id(c, "id")
		i := find_user(userbase, query_id)
		if i < 0 {
			user_id_404(c, query_id)
			return
		}

		file, err := c.FormFile("ct")
		if err != nil {
			c.String(http.StatusBadRequest, "no embedding file")
			return
		}
		if file.Size != encryptSize && file.Size != tokenByte {
			c.String(http.StatusBadRequest, "inappropriate file")
			return
		}
		emb_uuid := uuid.New()
		filename := fmt.Sprintf("%d_%s.ct", query_id, emb_uuid)

		err = c.SaveUploadedFile(file, filename)
		if err != nil {
			c.String(http.StatusInternalServerError, "error saving file")
			return
		}

		new_mail := Mail{fmt.Sprintf("%s", emb_uuid)}
		rcv_user := userbase[i]
		rcv_user.inbox = append(rcv_user.inbox, new_mail)
		userbase[i] = rcv_user

		if file.Size == encryptSize {
			c.String(http.StatusOK, "start calculating HE")

			// context & params
			////////// Lattigo Setting //////////
			var defaultParam ckks.ParametersLiteral
			defaultParam = ckks.PN16QP1761
			params, err := ckks.NewParametersFromLiteral(defaultParam)
			check(err)

			context := new(Params)
			context.params = params
			context.kgen = ckks.NewKeyGenerator(context.params)
			context.pk = constructPublicKey(query_id)
			rotKey := constructRotationKey(query_id)
			context.rlk = constructRelinearizationKey(query_id)

			ct := constructEmbedding(query_id, fmt.Sprintf("%s", emb_uuid))
			context.evaluator = ckks.NewEvaluator(context.params, rlwe.EvaluationKey{Rlk: context.rlk})
			evaluator := context.evaluator.WithKey(rlwe.EvaluationKey{Rlk: context.rlk, Rtks: rotKey})
			context.encoder = ckks.NewEncoder(context.params)
			probdiff_plain := context.encoder.EncodeNTTAtLvlNew(
				context.params.MaxLevel(),
				inject_noise(probdiff, noise_margin),
				logSlots)

			// calc
			evaluator.MulRelin(ct, probdiff_plain, ct)

			for rot_time := 0; rot_time < logSlots; rot_time++ {
				tmp_ctxt := evaluator.RotateNew(ct, 1<<rot_time)
				evaluator.Add(ct, tmp_ctxt, ct)
			}
			evaluator.AddConst(ct, -math.Log(1/threshold-1), ct)
			// calc done
			// store
			marshalledCtxt, err := ct.MarshalBinary()
			check(err)

			err = ioutil.WriteFile(fmt.Sprintf("%d_%s.ctr", query_id, emb_uuid),
				marshalledCtxt, 0644)
			check(err)

			c.String(http.StatusOK, "done calculating HE")
		} else if file.Size == tokenByte {
			// %d_%s.ct --> vector v1
			v1, err := ioutil.ReadFile(fmt.Sprintf("%d_%s.ct", query_id, emb_uuid))
			check(err)

			// dot product
			var sum float64 = 0.
			for i := 0; i < (1 << logSlots); i += 8 {
				b := v1[i>>3]
				for j := 0; j < 8; j += 1 {
					sum += float64((b&0x80)>>7) * real(probdiff[i+j])
					b <<= 1
				}
			}

			// save %d_%s.ctr (64-bit, 8-Byte)
			err = ioutil.WriteFile(fmt.Sprintf("%d_%s.ctr", query_id, emb_uuid), float64ToByte(sum), 0644)
			check(err)
		}
	})

	specific_id.GET("/inbox/len", func(c *gin.Context) {
		query_id := parse_uint_id(c, "id")
		i := find_user(userbase, query_id)
		if i < 0 {
			user_id_404(c, query_id)
			return
		}
		c.String(http.StatusOK, fmt.Sprintf("%d", len(userbase[i].inbox)))
	})

	specific_id.GET("/inbox/:index", func(c *gin.Context) {
		query_id := parse_uint_id(c, "id")
		i := find_user(userbase, query_id)
		if i < 0 {
			user_id_404(c, query_id)
		}
		query_index := parse_uint_id(c, "index")
		inbox_len := uint(len(userbase[i].inbox))
		if query_index >= inbox_len {
			c.String(http.StatusBadRequest, "index out of bound")
			return
		}
		// stream file `ctr`
		emb_uuid := userbase[i].inbox[query_index].uuid
		c.File(fmt.Sprintf("./%d_%s.ctr", query_id, emb_uuid))
	})
	r.Run() // 8080
}
