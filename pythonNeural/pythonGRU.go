package pythonneural

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"net"
	"os"
	"strconv"
	"strings"
)

const (
	SetWeights = iota
	GetWeights
	Train
	Evaluate
	Predict
	SendData
)

type ServerMessage struct {
	Type int `json:"type"`
	Msg  any `json:"msg"`
}

type GetWeightsResponseMessage struct {
	Weights [][][]float64
}

type PythonNeural struct {
	conn net.Conn
}

func NewPythonNeural(address string) (*PythonNeural, error) {
	conn, err := net.Dial("tcp", address)
	if err != nil {
		return nil, err
	}

	//

	return &PythonNeural{
		conn: conn,
	}, nil

}

func (pn *PythonNeural) SendData(X, Y [][][]float64, id int) error {
	dataMsg := ServerMessage{
		Type: SendData,
		Msg: map[string]any{
			"X":  X,
			"Y":  Y,
			"id": id,
		},
	}

	return json.NewEncoder(pn.conn).Encode(dataMsg)
}
func (pn *PythonNeural) SetWeights(weights [][][]float64) error {
	// send set weights message
	newMsg := ServerMessage{
		Type: SetWeights,
		Msg: map[string]any{
			"weights": weights,
		},
	}
	return json.NewEncoder(pn.conn).Encode(newMsg)
}
func (pn *PythonNeural) Predict(X [][]float64) ([][]float64, error) {

	tx := make([][][]float64, 100)
	for i := range 100 {
		tx[i] = [][]float64{{float64(i + 1000), float64(i + 1001), float64(i + 1002), float64(i + 1003)}}
	}
	// send predict message
	newMsg := ServerMessage{
		Type: Predict,
		Msg: map[string]any{
			"x": tx,
		},
	}
	err := json.NewEncoder(pn.conn).Encode(newMsg)
	//TODO: CHANGE THIS
	return nil, err
}
func (pn *PythonNeural) Train(epochs, batchSize int) error {
	// send train message
	newMsg := ServerMessage{
		Type: Train,
		Msg: map[string]any{
			"epochs":    epochs,
			"batchSize": batchSize,
		},
	}
	err := json.NewEncoder(pn.conn).Encode(newMsg)
	if err != nil {
		return err
	}

	b := make([]byte, 1024)
	_, err = pn.conn.Read(b)
	if err != nil {
		return err
	}

	fmt.Println("Finished Training")
	return nil
}

func (pn *PythonNeural) GetWeights() [][][]float64 {
	// send get weights message
	newMsg := ServerMessage{
		Type: GetWeights,
	}
	err := json.NewEncoder(pn.conn).Encode(newMsg)
	if err != nil {
		return nil
	}
	response := GetWeightsResponseMessage{}

	json.NewDecoder(pn.conn).Decode(&response)
	return response.Weights
}

func (pn *PythonNeural) Evaluate() error {
	// send evaluate message
	newMsg := ServerMessage{
		Type: Evaluate,
	}
	err := json.NewEncoder(pn.conn).Encode(newMsg)

	return err
}

func (pn *PythonNeural) ParseFile(filename string) ([][][]float64, [][][]float64, error) {
	f, err := os.ReadFile(filename)
	if err != nil {
		return nil, nil, err
	}

	lines := strings.Split(string(f[:len(f)-1]), "\n")
	var X [][][]float64
	var Y [][][]float64
	for line := range len(lines) {
		if line+5 > len(lines) {
			break
		}
		t := make([][]float64, 4)
		for i := range 4 {
			n, err := strconv.ParseFloat(lines[line+i], 64)
			if err != nil {
				panic(err)
			}
			t[i] = []float64{n}
		}
		X = append(X, t)
		n, err := strconv.ParseFloat(lines[line+4], 64)
		if err != nil {
			panic(err)
		}
		t2 := make([][]float64, 1)
		t2[0] = []float64{n}
		Y = append(Y, t2)
	}

	shuffleData(X, Y)

	// g.X = g.Sx.FitTransform(X)
	// g.Y = g.Sy.FitTransform(Y)

	return X, Y, nil

}

func shuffleData(X, Y [][][]float64) {
	if len(X) != len(Y) {
		panic("X and Y must have the same number of samples")
	}

	// Shuffle in place.
	rand.Shuffle(len(X), func(i, j int) {
		X[i], X[j] = X[j], X[i]
		Y[i], Y[j] = Y[j], Y[i]
	})
}
