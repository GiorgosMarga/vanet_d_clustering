package pythonneural

import (
	"encoding/json"
	"fmt"
	"net"
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

	X := make([][][]float64,500)
	Y := make([][][]float64,500)

	tx := make([][]float64,4)
	for i := range 500 {
		tx[0] = []float64{float64(i)}
		tx[1] = []float64{float64(i+1)}
		tx[2] = []float64{float64(i+2)}
		tx[3] = []float64{float64(i+3)}
		X[i] = tx
		Y[i] = [][]float64{{float64(i+4)}}
	}


	for _, x := range X{
		if len(x) != 4 {
			fmt.Println("here",x)
		}
	}
	dataMsg := ServerMessage{
		Type: SendData,
		Msg: map[string]any{
			"X": X,
			"Y": Y,
		},
	}

	if err := json.NewEncoder(conn).Encode(dataMsg); err != nil {
		return nil, err
	}

	return &PythonNeural{
		conn: conn,
	}, nil

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

	tx := make([][][]float64,100)
	for i := range 100{
		tx[i] = [][]float64{{float64(i+1000),float64(i+1001),float64(i+1002),float64(i+1003)}}
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
