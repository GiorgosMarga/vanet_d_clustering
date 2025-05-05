package pythonneural

import (
	"bytes"
	"encoding/json"
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

	dataMsg := ServerMessage{
		Type: SendData,
		Msg: map[string]any{
			"X": [][][]float64{{{1.0}, {2.0}, {3.0}, {4.0}}, {{5.0}, {6.0}, {7.0}, {8.0}}},
			"Y": [][][]float64{{{5.0}}, {{6.0}}},
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
	// send predict message
	newMsg := ServerMessage{
		Type: Predict,
		Msg: map[string]any{
			"x": X,
		},
	}
	b := new(bytes.Buffer)
	err := json.NewEncoder(b).Encode(newMsg)
	if err != nil {
		return nil, err
	}
	_, err = pn.conn.Write(b.Bytes())
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
	b := new(bytes.Buffer)
	err := json.NewEncoder(b).Encode(newMsg)
	if err != nil {
		return err
	}
	_, err = pn.conn.Write(b.Bytes())
	return err
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
	b := new(bytes.Buffer)
	err := json.NewEncoder(b).Encode(newMsg)
	if err != nil {
		return err
	}
	_, err = pn.conn.Write(b.Bytes())
	return err
}
