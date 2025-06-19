package pythonneural

import (
	"fmt"
	"testing"
)

func TestNetwork(t *testing.T) {
	n, err := NewPythonNeural(":5000", 10)
	if err != nil {
		fmt.Println(err)
		t.FailNow()
	}

	n.Train(10, 20)

	weights := n.GetWeights()
	if err := n.SetWeights(weights); err != nil {
		fmt.Println(err)
	}

	n.Train(10, 20)
	n.Evaluate()

	n.Predict([][]float64{{1000.0}, {1001.0}, {1002.0}, {1003.0}})
}
