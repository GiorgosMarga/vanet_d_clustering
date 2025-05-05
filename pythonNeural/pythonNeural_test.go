package pythonneural

import (
	"fmt"
	"testing"
)

func TestNetwork(t *testing.T) {
	n, err := NewPythonNeural(":5000")
	if err != nil {
		fmt.Println(err)
		t.FailNow()
	}
	weights := n.GetWeights()
	if err := n.SetWeights(weights); err != nil {
		fmt.Println(err)
	}
}
