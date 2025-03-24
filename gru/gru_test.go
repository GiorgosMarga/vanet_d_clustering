package gru

import (
	"fmt"
	"testing"
)

func TestSigmoid(t *testing.T) {
	if sigmoid(0) != 0.5 {
		t.Errorf("Expected 0.5 got %f\n", sigmoid(0))
	}
}

func TestMatrixMul(t *testing.T) {
	size := 4
	a := make([][]float64, size)
	b := make([][]float64, size)
	for i := range size {
		a[i] = make([]float64, size)
		b[i] = make([]float64, size)
		for j := 0; j < size; j++ {
			a[i][j] = float64((i + 1) * j)
			b[i][j] = float64((i + 1) * j)
		}
	}

	res := [4][4]float64{{0, 20, 40, 60}, {0, 40, 80, 120}, {0, 60, 120, 180}, {0, 80, 160, 240}}
	myRes := matrixMul(a, b)
	for row := range myRes {
		for col := range myRes {
			if res[row][col] != myRes[row][col] {
				t.Fatalf("%f-%f\n", res[row][col], myRes[row][col])
			}
		}
	}
}

func TestElementMatrixMul(t *testing.T) {
	a := [][]float64{{0.2}, {0.8}, {0.5}}
	b := [][]float64{{1.0}, {-0.5}, {0.3}}
	res := [][]float64{{0.2}, {-0.4}, {0.15}}
	myRes := elementMatrixMul(a, b)
	for row := range res {
		for col := range res[row] {
			if myRes[row][col] != res[row][col] {
				t.Fail()
			}
		}
	}
	// t.Error()

}

func TestTrain(t *testing.T) {
	X := make([][][]float64, 100)
	Y := make([][][]float64, 100)

	for i := range 100 {
		X[i] = make([][]float64, 4)
		Y[i] = make([][]float64, 1)
		for j := range 4 {
			X[i][j] = []float64{float64(i + j)}
		}
		Y[i][0] = []float64{float64(i + 4)}
	}
	s := NewScaler()
	s.Fit(X)
	X = s.Transform(X)
	Y = s.Transform(Y)
	g := NewGRU(256, 4, MeanSquareError, 0.0001)
	if err := g.Train(X[:len(X)-5], Y[:len(Y)-5], 100); err != nil {
		t.Error(err)
	}
	fmt.Println("Predictions:")

	testX := X[len(X)-5:]
	testY := Y[len(Y)-5:]
	for i := range testX {
		g.Input = testX[i]
		output, _ := g.forwardPass()
		fmt.Printf("Predicted: %v, Target: %v\n",
			s.InverseTransform([][][]float64{output}), s.InverseTransform([][][]float64{testY[i]}))
	}

	g.Input = X[25]
	output, _ := g.forwardPass()
	fmt.Printf("Predicted: %v, Target: %v\n",
		s.InverseTransform([][][]float64{output}), s.InverseTransform([][][]float64{Y[25]}))
}
