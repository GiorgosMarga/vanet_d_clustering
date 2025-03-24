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
	size := 500
	X := make([][][]float64, size)
	Y := make([][][]float64, size)
	trainSize := int(float64(size) * 0.8)

	for i := range size {
		X[i] = make([][]float64, 4)
		Y[i] = make([][]float64, 1)
		for j := range 4 {
			X[i][j] = []float64{float64(i + j)}
		}
		Y[i][0] = []float64{float64(i + 4)}
	}
	shuffleData(X, Y)
	sx := NewScaler()
	sx.Fit(X)
	X = sx.Transform(X)

	sy := NewScaler()
	sy.Fit(Y)
	Y = sy.Transform(Y)
	g := NewGRU(64, 4, MeanSquareError, 0.005)
	if err := g.Train(X[:trainSize], Y[:trainSize], 250, 20); err != nil {
		t.Error(err)
	}

	testX := X[trainSize:]
	testY := Y[trainSize:]
	fmt.Printf("[")
	for i := range testX {
		g.Input = testX[i]
		output, _ := g.forwardPass()
		if i == 0 {
			fmt.Printf("%f", sx.InverseTransform([][][]float64{output})[0][0][0])
		} else {
			fmt.Printf(",%f", sx.InverseTransform([][][]float64{output})[0][0][0])
		}
	}
	fmt.Println("]")
	fmt.Printf("[")
	for i := range testY {
		if i == 0 {
			fmt.Printf("%f", sy.InverseTransform([][][]float64{testY[i]})[0][0][0])
		} else {
			fmt.Printf(",%f", sy.InverseTransform([][][]float64{testY[i]})[0][0][0])
		}
	}
	fmt.Println("]")

	X = [][][]float64{{{8000}, {8001}, {8002}, {8003}}}
	predScaler := NewScaler()
	predScaler.Fit(X)
	predScaler.Transform(X)
	g.Input = X[0]
	output, _ := g.forwardPass()
	fmt.Printf("%f", predScaler.InverseTransform([][][]float64{output})[0][0][0])
}
