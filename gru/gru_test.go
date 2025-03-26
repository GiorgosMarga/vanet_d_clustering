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
	size := 200
	X := make([][][]float64, size)
	Y := make([][][]float64, size)
	trainSize := int(float64(size) * 0.8)

	for i := range size {
		X[i] = make([][]float64, 4)
		Y[i] = make([][]float64, 1)
		for j := range 4 {
			X[i][j] = []float64{float64((i + 1) * j)}
		}
		Y[i][0] = []float64{float64(X[i][len(X[i])-1][0] + 1)}
	}

	fmt.Println(X[0], Y[0])
	fmt.Println(X[1], Y[1])
	fmt.Println(X[3], Y[2])
	shuffleData(X, Y)
	sx := NewScaler()
	sx.Fit(X)
	X = sx.Transform(X)

	sy := NewScaler()
	sy.Fit(Y)
	Y = sy.Transform(Y)
	g := NewGRU(128, 4, MeanSquareError, 0.005)
	if err := g.Train(X[:trainSize], Y[:trainSize], 500, 20); err != nil {
		t.Error(err)
	}

	testX := X[trainSize:]
	testY := Y[trainSize:]
	yPred := make([][]float64, len(testX))
	yActual := make([][]float64, len(testX))
	fmt.Printf("[")
	for i := range testX {
		g.Input = testX[i]
		output, _ := g.forwardPass()
		yPred[i] = sx.InverseTransform([][][]float64{output})[0][0]
		if i == 0 {
			fmt.Printf("%f ", yPred[i][0])
		} else {
			fmt.Printf(",%f ", yPred[i][0])
		}
	}
	fmt.Printf("]\n[")
	for i := range testY {
		yActual[i] = sy.InverseTransform([][][]float64{testY[i]})[0][0]
		if i == 0 {
			fmt.Printf("%f", yActual[i][0])
		} else {
			fmt.Printf(", %f", yActual[i][0])
		}
	}
	fmt.Printf("]\n")
	X = [][][]float64{{{80000}, {80001}, {80002}, {80003}}}
	predScaler := NewScaler()
	predScaler.Fit(X)
	predScaler.Transform(X)
	g.Input = X[0]
	output, _ := g.forwardPass()
	fmt.Printf("%f\n", predScaler.InverseTransform([][][]float64{output})[0][0][0])
	fmt.Println(R2Score(yActual, yPred) * 100)
}

func TestIdentityMatrix(t *testing.T) {
	t1 := identityMatrix(10, 10)
	if len(t1) != 10 || len(t1[0]) != 10 {
		t.FailNow()
	}
	for row := range len(t1) {
		for _, val := range t1[row] {
			if val != 1 {
				t.FailNow()
			}
		}
	}
}
func TestMatrixAverage(t *testing.T) {
	t1 := identityMatrix(10, 10)
	t2 := identityMatrix(10, 10)
	t3 := identityMatrix(10, 10)

	average := MatrixAverage([][][]float64{t1, t2, t3})

	for row := range average {
		for _, val := range average[row] {
			if val != 1 {
				t.FailNow()
			}
		}
	}

}
