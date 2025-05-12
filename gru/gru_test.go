package gru

import (
	"fmt"
	"testing"

	"github.com/GiorgosMarga/vanet_d_clustering/matrix"
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
	myRes := matrix.MatrixMul(a, b)
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
	myRes := matrix.ElementMatrixMul(a, b)
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
	size := 800
	X := make([][][]float64, size)
	Y := make([][][]float64, size)

	for i := range size {
		X[i] = make([][]float64, 4)
		Y[i] = make([][]float64, 1)
		for j := range 4 {
			X[i][j] = []float64{float64((i + 1) * j)}
		}
		Y[i][0] = []float64{float64(X[i][len(X[i])-1][0] + 1)}
	}

	shuffleData(X, Y)
	g := NewGRU(16, 4, 10, MeanSquareError, 0.005, 0.8)
	g.X = g.Sx.FitTransform(X)
	g.Y = g.Sy.FitTransform(Y)
	if err := g.Train(500, 20); err != nil {
		t.Error(err)
	}

	pred, actual, err := g.Evaluate()
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println(pred)
	fmt.Println(actual)

}

func TestIdentityMatrix(t *testing.T) {
	t1 := matrix.IdentityMatrix(10, 10)
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
	t1 := matrix.IdentityMatrix(10, 10)
	t2 := matrix.IdentityMatrix(10, 10)
	t3 := matrix.IdentityMatrix(10, 10)

	average := matrix.MatrixAverage([][][]float64{t1, t2, t3})

	for row := range average {
		for _, val := range average[row] {
			if val != 1 {
				t.FailNow()
			}
		}
	}

}

// func TestParseFile(t *testing.T) {
// 	for i := range 70 {
// 		g := NewGRU(1, 4, 10, MeanSquareError, 0.005)
// 		err := g.ParseFile(fmt.Sprintf("../data/car_%d.txt", i%60))
// 		if err != nil {
// 			fmt.Println(err)
// 			t.FailNow()
// 		}

// 		trainSize := int(float64(len(g.X)) * 0.8)
// 		if err := g.Train(g.X[:trainSize], g.Y[:trainSize], 2, 5); err != nil {
// 			t.Error(err)

// 		}
// 		// }
// 		// fmt.Printf("[")
// 		// for i := range testX {
// 		// 	g.Input = testX[i]
// 		// 	output, _ := g.forwardPass()
// 		// 	yPred[i] = g.sx.InverseTransform([][][]float64{output})[0][0]
// 		// 	if i == 0 {
// 		// 		fmt.Printf("%f ", yPred[i][0])
// 		// 	} else {
// 		// 		fmt.Printf(",%f ", yPred[i][0])
// 		// 	}
// 		// }
// 		// fmt.Printf("]\n[")
// 		// for i := range testY {
// 		// 	yActual[i] = g.sy.InverseTransform([][][]float64{testY[i]})[0][0]
// 		// 	if i == 0 {
// 		// 		fmt.Printf("%f", yActual[i][0])
// 		// 	} else {
// 		// 		fmt.Printf(", %f", yActual[i][0])
// 		// 	}
// 		// }
// 		// fmt.Printf("]\n")
// 	}
// }

func TestDFT(t *testing.T) {
	x := []float64{10.0, 22.0, 33.0, 44.0}

	fmt.Println(matrix.FFT(x))
}
