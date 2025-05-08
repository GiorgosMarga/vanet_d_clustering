package gru

import (
	"fmt"
	"math"

	"github.com/GiorgosMarga/vanet_d_clustering/matrix"
)

type ActivationFunction func(float64) float64

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func sigmoidDerivative(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

func sigmoidMatrix(x [][]float64) [][]float64 {
	res := make([][]float64, len(x))
	for row := range x {
		res[row] = make([]float64, len(x[row]))
		for colIdx := range x[row] {
			res[row][colIdx] = sigmoid(x[row][colIdx])
		}
	}
	return res
}

func sigmoidDerivativeMatrix(x [][]float64) [][]float64 {
	res := make([][]float64, len(x))
	for row := range x {
		res[row] = make([]float64, len(x[row]))
		for colIdx := range x[row] {
			res[row][colIdx] = sigmoidDerivative(x[row][colIdx])
		}
	}
	return res
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func dtanh(x float64) float64 {
	return 1 - math.Pow(tanh(x), 2)
}

func tanhDerivativeMatrix(a [][]float64) [][]float64 {
	res := make([][]float64, len(a))
	for row := range a {
		res[row] = make([]float64, len(a[row]))
		for colIdx := range a[row] {
			res[row][colIdx] = dtanh(a[row][colIdx])
		}
	}
	return res
}

type Gate struct {
	// activation function for the gate ex. sigmoid
	activationFunc ActivationFunction
	// weight matrix for the input Xt
	WeightX [][]float64
	// weight matrix for the previous hidden state Ht-1
	WeightH [][]float64
	// bias hiddenSizex1
	bias [][]float64

	// gradients for backpropagation
	// for reset gate
	dWX [][]float64
	dWH [][]float64
	db  [][]float64
}

func NewGate(bias, WeightH, WeightX [][]float64, activationFunction ActivationFunction) *Gate {

	return &Gate{
		activationFunc: activationFunction,
		WeightX:        WeightX,
		WeightH:        WeightH,
		bias:           bias,
		dWX:            matrix.RandomMatrix(len(WeightX), len(WeightX[0]), 1.0/float64(len(bias))),
		dWH:            matrix.RandomMatrix(len(WeightH), len(WeightH[0]), 1.0/float64(len(bias))),
		db:             matrix.RandomMatrix(len(bias), len(bias[0]), 1.0/float64(len(bias))),
	}
}

// calculate calculates the result of the gate
// activationFunction(WeightX*x + WeightH*prevH + b)
// -> activationFunction(wx+wh+b)
func (rs *Gate) calculate(x, prevH [][]float64) ([][]float64, error) {
	wx := matrix.MatrixMul(rs.WeightX, x)
	wh := matrix.MatrixMul(rs.WeightH, prevH)
	if len(wx) != len(wh) {
		return nil, fmt.Errorf("Invalid xw and hw sizes: %d and %d\n", len(wx), len(wh))
	}
	if len(wx[0]) != len(wh[0]) {
		return nil, fmt.Errorf("Invalid xw[] and hw[] sizes: %d and %d\n", len(wx[0]), len(wh[0]))
	}
	output := make([][]float64, len(wx))
	for row := range len(output) {
		output[row] = make([]float64, len(wx[0]))
	}

	for row := range len(output) {
		for idx := range output[row] {
			output[row][idx] = sigmoid(wx[row][idx] + wh[row][idx] + rs.bias[row][0])
		}
	}

	return output, nil
}

func (g *Gate) updateWeights(lr float64) {
	g.WeightX = matrix.MatrixSubWithScalar(g.WeightX, g.dWX, lr)
	g.WeightH = matrix.MatrixSubWithScalar(g.WeightH, g.dWH, lr)
	g.bias = matrix.MatrixSubWithScalar(g.bias, g.db, lr)
}
func (g *Gate) resetGradients() {
	for row := range g.dWH {
		for col := range g.dWH[row] {
			g.dWH[row][col] = 0
		}
	}

	for row := range g.dWX {
		for col := range g.dWX[row] {
			g.dWX[row][col] = 0
		}
	}
	for row := range g.db {
		for col := range g.db[row] {
			g.db[row][col] = 0
		}
	}
}
func printMatrix(a [][]float64) {
	fmt.Println()

	for row := range a {
		for col := range a[row] {
			fmt.Printf("%f\t", a[row][col])
		}
		fmt.Println()
	}
	fmt.Println()
}
