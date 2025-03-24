package gru

import (
	"fmt"
	"math"
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
	weightX [][]float64
	// weight matrix for the previous hidden state Ht-1
	weightH [][]float64
	// bias hiddenSizex1
	bias [][]float64

	// gradients for backpropagation
	// for reset gate
	dWX [][]float64
	dWH [][]float64
	db  [][]float64
}

func NewGate(bias, weightH, weightX [][]float64, activationFunction ActivationFunction) *Gate {

	return &Gate{
		activationFunc: activationFunction,
		weightX:        weightX,
		weightH:        weightH,
		bias:           bias,
		dWX:            randomMatrix(len(weightX), len(weightX[0]), 1.0/float64(len(bias))),
		dWH:            randomMatrix(len(weightH), len(weightH[0]), 1.0/float64(len(bias))),
		db:             randomMatrix(len(bias), len(bias[0]), 1.0/float64(len(bias))),
	}
}

// calculate calculates the result of the gate
// activationFunction(weightX*x + weightH*prevH + b)
// -> activationFunction(wx+wh+b)
func (rs *Gate) calculate(x, prevH [][]float64) ([][]float64, error) {
	wx := matrixMul(rs.weightX, x)
	wh := matrixMul(rs.weightH, prevH)
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
	g.weightX = matrixSubWithScalar(g.weightX, g.dWX, lr)
	g.weightH = matrixSubWithScalar(g.weightH, g.dWH, lr)
	g.bias = matrixSubWithScalar(g.bias, g.db, lr)
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
