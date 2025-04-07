package gru

import (
	"fmt"
<<<<<<< HEAD
	"math"
	"math/rand/v2"
=======
	"math/rand"
	"os"
	"strconv"
	"strings"
>>>>>>> 7b551a218ba6306d7e17e8d3ba926a84ef878404
)

type LossFunction func(yActual [][]float64, yPred [][]float64) float64

func MeanSquareError(yActual, yPred [][]float64) float64 {
	sum := 0.0
	n := len(yActual)
	for row := range n {
		diff := yActual[row][0] - yPred[row][0]
		sum += diff * diff
	}
	return sum / float64(n)
}

func elementMatrixMul(a, b [][]float64) [][]float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("a and b have different dimensions (%dx%d) and (%dx%d)\n", len(a), len(a[0]), len(b), len(b[0])))
	}

	result := make([][]float64, len(a))
	for row := range result {
		result[row] = make([]float64, 1)
	}

	for idx := range a {
		aRow := a[idx]
		bRow := b[idx]
		if len(aRow) != len(bRow) {
			panic(fmt.Sprintf("Different row size: %d to %d\n", len(aRow), len(bRow)))
		}

		for colIdx := range aRow {
			result[idx][colIdx] = aRow[colIdx] * bRow[colIdx]
		}
	}

	return result

}
func identityMatrix(rows, cols int) [][]float64 {
	res := make([][]float64, rows)
	for row := range res {
		res[row] = make([]float64, cols)
		for colIdx := range res[row] {
			res[row][colIdx] = 1
		}
	}
	return res
}
func matrixMul(a, b [][]float64) [][]float64 {
	if len(a[0]) != len(b) {
		panic(fmt.Sprintf("Mul: Invalid dimensions: a:(%dx%d) * b(%dx%d)\n", len(a), len(a[0]), len(b), len(b[0])))
	}
	result := make([][]float64, len(a))
	for row := range len(result) {
		result[row] = make([]float64, len(b[0]))
	}

	for idxRow := range a {
		for idxCol := range b[0] {
			var sum float64 = 0
			for c := range a[0] {
				sum += a[idxRow][c] * b[c][idxCol]
			}
			result[idxRow][idxCol] = sum
		}
	}

	return result
}

func matrixAdd(a, b [][]float64) [][]float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("Add: Invalid dimensions (%dx%d) and (%dx%d)\n", len(a), len(a[0]), len(b), len(b[0])))
	}

	output := make([][]float64, len(a))

	for row := range a {
		output[row] = make([]float64, len(a[0]))
		for col := range a[row] {
			output[row][col] = a[row][col] + b[row][col]
		}
	}
	return output

}

func matrixSub(a, b [][]float64) [][]float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("Sub: Invalid dimensions (%dx%d) and (%dx%d)\n", len(a), len(a[0]), len(b), len(b[0])))
	}

	output := make([][]float64, len(a))

	for row := range a {
		output[row] = make([]float64, len(a[0]))
		for col := range a[row] {
			output[row][col] = a[row][col] - b[row][col]
		}
	}
	return output
}

func matrixSubWithScalar(a, b [][]float64, scale float64) [][]float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("SubScale: Invalid dimensions (%dx%d) and (%dx%d)\n", len(a), len(a[0]), len(b), len(b[0])))
	}

	output := make([][]float64, len(a))

	for row := range a {
		output[row] = make([]float64, len(a[0]))
		for col := range a[row] {
			output[row][col] = a[row][col] - b[row][col]*scale
		}
	}
	return output

}

func matrixOuterProduct(a, b [][]float64) [][]float64 {
	m := len(a)
	n := len(b)
	result := make([][]float64, m)
	for i := 0; i < m; i++ {
		result[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			result[i][j] = a[i][0] * b[j][0]
		}
	}
	return result
}
func randomMatrix(rows, cols int, scale float64) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = (rand.Float64()*2 - 1) * scale // Range: [-scale, scale]
		}
	}
	return matrix
}

func transposeMatrix(a [][]float64) [][]float64 {
	res := make([][]float64, len(a[0]))
	for rowIdx := range res {
		res[rowIdx] = make([]float64, len(a))
	}
	for rowIdx := range a {
		for colIdx := range a[rowIdx] {
			res[colIdx][rowIdx] = a[rowIdx][colIdx]
		}
	}
	return res
}
func sumRows(matrix [][]float64) [][]float64 {
	rows := len(matrix)
	cols := len(matrix[0])
	sum := make([][]float64, rows)

	for i := range rows {
		sum[i] = make([]float64, 1)
		for j := range cols {
			sum[i][0] += matrix[i][j]
		}
	}
	return sum
}

type GRU struct {

	// update gate
	UpdateGate *Gate
	// reset gate
	ResetGate *Gate
	// hidden size
	hiddenSize int
	// input inputSizex1
	Input [][]float64
	// W for input. Used in calculating the hidden state ht
	Whx [][]float64
	// W for prevH. Used in calculating the hidden state ht
	Whh [][]float64
	// bias. Used in calculating the hidden state ht
	bh [][]float64
	// r is the result of the reset gate
	r [][]float64
	// prevH is the previous H
	prevH [][]float64
	// z is the result of the update gate
	z [][]float64
	// candidateH is the ht bar
	candidateH [][]float64
	// finalH combines the r z and candidateH. Used is forward pass
	finalH       [][]float64
	lossFunction LossFunction
	learningRate float64

	// gradients used in backpropagation for the hidden state
	dWhiddenX [][]float64
	dWhiddenH [][]float64
	dbhidden  [][]float64

	// for output layer
	wOut   [][]float64
	bOut   [][]float64
	output [][]float64

	// early stopping patience
	earlyStop *EarlyStop

	// input and target
	X [][][]float64
	Y [][][]float64

	// save errors in case of plotting
	Errors []float64

	// scalers
	Sx *Scaler
	Sy *Scaler
}

func NewGRU(hiddenSize, inputSize, patience int, lossFunction LossFunction, learningRate float64) *GRU {
	scale := 1.0 / float64(hiddenSize) // Xavier-like scaling
	updateGate := NewGate(randomMatrix(hiddenSize, 1, scale), randomMatrix(hiddenSize, hiddenSize, scale), randomMatrix(hiddenSize, inputSize, scale), sigmoid)
	resetGate := NewGate(randomMatrix(hiddenSize, 1, scale), randomMatrix(hiddenSize, hiddenSize, scale), randomMatrix(hiddenSize, inputSize, scale), sigmoid)
	return &GRU{
		UpdateGate:   updateGate,
		ResetGate:    resetGate,
		Whx:          randomMatrix(hiddenSize, inputSize, scale),
		Whh:          randomMatrix(hiddenSize, hiddenSize, scale),
		bh:           randomMatrix(hiddenSize, 1, scale),
		prevH:        randomMatrix(hiddenSize, 1, 0),
		lossFunction: lossFunction,
		hiddenSize:   hiddenSize,
		dWhiddenX:    randomMatrix(hiddenSize, inputSize, scale),
		dWhiddenH:    randomMatrix(hiddenSize, hiddenSize, scale),
		dbhidden:     randomMatrix(hiddenSize, 1, scale),
		wOut:         randomMatrix(1, hiddenSize, scale),
		bOut:         randomMatrix(1, 1, scale),
		learningRate: learningRate,
		earlyStop:    NewEarlyStop(patience, 0.001),
		Errors:       make([]float64, 0),
		Sx:           NewScaler(),
		Sy:           NewScaler(),
	}
}

func (g *GRU) calculateCandidateHiddenState() [][]float64 {
	a := matrixMul(g.Whx, g.Input)
	b := matrixMul(g.Whh, elementMatrixMul(g.r, g.prevH))

	g.candidateH = make([][]float64, len(a))
	for row := range a {
		g.candidateH[row] = make([]float64, len(a[0]))
		for colIdx := range a[row] {
			g.candidateH[row][colIdx] = tanh(a[row][colIdx] + b[row][colIdx] + g.bh[row][0])
		}
	}

	return g.candidateH
}
func (g *GRU) Predict(X [][]float64) ([][]float64, error) {
	var err error
	g.Input = X
	g.output, err = g.forwardPass()
	return g.output, err
}

func (g *GRU) calculateFinalHiddenState() [][]float64 {
	a := elementMatrixMul(g.z, g.prevH)
	b := matrixSub(identityMatrix(len(g.z), len(g.z[0])), g.z)

	return matrixAdd(a, elementMatrixMul(b, g.candidateH))
}

func (g *GRU) forwardPass() ([][]float64, error) {
	var err error

	g.r, err = g.ResetGate.calculate(g.Input, g.prevH)
	if err != nil {
		return nil, err
	}

	g.z, err = g.UpdateGate.calculate(g.Input, g.prevH)
	if err != nil {
		return nil, err
	}

	g.candidateH = g.calculateCandidateHiddenState()
	g.finalH = g.calculateFinalHiddenState()
	g.output = matrixAdd(matrixMul(g.wOut, g.finalH), g.bOut)
	return g.output, nil
}

func computeLossGrad(finalH, yActual [][]float64) [][]float64 {
	numSamples := len(yActual)
	lossGrad := make([][]float64, numSamples)
	for i := range lossGrad {
		lossGrad[i] = make([]float64, len(yActual[i]))
	}

	for i := range lossGrad {
		lossGrad[i] = make([]float64, len(yActual[i]))
		for j := range lossGrad[i] {
			lossGrad[i][j] = (2.0 / float64(numSamples)) * (finalH[i][j] - yActual[i][j])
		}
	}
	return lossGrad
}

func (g *GRU) backwardPass(yActual [][]float64) error {

	// Compute gradient of loss w.r.t. output (dOutput)
	dOutput := computeLossGrad(g.output, yActual) // dOutput is [1x1]
	// Compute gradient of loss w.r.t. W_out (dW_out)
	dWout := matrixMul(dOutput, transposeMatrix(g.finalH)) // dW_out is [1x hiddenSize]

	// Compute gradient of loss w.r.t. b_out (db_out)
	dbout := dOutput // db_out is [1x1]

	// Compute gradient of loss w.r.t. finalH (dFinalH)
	dH := matrixMul(transposeMatrix(g.wOut), dOutput) // dFinalH is [hiddenSize x 1]

	// Update output layer parameters
	g.wOut = matrixSubWithScalar(g.wOut, dWout, g.learningRate)
	g.bOut = matrixSubWithScalar(g.bOut, dbout, g.learningRate)

	// gradients for update gate
	dz := elementMatrixMul(matrixSub(g.prevH, g.candidateH), dH)
	sigmoidDerivZ := elementMatrixMul(g.z, matrixSub(identityMatrix(len(g.z), len(g.z[0])), g.z))
	deltaZ := elementMatrixMul(dz, sigmoidDerivZ)

	g.UpdateGate.dWX = matrixAdd(g.UpdateGate.dWX, matrixMul(deltaZ, transposeMatrix(g.Input)))
	g.UpdateGate.dWH = matrixAdd(g.UpdateGate.dWH, matrixMul(deltaZ, transposeMatrix(g.prevH)))
	g.UpdateGate.db = matrixAdd(g.UpdateGate.db, sumRows(deltaZ))

	// gradient for Candidate Hidden State
	dCandidateH := elementMatrixMul(matrixSub(identityMatrix(len(g.z), len(g.z[0])), g.z), dH)
	tanhDeriv := matrixSub(identityMatrix(len(g.candidateH), len(g.candidateH[0])), elementMatrixMul(g.candidateH, g.candidateH))

	deltaA := elementMatrixMul(dCandidateH, tanhDeriv)

	g.dWhiddenX = matrixAdd(g.dWhiddenX, matrixMul(deltaA, transposeMatrix(g.Input)))
	Whh_prevH := matrixMul(g.Whh, g.prevH)
	g.dWhiddenH = matrixAdd(g.dWhiddenH, matrixMul(elementMatrixMul(g.r, deltaA), transposeMatrix(g.prevH)))
	g.dbhidden = matrixAdd(g.dbhidden, sumRows(deltaA))

	// Gradient for Reset Gate (r)
	drCandidate := elementMatrixMul(Whh_prevH, deltaA)
	sigmoidDerivR := elementMatrixMul(g.r, matrixSub(identityMatrix(len(g.r), len(g.r[0])), g.r))
	deltaR := elementMatrixMul(drCandidate, sigmoidDerivR)

	g.ResetGate.dWX = matrixAdd(g.ResetGate.dWX, matrixMul(deltaR, transposeMatrix(g.Input)))
	g.ResetGate.dWH = matrixAdd(g.ResetGate.dWH, matrixMul(deltaR, transposeMatrix(g.prevH)))
	g.ResetGate.db = matrixAdd(g.ResetGate.db, sumRows(deltaR))

	return nil
}

func (g *GRU) updateWeights() {
	g.Whx = matrixSubWithScalar(g.Whx, g.dWhiddenX, g.learningRate)

	g.Whh = matrixSubWithScalar(g.Whh, g.dWhiddenH, g.learningRate)

	g.bh = matrixSubWithScalar(g.bh, g.dbhidden, g.learningRate)
	g.UpdateGate.updateWeights(g.learningRate)
	g.ResetGate.updateWeights(g.learningRate)

}
func (g *GRU) initializeHiddenState(hiddenSize int) {
	h := make([][]float64, hiddenSize)
	for idx := range h {
		h[idx] = make([]float64, 1)
	}
	g.prevH = h
}
<<<<<<< HEAD
func (g *GRU) Train(inputs, targets [][][]float64, epochs int) error {
=======

func (g *GRU) Train(inputs, targets [][][]float64, epochs, batchSize int) error {
>>>>>>> 7b551a218ba6306d7e17e8d3ba926a84ef878404
	g.initializeHiddenState(g.hiddenSize)
	for range epochs {
		var totalLoss float64 = 0

		for t := range inputs {
			// Forward pass
			g.Input = inputs[t]
			predicted, err := g.forwardPass()
			if err != nil {
				return err
			}
			loss := g.lossFunction(targets[t], predicted)
			totalLoss += loss
			// Backward pass
			g.backwardPass(targets[t])

			// Update GRU parameters using gradient descent
			g.updateWeights()
			g.prevH = g.finalH
		}
<<<<<<< HEAD
		// Print average loss for the epoch
		averageLoss := totalLoss / float64(len(inputs))
		fmt.Printf("Epoch: %d, Loss: %.4f\n", epoch, averageLoss)
=======
		averageLoss := totalLoss / float64(len(inputs)/batchSize)
		if g.earlyStop.CheckEarlyStop(averageLoss) {
			// fmt.Printf("Early stopping at epoch %d\n", epoch)
			break
		}
		// fmt.Printf("Epoch: %d, Avg Loss: %.4f\n", epoch, averageLoss)
		g.Errors = append(g.Errors, averageLoss)
>>>>>>> 7b551a218ba6306d7e17e8d3ba926a84ef878404
	}

	return nil
}

func computeMean(X [][][]float64) float64 {
	sum := 0.0
	count := 0
	for i := range X {
		for ii := range X[i] {
			for iii := range X[i][ii] {
				sum += X[i][ii][iii]
				count++
			}
		}
	}
	return sum / float64(count)
}

func computeStd(X [][][]float64, mean float64) float64 {
	sumSq := 0.0
	count := 0
	for i := range X {
		for ii := range X[i] {
			for iii := range X[i][ii] {
				diff := X[i][ii][iii] - mean
				sumSq += diff * diff
				count++
			}
		}
	}
	return math.Sqrt(sumSq / float64(count))
}

func standardizeData(X [][][]float64) [][][]float64 {
	mean := computeMean(X)
	std := computeStd(X, mean)

	standardized := make([][][]float64, len(X))
	for i := range X {
		standardized[i] = make([][]float64, len(X[i]))
		for ii := range X[i] {
			standardized[i][ii] = make([]float64, len(X[i][ii]))
			for iii := range X[i][ii] {
				standardized[i][ii][iii] = (X[i][ii][iii] - mean) / std
			}
		}
	}

	return standardized
}

func (g *GRU) GetWeights() [][][]float64 {
	weights := make([][][]float64, 9)

	weights[0] = g.Whx
	weights[1] = g.Whh
	weights[2] = g.bh
	weights[3] = g.UpdateGate.dWH
	weights[4] = g.UpdateGate.dWX
	weights[5] = g.UpdateGate.bias
	weights[6] = g.ResetGate.dWX
	weights[7] = g.ResetGate.dWH
	weights[8] = g.ResetGate.bias
	return weights
}

func (g *GRU) SetWeights(weights [][][]float64) error {
	if len(weights) != 9 {
		return fmt.Errorf("weights length: %d, expected: 9", len(weights))
	}

	g.Whx = weights[0]
	g.Whh = weights[1]
	g.bh = weights[2]
	g.UpdateGate.dWH = weights[3]
	g.UpdateGate.dWX = weights[4]
	g.UpdateGate.bias = weights[5]
	g.ResetGate.dWX = weights[6]
	g.ResetGate.dWH = weights[7]
	g.ResetGate.bias = weights[8]
	return nil
}
func R2Score(yActual, yPred [][]float64) float64 {
	if len(yActual) != len(yPred) {
		panic("yActual and yPred must have the same length")
	}

	var sumActual float64 = 0
	n := len(yActual)
	for i := range n {
		sumActual += yActual[i][0]
	}
	meanActual := sumActual / float64(n)

	var rss, tss float64 = 0, 0
	for i := range n {
		diff := yActual[i][0] - yPred[i][0]
		rss += diff * diff
		tss += (yActual[i][0] - meanActual) * (yActual[i][0] - meanActual)
	}

	// If TSS is zero (constant actual values), return R² = 1
	if tss == 0 {
		return 1
	}

	r2 := 1 - (rss / tss)
	return r2
}

func (g *GRU) ParseFile(filename string) error {
	f, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	lines := strings.Split(string(f[:len(f)-1]), "\n")
	var X [][][]float64
	var Y [][][]float64
	for line := range len(lines) {
		if line+5 > len(lines) {
			break
		}
		t := make([][]float64, 4)
		for i := range 4 {
			n, err := strconv.ParseFloat(lines[line+i], 64)
			if err != nil {
				panic(err)
			}
			t[i] = []float64{n}
		}
		X = append(X, t)
		n, err := strconv.ParseFloat(lines[line+4], 64)
		if err != nil {
			panic(err)
		}
		t2 := make([][]float64, 1)
		t2[0] = []float64{n}
		Y = append(Y, t2)
	}

	shuffleData(X, Y)

	g.X = g.Sx.FitTransform(X)
	g.Y = g.Sy.FitTransform(Y)

	return nil

}
