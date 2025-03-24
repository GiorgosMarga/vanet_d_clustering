package gru

import (
	"fmt"
	"math/rand"
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
}

func NewGRU(hiddenSize, inputSize int, lossFunction LossFunction, learningRate float64) *GRU {
	scale := 1.0 / float64(hiddenSize) // Xavier-like scaling
	updateGate := NewGate(randomMatrix(hiddenSize, 1, scale), randomMatrix(hiddenSize, hiddenSize, scale), randomMatrix(hiddenSize, inputSize, scale), sigmoid)
	resetGate := NewGate(randomMatrix(hiddenSize, 1, scale), randomMatrix(hiddenSize, hiddenSize, scale), randomMatrix(hiddenSize, inputSize, scale), sigmoid)
	return &GRU{
		UpdateGate:   updateGate,
		ResetGate:    resetGate,
		Whx:          randomMatrix(hiddenSize, inputSize, scale),
		Whh:          randomMatrix(hiddenSize, hiddenSize, scale),
		bh:           randomMatrix(hiddenSize, 1, scale),
		finalH:       randomMatrix(hiddenSize, 1, 0),
		lossFunction: lossFunction,
		hiddenSize:   hiddenSize,
		dWhiddenX:    randomMatrix(hiddenSize, inputSize, scale),
		dWhiddenH:    randomMatrix(hiddenSize, hiddenSize, scale),
		dbhidden:     randomMatrix(hiddenSize, 1, scale),
		wOut:         randomMatrix(1, hiddenSize, scale),
		bOut:         randomMatrix(1, 1, scale),
		learningRate: learningRate,
	}
}

func shuffleData(X, Y [][][]float64) {
	if len(X) != len(Y) {
		panic("X and Y must have the same number of samples")
	}

	// Shuffle in place.
	rand.Shuffle(len(X), func(i, j int) {
		X[i], X[j] = X[j], X[i]
		Y[i], Y[j] = Y[j], Y[i]
	})
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

func (g *GRU) calculateFinalHiddenState() [][]float64 {
	a := elementMatrixMul(g.z, g.prevH)
	b := matrixSub(identityMatrix(len(g.z), len(g.z[0])), g.z)

	return matrixAdd(a, elementMatrixMul(b, g.candidateH))
}

func (g *GRU) forwardPass() ([][]float64, error) {
	g.prevH = g.finalH
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
	WhhPrevH := matrixMul(g.Whh, g.prevH)
	g.dWhiddenH = matrixAdd(g.dWhiddenH, matrixMul(elementMatrixMul(g.r, deltaA), transposeMatrix(g.prevH)))
	g.dbhidden = matrixAdd(g.dbhidden, sumRows(deltaA))

	// Gradient for Reset Gate (r)
	drCandidate := elementMatrixMul(WhhPrevH, deltaA)
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
func (g *GRU) Train(inputs, targets [][][]float64, epochs, batchSize int) error {
	g.initializeHiddenState(g.hiddenSize)
	for epoch := range epochs {
		var totalLoss float64 = 0

		for batch := 0; batch < len(inputs); batch += batchSize {

			batchEnd := min(batch+batchSize, len(inputs))
			batchX := inputs[batch:batchEnd]
			batchY := targets[batch:batchEnd]
			batchLoss := 0.0
			g.resetGradients()
			for t := range batchX {
				// Forward pass
				g.Input = batchX[t]
				predicted, err := g.forwardPass()
				if err != nil {
					return err
				}

				loss := g.lossFunction(batchY[t], predicted)
				batchLoss += loss
				// Backward pass
				g.backwardPass(batchY[t])

			}
			// Update GRU parameters using gradient descent
			g.updateWeights()
			totalLoss += batchLoss
		}
		averageLoss := totalLoss / float64(len(inputs)/batchSize)
		fmt.Printf("Epoch: %d, Avg Loss: %.4f\n", epoch, averageLoss)
	}

	return nil
}

func (g *GRU) resetGradients() {
	for row := range g.dWhiddenH {
		for colIdx := range g.dWhiddenH[row] {
			g.dWhiddenH[row][colIdx] = 0
		}
	}

	for row := range g.dWhiddenX {
		for colIdx := range g.dWhiddenX[row] {
			g.dWhiddenX[row][colIdx] = 0
		}
	}

	for row := range g.dbhidden {
		for colIdx := range g.dbhidden[row] {
			g.dbhidden[row][colIdx] = 0
		}
	}

	g.ResetGate.resetGradients()
	g.UpdateGate.resetGradients()

}
