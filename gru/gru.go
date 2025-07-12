package gru

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/GiorgosMarga/vanet_d_clustering/matrix"
	neuralnetwork "github.com/GiorgosMarga/vanet_d_clustering/neuralNetwork"
)

type GRUConfig struct {
	TrainSizePercentage float64
	HiddenStateSize     int
	InputSize           int
	Epochs              int
	BatchSize           int
	Patience            int
	LearningRate        float64
	LossThreshold       float64
}

type LossFunction func(yActual [][]float64, yPred [][]float64) float64

func (g *GRU) MeanSquareError(yActual, yPred [][]float64) float64 {
	sum := 0.0
	n := len(yActual)
	for row := range n {
		yA := [][][]float64{{{yActual[row][0]}}}
		yP := [][][]float64{{{yPred[row][0]}}}
		diff := g.Sy.InverseTransform(yA)[0][0][0] - g.Sy.InverseTransform(yP)[0][0][0]
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

	// early stopping patience
	earlyStop *EarlyStop

	// input and target
	X [][][]float64
	Y [][][]float64

	// save errors in case of plotting
	Errors     []float64
	Accuracies []float64

	// scalers
	Sx *Scaler
	Sy *Scaler

	trainingSize      float64
	inputSize         int
	lossThreshold     float64
	prevLoss          float64
	convergenceRounds int
	epochs            int
	batchSize         int
}

func NewGRU(config *GRUConfig) neuralnetwork.NeuralNetwork {
	hiddenSize := 16
	trainingSize := 0.8
	inputSize := 4
	epochs := 50
	batchSize := 3
	patience := 20
	learningRate := 0.05
	lossThreshold := 0.001
	if config.HiddenStateSize != 0 {
		hiddenSize = config.HiddenStateSize
	}
	if config.TrainSizePercentage != 0.0 {
		trainingSize = config.TrainSizePercentage
	}
	if config.InputSize != 0 {
		inputSize = config.InputSize
	}
	if config.Epochs != 0 {
		epochs = config.Epochs
	}
	if config.Patience != 0 {
		patience = config.Patience
	}

	if config.Epochs != 0 {
		epochs = config.Epochs
	}
	if config.BatchSize != 0 {
		batchSize = config.BatchSize
	}

	scale := 1.0 / float64(hiddenSize) // Xavier-like scaling
	updateGate := NewGate(matrix.RandomMatrix(hiddenSize, 1, scale), matrix.RandomMatrix(hiddenSize, hiddenSize, scale), matrix.RandomMatrix(hiddenSize, inputSize, scale), sigmoid)
	resetGate := NewGate(matrix.RandomMatrix(hiddenSize, 1, scale), matrix.RandomMatrix(hiddenSize, hiddenSize, scale), matrix.RandomMatrix(hiddenSize, inputSize, scale), sigmoid)
	g := &GRU{
		UpdateGate:    updateGate,
		ResetGate:     resetGate,
		Whx:           matrix.RandomMatrix(hiddenSize, inputSize, scale),
		Whh:           matrix.RandomMatrix(hiddenSize, hiddenSize, scale),
		bh:            matrix.RandomMatrix(hiddenSize, 1, scale),
		finalH:        matrix.RandomMatrix(hiddenSize, 1, 0),
		hiddenSize:    hiddenSize,
		dWhiddenX:     matrix.RandomMatrix(hiddenSize, inputSize, scale),
		dWhiddenH:     matrix.RandomMatrix(hiddenSize, hiddenSize, scale),
		dbhidden:      matrix.RandomMatrix(hiddenSize, 1, scale),
		wOut:          matrix.RandomMatrix(1, hiddenSize, scale),
		bOut:          matrix.RandomMatrix(1, 1, scale),
		learningRate:  learningRate,
		earlyStop:     NewEarlyStop(patience, 0.001),
		Errors:        make([]float64, 0),
		Sx:            NewScaler(),
		Sy:            NewScaler(),
		trainingSize:  trainingSize,
		inputSize:     inputSize,
		lossThreshold: lossThreshold,
		prevLoss:      math.MaxFloat64,
		Accuracies:    make([]float64, 0),
		epochs:        epochs,
		batchSize:     batchSize,
	}
	g.lossFunction = g.MeanSquareError
	return g
}
func shuffleData(X, Y [][][]float64) {
	if len(X) != len(Y) {
		panic("X and Y must have the same number of samples")
	}
	r := rand.New(rand.NewSource(10))
	// Shuffle in place.
	r.Shuffle(len(X), func(i, j int) {
		X[i], X[j] = X[j], X[i]
		Y[i], Y[j] = Y[j], Y[i]
	})
}
func (g *GRU) GetParsevalValues(numOfParsevalValues int) []float64 {
	flattenX := make([]float64, 0)
	for _, mat := range g.X {
		flattenX = append(flattenX, matrix.Flatten(mat)...)
	}

	return matrix.Parseval(matrix.FFT(flattenX))[:numOfParsevalValues]
}

func (g *GRU) calculateCandidateHiddenState() [][]float64 {
	a := matrix.MatrixMul(g.Whx, g.Input)
	b := matrix.MatrixMul(g.Whh, matrix.ElementMatrixMul(g.r, g.prevH))

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
	g.output, err = g.forwardPass(0)
	return g.output, err
}

func (g *GRU) calculateFinalHiddenState() [][]float64 {
	a := matrix.ElementMatrixMul(g.z, g.prevH)
	b := matrix.MatrixSub(matrix.IdentityMatrix(len(g.z), len(g.z[0])), g.z)

	return matrix.MatrixAdd(a, matrix.ElementMatrixMul(b, g.candidateH))
}

func (g *GRU) forwardPass(dropoutRate float64) ([][]float64, error) {
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
	g.finalH = dropoutInput(g.calculateFinalHiddenState(), dropoutRate)
	g.output = matrix.MatrixAdd(matrix.MatrixMul(g.wOut, g.finalH), g.bOut)
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
	dWout := matrix.MatrixMul(dOutput, matrix.TransposeMatrix(g.finalH)) // dW_out is [1x hiddenSize]

	// Compute gradient of loss w.r.t. b_out (db_out)
	dbout := dOutput // db_out is [1x1]

	// Compute gradient of loss w.r.t. finalH (dFinalH)
	dH := matrix.MatrixMul(matrix.TransposeMatrix(g.wOut), dOutput) // dFinalH is [hiddenSize x 1]

	// Update output layer parameters
	g.wOut = matrix.MatrixSubWithScalar(g.wOut, dWout, g.learningRate)
	g.bOut = matrix.MatrixSubWithScalar(g.bOut, dbout, g.learningRate)

	// gradients for update gate
	dz := matrix.ElementMatrixMul(matrix.MatrixSub(g.prevH, g.candidateH), dH)
	sigmoidDerivZ := matrix.ElementMatrixMul(g.z, matrix.MatrixSub(matrix.IdentityMatrix(len(g.z), len(g.z[0])), g.z))
	deltaZ := matrix.ElementMatrixMul(dz, sigmoidDerivZ)

	g.UpdateGate.dWX = matrix.MatrixAdd(g.UpdateGate.dWX, matrix.MatrixMul(deltaZ, matrix.TransposeMatrix(g.Input)))
	g.UpdateGate.dWH = matrix.MatrixAdd(g.UpdateGate.dWH, matrix.MatrixMul(deltaZ, matrix.TransposeMatrix(g.prevH)))
	g.UpdateGate.db = matrix.MatrixAdd(g.UpdateGate.db, matrix.SumRows(deltaZ))

	// gradient for Candidate Hidden State
	dCandidateH := matrix.ElementMatrixMul(matrix.MatrixSub(matrix.IdentityMatrix(len(g.z), len(g.z[0])), g.z), dH)
	tanhDeriv := matrix.MatrixSub(matrix.IdentityMatrix(len(g.candidateH), len(g.candidateH[0])), matrix.ElementMatrixMul(g.candidateH, g.candidateH))

	deltaA := matrix.ElementMatrixMul(dCandidateH, tanhDeriv)

	g.dWhiddenX = matrix.MatrixAdd(g.dWhiddenX, matrix.MatrixMul(deltaA, matrix.TransposeMatrix(g.Input)))
	WhhPrevH := matrix.MatrixMul(g.Whh, g.prevH)
	g.dWhiddenH = matrix.MatrixAdd(g.dWhiddenH, matrix.MatrixMul(matrix.ElementMatrixMul(g.r, deltaA), matrix.TransposeMatrix(g.prevH)))
	g.dbhidden = matrix.MatrixAdd(g.dbhidden, matrix.SumRows(deltaA))

	// Gradient for Reset Gate (r)
	drCandidate := matrix.ElementMatrixMul(WhhPrevH, deltaA)
	sigmoidDerivR := matrix.ElementMatrixMul(g.r, matrix.MatrixSub(matrix.IdentityMatrix(len(g.r), len(g.r[0])), g.r))
	deltaR := matrix.ElementMatrixMul(drCandidate, sigmoidDerivR)

	g.ResetGate.dWX = matrix.MatrixAdd(g.ResetGate.dWX, matrix.MatrixMul(deltaR, matrix.TransposeMatrix(g.Input)))
	g.ResetGate.dWH = matrix.MatrixAdd(g.ResetGate.dWH, matrix.MatrixMul(deltaR, matrix.TransposeMatrix(g.prevH)))
	g.ResetGate.db = matrix.MatrixAdd(g.ResetGate.db, matrix.SumRows(deltaR))

	return nil
}

func (g *GRU) updateWeights() {
	g.Whx = matrix.MatrixSubWithScalar(g.Whx, g.dWhiddenX, g.learningRate)

	g.Whh = matrix.MatrixSubWithScalar(g.Whh, g.dWhiddenH, g.learningRate)

	g.bh = matrix.MatrixSubWithScalar(g.bh, g.dbhidden, g.learningRate)
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

func (g *GRU) Train() error {
	trainSize := int(float64(len(g.X)) * g.trainingSize)
	inputs := g.X[:trainSize]
	targets := g.Y[:trainSize]
	g.initializeHiddenState(g.hiddenSize)
	for epoch := range g.epochs {
		_ = epoch
		var totalLoss float64 = 0

		for batch := 0; batch < len(inputs); batch += g.batchSize {

			batchEnd := min(batch+g.batchSize, len(inputs))
			batchX := inputs[batch:batchEnd]
			batchY := targets[batch:batchEnd]
			batchLoss := 0.0
			g.resetGradients()
			for t := range batchX {
				// Forward pass
				g.Input = batchX[t]
				predicted, err := g.forwardPass(0.3)
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
		averageLoss := totalLoss / float64(len(inputs)/g.batchSize)
		// g.Errors = append(g.Errors, averageLoss)

		if averageLoss < g.lossThreshold || math.Abs(g.prevLoss-averageLoss) < 1e-6 {
			g.convergenceRounds = epoch + 1
			fmt.Printf("Converged at epoch: %d\n", g.convergenceRounds)
			break
		}
		g.prevLoss = averageLoss
		if g.earlyStop != nil && g.earlyStop.CheckEarlyStop(averageLoss) {
			fmt.Printf("Early stop at epoch: %d\n", epoch)
			break
		}
		// fmt.Printf("Epoch: %d, Avg Loss: %.4f\n", epoch, averageLoss)
	}
	g.earlyStop.Reset()
	// TODO: remove this, only for testing, to get accuracy per round
	g.Evaluate()

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

func (g *GRU) GetWeights() [][][]float64 {
	weights := make([][][]float64, 10)

	weights[0] = g.Whx
	weights[1] = g.Whh
	weights[2] = g.bh
	weights[3] = g.UpdateGate.dWH
	weights[4] = g.UpdateGate.dWX
	weights[5] = g.UpdateGate.bias
	weights[6] = g.ResetGate.dWX
	weights[7] = g.ResetGate.dWH
	weights[8] = g.ResetGate.bias
	weights[9] = g.wOut
	return weights
}

func (g *GRU) SetWeights(weights [][][]float64) error {
	if len(weights) != 10 {
		return fmt.Errorf("weights length: %d, expected: 10", len(weights))
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
	g.wOut = weights[9]
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
		if line+g.inputSize >= len(lines) {
			break
		}
		t := make([][]float64, g.inputSize)
		for i := range g.inputSize {
			n, err := strconv.ParseFloat(lines[line+i], 64)
			if err != nil {
				panic(err)
			}
			t[i] = []float64{n}
		}
		X = append(X, t)
		n, err := strconv.ParseFloat(lines[line+g.inputSize], 64)
		if err != nil {
			panic(err)
		}
		t2 := make([][]float64, 1)
		t2[0] = []float64{n}
		Y = append(Y, t2)
	}

	g.X = g.Sx.FitTransform(X)
	g.Y = g.Sy.FitTransform(Y)

	// shuffleData(g.X, g.Y)
	// g.X = X
	// g.Y = Y

	return nil

}

func (g *GRU) Evaluate() ([]float64, []float64, error) {
	dataSize := int(float64(len(g.X)) * g.trainingSize)
	X := g.X[dataSize:]
	Y := g.Y[dataSize:]

	predictions := make([]float64, 0, len(X))
	expected := make([]float64, 0, len(Y))
	accuracy := 0
	mse := 0.0

	for i := range len(X) {
		output, err := g.Predict(X[i])
		if err != nil {
			return nil, nil, err
		}
		guess := g.Sx.InverseTransform([][][]float64{output})[0][0][0]
		actual := g.Sy.InverseTransform([][][]float64{Y[i]})[0][0][0]
		predictions = append(predictions, guess)
		expected = append(expected, actual)
		mse += math.Pow(guess-actual, 2)
		if math.Abs(guess-actual)/math.Abs(actual) < 0.2 {
			accuracy += 1
		}
	}
	g.Accuracies = append(g.Accuracies, float64(accuracy)*100.0/float64(len(X)))
	g.Errors = append(g.Errors, mse/float64(len(X)))
	return predictions, expected, nil

}
func (g *GRU) GetErrors() []float64 {
	return g.Errors
}
func (g *GRU) GetAccuracies() []float64 {
	return g.Accuracies
}
