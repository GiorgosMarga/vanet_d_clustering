package neuralnetwork

type NeuralNetwork interface {
	SetWeights(weights [][][]float64) error
	Predict(X [][]float64) ([][]float64, error)
	Train(epochs, batchSize int) error
	GetWeights() [][][]float64
	Evaluate() error
}
