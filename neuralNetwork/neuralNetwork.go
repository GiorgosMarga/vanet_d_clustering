package neuralnetwork

type NeuralNetwork interface {
	SetWeights(weights [][][]float64) error
	Predict(X [][]float64) ([][]float64, error)
	Train() error
	GetWeights() [][][]float64
	Evaluate() ([]float64, []float64, error)
	ParseFile(string) error
	GetErrors() []float64
	GetAccuracies() []float64
	GetParsevalValues(numOfParsevalValues int) []float64
}
