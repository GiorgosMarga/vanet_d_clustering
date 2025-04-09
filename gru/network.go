package gru

type network interface {
	Train(inputs, targets [][][]float64, epochs int) error
}
