package gru

import "math"

type EarlyStop struct {
	patience int
	counter  int
	delta    float64
	bestLoss float64
}

func NewEarlyStop(patience int, delta float64) *EarlyStop {
	return &EarlyStop{
		patience: patience,
		counter:  0,
		delta:    delta,
		bestLoss: math.MaxFloat64,
	}
}
func (es *EarlyStop) Reset() {
	es.counter = 0
	es.bestLoss = math.MaxFloat64
}
func (es *EarlyStop) CheckEarlyStop(valLoss float64) bool {
	if valLoss < es.bestLoss-es.delta {
		es.counter = 0
		es.bestLoss = valLoss
	} else {
		es.counter++

	}
	return es.counter >= es.patience
}
