package gru

import (
	"math/rand"
)

func dropoutInput(input [][]float64, rate float64) [][]float64 {
	if rate == 0 {
		return input
	}
	if rate < 0 || rate > 1 {
		panic("Invalid rate, should be between 0 and 1")
	}
	scaledRate := int(rate) * 10
	output := make([][]float64, len(input))
	numOfDrops := int(rate * float64(len(input)))
	for i := range len(input) {
		output[i] = make([]float64, 1)
		r := rand.Intn(10)
		if (r < scaledRate && numOfDrops > 0) || numOfDrops-i >= 0 {
			numOfDrops--
			output[i] = []float64{0.0}
		} else {
			output[i] = []float64{input[i][0] * (1 - rate)}
		}
	}
	return output

}
