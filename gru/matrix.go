package gru

import "fmt"

func getDims(matrix [][]float64) (int, int) {
	return len(matrix), len(matrix[0])
}

func printDims(matrix [][]float64, matrixName string) {
	a, b := getDims(matrix)
	fmt.Printf("%s: (%dx%d)\n", matrixName, a, b)
}
