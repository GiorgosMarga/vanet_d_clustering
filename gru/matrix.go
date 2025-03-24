package gru

import (
	"fmt"
	"math/rand/v2"
)

func getDims(matrix [][]float64) (int, int) {
	return len(matrix), len(matrix[0])
}

func printDims(matrix [][]float64, matrixName string) {
	a, b := getDims(matrix)
	fmt.Printf("%s: (%dx%d)\n", matrixName, a, b)
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
