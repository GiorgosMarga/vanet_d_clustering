package matrix

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
)

func getDims(matrix [][]float64) (int, int) {
	return len(matrix), len(matrix[0])
}

func PrintDims(matrix [][]float64, matrixName string) {
	a, b := getDims(matrix)
	fmt.Printf("%s: (%dx%d)\n", matrixName, a, b)
}
func ElementMatrixMul(a, b [][]float64) [][]float64 {
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
func IdentityMatrix(rows, cols int) [][]float64 {
	res := make([][]float64, rows)
	for row := range res {
		res[row] = make([]float64, cols)
		for colIdx := range res[row] {
			res[row][colIdx] = 1
		}
	}
	return res
}
func MatrixMul(a, b [][]float64) [][]float64 {
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

func MatrixAdd(a, b [][]float64) [][]float64 {
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

func MatrixSub(a, b [][]float64) [][]float64 {
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

func MatrixSubWithScalar(a, b [][]float64, scale float64) [][]float64 {
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

func MatrixOuterProduct(a, b [][]float64) [][]float64 {
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
func RandomMatrix(rows, cols int, scale float64) [][]float64 {
	r := rand.New(rand.NewSource(10))
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = (r.Float64()*2 - 1) * scale // Range: [-scale, scale]
		}
	}
	return matrix
}

func TransposeMatrix(a [][]float64) [][]float64 {
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
func SumRows(matrix [][]float64) [][]float64 {
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

func MatrixScale(a [][]float64, scaler float64) [][]float64 {
	res := make([][]float64, len(a))
	for row := range a {
		res[row] = make([]float64, len(a[row]))
		for colIdx := range a[row] {
			res[row][colIdx] = a[row][colIdx] * scaler
		}
	}
	return res
}

func MatrixAverage(matrices [][][]float64) [][]float64 {
	if len(matrices) == 1 {
		return matrices[0]
	}
	// matrix -> rows x col
	rows := len(matrices[0])
	cols := len(matrices[0][0])

	for _, matrix := range matrices {
		if len(matrix) != rows || len(matrix[0]) != cols {
			panic(fmt.Sprintf("Invalid matrix size: Expected: (%dx%d), got: (%dx%d)\n", rows, cols, len(matrix), len(matrix[0])))
		}
	}

	t := make([][]float64, rows)

	for row := range t {
		t[row] = make([]float64, cols)
		for colIdx := range t[row] {
			for _, m := range matrices {
				t[row][colIdx] += m[row][colIdx]
			}
		}
	}
	return MatrixScale(t, 1/float64(len(matrices)))
}

func CalculateAverageWeights(matrices [][][][]float64) [][][]float64 {
	if len(matrices) == 1 {
		return matrices[0]
	}
	average := make([][][]float64, len(matrices[0]))
	for weightType := range matrices[0] {
		weightsToProcess := make([][][]float64, len(matrices))
		for idx := range matrices {
			weightsToProcess[idx] = matrices[idx][weightType]
		}
		average[weightType] = MatrixAverage(weightsToProcess)
	}
	return average
}

func FFT(x []float64) []complex128 {
	N := len(x)
	res := make([]complex128, N)
	for i := range len(res) {
		res[i] = complex(x[i], 0)
	}
	fftRec(res, N)

	// normalize values
	sqrtN := math.Sqrt(float64(N))
	for i := range res {
		res[i] /= complex(sqrtN, 0)
	}
	return res
}

func fftRec(x []complex128, N int) {
	if N <= 1 {
		return
	}

	odd := make([]complex128, N/2)
	even := make([]complex128, N/2)

	for i := range N / 2 {
		even[i] = x[i*2]
		odd[i] = x[i*2+1]
	}

	fftRec(odd, N/2)
	fftRec(even, N/2)

	for k := range N / 2 {

		t := cmplx.Exp(complex(0, -2.0*math.Pi*float64(k)/float64(N))) * odd[k]
		x[k] = even[k] + t
		x[N/2+k] = even[k] - t
	}
}

func Flatten(arr [][]float64) []float64 {
	newArr := make([]float64, len(arr)*len(arr[0]))
	for i, a := range arr {
		copy(newArr[i*len(arr[0]):], a)
	}
	return newArr
}

func Flatten3D(arr [][][]float64) []float64 {
	newArr := make([]float64, 0, len(arr)*len(arr[0])*len(arr[0][0]))

	for _, row := range arr {
		newArr = append(newArr, Flatten(row)...)
	}

	return newArr
}

func Parseval(arr []complex128) []float64 {
	res := make([]float64, len(arr))

	for i, c := range arr {
		res[i] = math.Pow(cmplx.Abs(c), 2)
	}
	return res
}

func CalculateMatDistance(arr1, arr2 []float64) float64 {
	res := 0.0

	if len(arr1) != len(arr2) {
		panic("Different length")
	}
	for i := range arr1 {
		res += math.Pow(arr1[i]-arr2[i], 2)
	}
	return math.Sqrt(res) / float64(len(arr1))
}

func CalculateEuclDistance(arr1, arr2 []complex128) float64 {
	res := 0.0

	if len(arr1) != len(arr2) {
		panic("Different length")
	}
	for i := range arr1 {
		dx := math.Pow(real(arr1[i])-real(arr2[i]), 2)
		dy := math.Pow(imag(arr1[i])-imag(arr2[i]), 2)
		res += math.Sqrt(dx + dy)

	}
	return res
}

func CalculateCosineSimilarity(a, b [][][]float64) float64 {
	fA, fB := Flatten3D(a), Flatten3D(b)

	var dot, normA, normB float64
	for i := range a {
		dot += fA[i] * fB[i]
		normA += fA[i] * fA[i]
		normB += fB[i] * fB[i]
	}

	if normA == 0 || normB == 0 {
		return 0 // avoid division by zero
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))

}
