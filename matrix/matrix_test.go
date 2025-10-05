package matrix

import (
	"fmt"
	"math/rand"
	"testing"
)

func createArr(n int) [][]float64 {
	arr := make([][]float64, n)
	for i := range n {
		arr[i] = make([]float64, n)
		for j := range n {
			arr[i][j] = float64(rand.Int() % 1_000_000)
		}
	}
	return arr
}

// createArrFromArray is used to produce a similar array
// to the given array.
func createArrFromArray(arr1 [][]float64) [][]float64 {
	arr := make([][]float64, len(arr1))
	for i := range arr {
		b := make([]float64, len(arr1[i]))
		for j := range len(b) {
			b[j] = arr1[i][j] + float64(rand.Int()%2)
		}
		arr[i] = b
	}

	return arr
}

func BenchmarkFlatten(t *testing.B) {
	arr := createArr(1000)

	for range t.N {
		Flatten(arr)
	}
}

func BenchmarkFlatten2(t *testing.B) {
	arr := createArr(1000)

	for range t.N {
		Flatten(arr)
	}
}

func TestFFT(t *testing.T) {
	// arr1 := [][]float64{{1}, {2}, {3}, {4}, {5}}
	// arr2 := [][]float64{{1.01}, {2.01}, {3.01}, {4.01}, {5.01}} // very close to inputA
	// // inputC := []float64{10, 20, 30, 40, 50}
	arr1 := createArr(1024)
	// arr2 := createArr(1024)
	arr2 := createArrFromArray(arr1)

	f1 := Flatten(arr1)
	f2 := Flatten(arr2)

	fft1 := FFT(f1)
	fft2 := FFT(f2)

	// fmt.Println(f1[:10])
	// fmt.Println(f2[:10])

	fmt.Println(CalculateMatDistance(Parseval(fft1[:10]), Parseval(fft2[:10])))
	fmt.Println(CalculateMatDistance(Parseval(fft1[:15]), Parseval(fft2[:15])))
	fmt.Println(CalculateMatDistance(Parseval(fft1[:20]), Parseval(fft2[:20])))

}
