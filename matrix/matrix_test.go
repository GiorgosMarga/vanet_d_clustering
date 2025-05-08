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
			arr[i][j] = rand.Float64()
		}
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
		Flatten2(arr)
	}
}

func TestFFT(t *testing.T) {
	arr1 := createArr(1024)
	arr2 := createArr(1024)

	f1 := Flatten(arr1)
	f2 := Flatten(arr2)

	fft1 := FFT(f1)
	fft2 := FFT(f2)

	
	

}
