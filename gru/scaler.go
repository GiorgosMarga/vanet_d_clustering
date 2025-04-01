package gru

import "math"

type Scaler struct {
	xmin float64
	xmax float64
}

func NewScaler() *Scaler {
	return &Scaler{
		xmin: math.MaxFloat64,
		xmax: math.MinInt64,
	}
}

func (s *Scaler) FitTransform(data [][][]float64) [][][]float64 {
	s.Fit(data)
	return s.Transform(data)
}

func (s *Scaler) Fit(data [][][]float64) {
	for i := range data {
		for ii := range data[i] {
			for iii := range data[i][ii] {
				if data[i][ii][iii] < s.xmin {
					s.xmin = data[i][ii][iii]
				} else if data[i][ii][iii] > s.xmax {
					s.xmax = data[i][ii][iii]
				}
			}
		}
	}
}

func (s *Scaler) Transform(data [][][]float64) [][][]float64 {
	scaled := make([][][]float64, len(data))

	for i := range data {
		scaled[i] = make([][]float64, len(data[i]))
		for ii := range data[i] {
			scaled[i][ii] = make([]float64, len(data[i][ii]))
			for iii := range data[i][ii] {
				scaled[i][ii][iii] = (data[i][ii][iii] - s.xmin) / (s.xmax - s.xmin)
			}
		}
	}
	return scaled
}

func (s *Scaler) InverseTransform(scaledData [][][]float64) [][][]float64 {
	original := make([][][]float64, len(scaledData))

	for i := range scaledData {
		original[i] = make([][]float64, len(scaledData[i]))
		for ii := range scaledData[i] {
			original[i][ii] = make([]float64, len(scaledData[i][ii]))
			for iii := range scaledData[i][ii] {
				original[i][ii][iii] = scaledData[i][ii][iii]*(s.xmax-s.xmin) + s.xmin
			}
		}
	}
	return original
}
