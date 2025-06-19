package messages

import (
	"math"
	"math/rand"
)

const (
	BcastId = -1
)

const DefaultTTL = 10 // rebroadcasts

type Message struct {
	ID   int
	From int
	To   int
	Msg  any
	Ttl  int
}

func NewMessage(from, to, ttl int, msg any) *Message {
	return &Message{
		ID:   rand.Intn(math.MaxInt),
		From: from,
		To:   to,
		Msg:  msg,
		Ttl:  ttl,
	}
}

type BeaconMessage struct {
	Velocity          float64
	PosX              float64
	PosY              float64
	Angle             float64
	Degree            int
	SenderId          int
	Round             int
	PCI               int
	ClusterHeadRounds int
}

type CNNMessage struct {
	SenderId int
	CNN      any
	Round    int
}

type SubscribeMsg struct {
	SenderId int
}

type ClusterMessage struct {
	ClusterId int
	IsCh      bool
	Sender    int
}

type WeightsMessage struct {
	SenderId int
	Weights  [][][]float64
}

type ClusterWeightsMessage struct {
	SenderId       int
	AverageWeights [][][]float64
}

type ParsevalMessage struct {
	SenderId       int
	ParsevalValues []float64
	Velocity       float64
}
