package messages

import (
	"math"
	"math/rand"
)

const (
	DefaultTTL = 10 // rebroadcasts
	BFSTTL     = 60
	BcastId    = -1
)

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
	Velocity float64
	PosX     float64
	PosY     float64
	Angle    float64
	Degree   int
	SenderId int
	Round    int
	PCI      int
}

type CNNMessage struct {
	SenderId int
	CNN      any
	Round    int
}

type CHMessage struct {
	CH int
}
type CHRequestMessage struct {
	SenderId       int
	NewPotentialCH int
}
type CHResponseMessage struct {
	SenderId int
	Ok       bool
}

type CHFinalMessage struct {
	SenderId  int
	ClusterId int
}
type SubscribeMsg struct {
	SenderId int
}

type EndRoundMessage struct {
	Round int
}

type ClusterMessage struct {
	ClusterId int
	SenderId  int
}

type GetClusterRequest struct {
	SenderId int
}

type GetClusterResponse struct {
	SenderId  int
	ClusterId int
}

type WeightsMessage struct {
	SenderId int
	Weights  [][][]float64
}

type ClusterWeightsMessage struct {
	SenderId       int
	AverageWeights [][][]float64
}

type ClusterRequestMessage struct {
	SenderId int
}

type ClusterResponseMessage struct {
	SenderId  int
	ClusterId int
}

type BFSRequestMessage struct {
	SenderId int
	Level    int
	ParentId int
	Path     []int
	Target   int
}

type BFSResponseMessage struct {
	Level  int
	Path   []int
	Target int
}

func NewWeightsMessage(senderId int, weights [][][]float64) *WeightsMessage {
	return &WeightsMessage{
		SenderId: senderId,
		Weights:  weights,
	}
}

func NewBeaconMessage(velocity, posx, posy, angle float64, senderId, degree, round, pci int) *BeaconMessage {
	return &BeaconMessage{
		Velocity: velocity,
		PosX:     posx,
		PosY:     posy,
		Degree:   degree,
		Angle:    angle,
		SenderId: senderId,
		Round:    round,
		PCI:      pci,
	}
}

func NewCNNMessage(cnn any, senderId, round int) *CNNMessage {
	return &CNNMessage{
		SenderId: senderId,
		CNN:      cnn,
		Round:    round,
	}
}

func NewSubscribeMessage(senderId int) *SubscribeMsg {
	return &SubscribeMsg{
		SenderId: senderId,
	}
}
