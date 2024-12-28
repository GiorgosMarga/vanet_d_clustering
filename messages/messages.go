package messages

import (
	"math"
	"math/rand"
)

const (
	BcastMsg = iota
	BeaconMsg
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
	Velocity float64
	PosX     float64
	PosY     float64
	Degree   int
	SenderId int
}

type PCHMessage struct {
	Node  any
	Round int
}

func NewBeaconMessage(velocity, posx, posy float64, senderId, degree int) *BeaconMessage {
	return &BeaconMessage{
		Velocity: velocity,
		PosX:     posx,
		PosY:     posy,
		Degree:   degree,
		SenderId: senderId,
	}
}

func NewPCHMessage(node any, round int) *PCHMessage {
	return &PCHMessage{
		Node:  node,
		Round: round,
	}
}
