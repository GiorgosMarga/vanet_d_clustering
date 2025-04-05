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

// BeaconMessage is used for sending the position and velocity of the node.
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

// CNNMessage is used for sending the CNN node for round >=1 in relative max.
type CNNMessage struct {
	SenderId int
	CNN      any
	Round    int
}

// CHRequestMessage is used for asking a node to have a new ch.
// It is used in exception 2
type CHRequestMessage struct {
	SenderId       int
	NewPotentialCH int
}

// CHResponseMessage is used for responding to the request.
// Ok is true if distance d is satisfied
type CHResponseMessage struct {
	SenderId int
	Ok       bool
}

// CHFinalMessage is used for sending the final CH of a node after exception 2.
type CHFinalMessage struct {
	SenderId  int
	ClusterId int
}

// SubscribeMsg is used for subscribing to a node to receive CNN messages from it.Used in relative max.
type SubscribeMsg struct {
	SenderId int
}

type EndRoundMessage struct {
	Round int
}

// ClusterMessage is used for sending the cluster id of a node.
// It is bcasted periodically from each node
type ClusterMessage struct {
	ClusterId int
	SenderId  int
}

// GetClusterRequest is sent to learn the CH of a node.
type GetClusterRequest struct {
	SenderId int
}

// GetClusterResponse is the response to the GetClusterRequest
type GetClusterResponse struct {
	SenderId  int
	ClusterId int
}

// WeightsMessage is sent from nodes to ch
type WeightsMessage struct {
	SenderId int
	Weights  [][][]float64
}

// ClusterWeightsMessage is sent from ch to ch, to calculate the average
type ClusterWeightsMessage struct {
	SenderId       int
	AverageWeights [][][]float64
}

// BFSRequestMessage is used to find a path between 2 nodes
type BFSRequestMessage struct {
	SenderId int
	Level    int
	ParentId int
	Path     []int
	Target   int
}

// BFSResponseMessage is sent from target node back to the first node
type BFSResponseMessage struct {
	Level  int
	Path   []int
	Target int
}
