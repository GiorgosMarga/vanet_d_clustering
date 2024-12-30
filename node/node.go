package node

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"sync"
	"time"

	"github.com/GiorgosMarga/vanet_d_clustering/messages"
)

const (
	a = 0.9
	b = 0.1
)

type Node struct {
	Id               int
	Neighbors        map[int]*Node
	dHopNeighbors    map[int]*Node
	Velocity         float64
	PosX             float64
	PosY             float64
	CNN              []*Node
	PCH              []*Node
	msgChan          chan *messages.Message
	neighChans       map[int]chan *messages.Message
	messagesReceived map[int]struct{}
	f                *os.File
	round            int
	mtx              sync.Mutex
}

func NewNode(id int, posx, posy, velocity float64, filename string) *Node {
	f, err := os.OpenFile(filename, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal(err)
	}
	return &Node{
		Id:               id,
		Neighbors:        make(map[int]*Node),
		Velocity:         velocity,
		PosX:             posx,
		PosY:             posy,
		CNN:              make([]*Node, 5),
		PCH:              make([]*Node, 5),
		msgChan:          make(chan *messages.Message, 1000),
		neighChans:       make(map[int]chan *messages.Message),
		messagesReceived: make(map[int]struct{}),
		round:            0,
		mtx:              sync.Mutex{},
		f:                f,
	}
}

func (n *Node) writePCH(d int) string {
	s := ""
	for i := range d {
		s += fmt.Sprintf("%d(%d) ", n.PCH[i].Id, n.PCH[i].PCH[d].Id)
	}
	s += "\n"
	return s
}
func (n *Node) writeCNN(d int) string {
	s := ""
	for i := range d {
		s += fmt.Sprintf("%d ", n.CNN[i].Id)
	}
	s += "\n"
	return s
}
func (n *Node) printPCH(d int) {
	fmt.Printf("[%d]: ", n.Id)
	for i := range d {
		fmt.Printf("%d(%d) ", n.PCH[i].Id, n.PCH[i].PCH[d].Id)
	}
	fmt.Println()
}

func (n *Node) AddNeighbor(neighbor *Node) {
	n.Neighbors[neighbor.Id] = neighbor
	n.neighChans[neighbor.Id] = neighbor.msgChan
	neighbor.Neighbors[n.Id] = n
	neighbor.neighChans[n.Id] = n.msgChan
}

func (n *Node) Degree() int {
	return len(n.Neighbors)
}
func (n *Node) isCH(d int) bool {
	return n.PCH[d].Id == n.Id
}

func (n *Node) Printf(s string) {
	n.f.WriteString(s)
}
func (n *Node) GetdHopNeighs(d int) map[int]*Node {
	visited := make(map[int]struct{})
	q := make([]*Node, 1)
	q[0] = n
	visited[n.Id] = struct{}{}
	for i := 0; i < d; i++ {
		for _, cn := range q {
			q = q[1:]
			for _, nn := range cn.Neighbors {
				if _, ok := visited[nn.Id]; ok {
					continue
				}
				visited[nn.Id] = struct{}{}
				q = append(q, nn)
			}
		}
	}
	neighs := make(map[int]*Node)
	for _, cn := range q {
		neighs[cn.Id] = cn
	}
	return neighs
}

func (n *Node) bcast(msg *messages.Message) {
	for _, cn := range n.Neighbors {
		timer := time.NewTimer(500 * time.Millisecond)
		select {
		case n.neighChans[cn.Id] <- msg:
			continue
		case <-timer.C:
			n.f.WriteString(fmt.Sprintf("Failed to send message (bcast) to: (%d)\n", cn.Id))
			timer.Stop()
		}
	}
}

// func (n *Node) relativeMax(i int) {
// 	n.mtx.Lock()
// 	defer n.mtx.Unlock()
// 	idx := i + 1
// 	// fmt.Printf("[%d]: Round %d\n", n.Id, idx)
// 	neighs := n.GetdHopNeighs(idx)
// 	for nodeId := range neighs {
// 		msg := messages.NewMessage(n.Id, nodeId, messages.DefaultTTL, messages.NewBeaconMessage(n.Velocity, n.PosX, n.PosY, n.Id, n.Degree()))
// 		// check if its neighbor else bcast the message
// 		if c, ok := n.neighChans[nodeId]; ok {
// 			c <- msg
// 		} else {
// 			n.bcast(msg)
// 		}
// 	}
// 	// n.f.WriteString(fmt.Sprintf("Sent all beacon messages\n"))
// 	n.f.WriteString(fmt.Sprintf("Round: %d: CNN: %d, degree: %d\n", i, n.CNN[i].Id, n.CNN[i].Degree()))
// 	minMobility := math.MaxFloat64
// 	var CNN int
// 	messagesReceived := make(map[int]struct{})
// 	for len(messagesReceived) < len(neighs) {
// 		msg := <-n.internalChan
// 		if _, ok := messagesReceived[msg.SenderId]; ok {
// 			continue
// 		}
// 		relativeSpeed := n.CNN[i].getRelativeMobility(msg.Velocity, msg.PosX, msg.PosY)
// 		if relativeSpeed < minMobility || (relativeSpeed == minMobility && n.CNN[i].Degree() < msg.Degree) {
// 			minMobility = relativeSpeed
// 			CNN = msg.SenderId
// 		}
// 		// if n.Id == 4 {
// 		n.f.WriteString(fmt.Sprintf("CNN: %d comparing with %d (%f)\n", n.CNN[i].Id, msg.SenderId, relativeSpeed))
// 		// }
// 		messagesReceived[msg.SenderId] = struct{}{}
// 	}
// 	cnnNode := neighs[CNN]
// 	n.CNN[idx] = cnnNode
// 	if cnnNode.Degree() > n.PCH[i].Degree() {
// 		n.PCH[idx] = cnnNode
// 	} else {
// 		n.PCH[idx] = n.PCH[i]
// 	}

// 	// if n.Id == 4 {
// 	n.f.WriteString(fmt.Sprintf("PCH Id: %d\n", n.PCH[idx].Id))
// 	// }
// 	if n.CNN[idx].Degree() == n.CNN[idx-1].Degree() && n.CNN[idx].Velocity < n.CNN[idx-1].Velocity {
// 		msg := messages.NewMessage(n.Id, n.CNN[idx-1].Id, messages.DefaultTTL, messages.NewPCHMessage(n.CNN[idx], idx))
// 		n.sendMsg(msg)
// 	}
// }

// func (n *Node) checkExceptions(d int) {

// 	if n.isCH(d) {

// 	}

// 	potentialCH := n.PCH[d]
// 	pathToCh := n.findPath(potentialCH)

// 	for i := 1; i < len(pathToCh);i++ {
// 		node := pathToCh[i]

// 	}

// }
func (n *Node) sendMsg(msg *messages.Message) {
	if c, ok := n.neighChans[msg.To]; ok {
		timer := time.NewTimer(500 * time.Millisecond)
		select {
		case c <- msg:
			break
		case <-timer.C:
			fmt.Printf("[%d]: Failed to send message to: (%d)\n", n.Id, msg.To)
			timer.Stop()
		}
		return
	}
	n.bcast(msg)
}

func (n *Node) start(ctx context.Context, d int) {
	var (
		beaconMessages = make([]*messages.BeaconMessage, 0)
	)
	// initialization
	n.CNN[0] = n
	n.PCH[0] = n

	n.round = 1
	n.dHopNeighbors = n.GetdHopNeighs(n.round)
	for {
		select {
		case m := <-n.msgChan:
			if _, ok := n.messagesReceived[m.ID]; ok {
				continue
			}
			n.messagesReceived[m.ID] = struct{}{}
			if m.Ttl <= 0 {
				continue
			}
			if m.To != n.Id {
				newMsg := *m
				newMsg.Ttl -= 1
				n.bcast(&newMsg)
				continue
			}
			switch m.Msg.(type) {
			case *messages.BeaconMessage:
				beaconMessages = append(beaconMessages, m.Msg.(*messages.BeaconMessage))
				if len(beaconMessages) == len(n.dHopNeighbors) {
					n.relativeMax(beaconMessages)
					if n.round == 3 {
						msg := messages.NewMessage(n.Id, n.PCH[d].Id, messages.DefaultTTL, &messages.CHMessage{})
						n.bcast(msg)
					}
				}
			case *messages.PCHMessage:
				go n.handlePCHMessage(m.Msg.(*messages.PCHMessage))
			}
		case <-ctx.Done():
			// fmt.Printf("[%d]: Terminating...\n", n.Id)
			return
		}
	}
}
func (n *Node) relativeMax(msgs []*messages.BeaconMessage) {

	minRelativeMob := math.MaxFloat64
	for _, msg := range msgs {
		relativeMobility := n.CNN[n.round-1].getRelativeMobility(msg.Velocity, msg.PosX, msg.PosY)
		if relativeMobility < minRelativeMob {
			n.CNN[n.round] = n.dHopNeighbors[msg.SenderId]
		}
	}
	if n.CNN[n.round].Degree() == n.CNN[n.round-1].Degree() && n.CNN[n.round].Velocity < n.CNN[n.round-1].Velocity {
		n.f.WriteString("Need to notify cnni-1")
	}

	if n.CNN[n.round].Degree() > n.PCH[n.round-1].Degree() {
		n.PCH[n.round] = n.CNN[n.round]
	} else {
		n.PCH[n.round] = n.PCH[n.round-1]
	}

	n.round++

}
func (n *Node) handlePCHMessage(msg *messages.PCHMessage) {
	n.mtx.Lock()
	defer n.mtx.Unlock()
	if msg.Node.(*Node).Degree() > n.PCH[msg.Round].Degree() {
		fmt.Printf("[%d]: Setting new pch %d\n", n.Id, msg.Node.(*Node).Id)
		n.PCH[msg.Round] = msg.Node.(*Node)
	}
}

func (n *Node) getRelativeMobility(vel float64, x, y float64) float64 {
	dx := math.Pow(n.PosX-x, 2)
	dy := math.Pow(n.PosY-y, 2)
	dxy := math.Sqrt(dx + dy)
	return a*dxy + b*math.Abs(n.Velocity-vel)

}
func (n *Node) findPath(tNode *Node) []*Node {
	type state struct {
		node *Node
		path []*Node
	}
	visited := make(map[int]struct{})
	q := make([]state, 1)
	q[0] = state{
		node: n,
		path: []*Node{n},
	}
	for len(q) > 0 {
		cState := q[0]
		if cState.node.Id == tNode.Id {
			return cState.path
		}
		visited[cState.node.Id] = struct{}{}
		q = q[1:]
		for _, n := range cState.node.Neighbors {
			if _, ok := visited[n.Id]; ok {
				continue
			}
			newPath := make([]*Node, len(cState.path))
			copy(newPath, cState.path)
			newPath = append(newPath, n)
			newState := state{
				node: n,
				path: newPath,
			}
			q = append(q, newState)
		}
	}
	return []*Node{}
}
func printPath(path []*Node) {
	for _, n := range path {
		fmt.Printf("%d ", n.Id)
	}
	fmt.Println()
}
