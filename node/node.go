package node

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"slices"
	"sort"
	"sync"
	"time"

	"github.com/GiorgosMarga/vanet_d_clustering/messages"
)

const (
	a = 0.1
	b = 0.9
	c = 1
)

type Node struct {
	Id            int
	DHopNeighbors map[int]map[int]*Node
	Velocity      float64
	PosX          float64
	PosY          float64
	CNN           []*Node
	PCH           []*Node
	msgChan       chan *messages.Message
	finishChan    chan struct{}
	internalChan  chan any
	clusters      map[int][]int
	f             *os.File
	round         int
	finishedRound bool
	mtx           *sync.Mutex
	subscribers   map[int]struct{}
}

func NewNode(id, d int, posx, posy, velocity float64, filename string) *Node {
	f, err := os.OpenFile(filename, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal(err)
	}
	dhop := make(map[int]map[int]*Node)
	dhop[1] = make(map[int]*Node)
	return &Node{
		Id:            id,
		Velocity:      velocity,
		PosX:          posx,
		PosY:          posy,
		CNN:           make([]*Node, d+1),
		PCH:           make([]*Node, d+1),
		msgChan:       make(chan *messages.Message, 1000), //TODO: change this
		internalChan:  make(chan any, 1000),
		round:         1,
		mtx:           &sync.Mutex{},
		finishChan:    make(chan struct{}),
		f:             f,
		DHopNeighbors: dhop, // TODO: I dont need to know all the graph, only the 1-hop neighbors
		subscribers:   make(map[int]struct{}),
	}
}

func PrintPath(path []*Node) {
	for _, n := range path {
		fmt.Printf("%d ", n.Id)
	}
	fmt.Println()
}

func (n *Node) writePCH(d int) string {
	s := ""
	for i := range d {
		s += fmt.Sprintf("%d ", n.PCH[i].Id)
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
	if len(neighbor.DHopNeighbors[1]) == 0 {
		neighbor.DHopNeighbors[1] = make(map[int]*Node)
	}
	n.DHopNeighbors[1][neighbor.Id] = neighbor
	neighbor.DHopNeighbors[1][n.Id] = n
}

func (n *Node) Degree() int {
	return len(n.DHopNeighbors[1])
}
func (n *Node) isCH(d int) bool {
	return n.PCH[d].Id == n.Id
}

func (n *Node) GetdHopNeighs(d int) map[int]*Node {
	visited := make(map[int]struct{})
	q := make([]*Node, 1)
	q[0] = n
	visited[n.Id] = struct{}{}
	for i := 0; i < d; i++ {
		for _, cn := range q {
			q = q[1:]
			for _, nn := range cn.DHopNeighbors[1] {
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
	m := *msg
	m.Ttl--

	for _, cn := range n.DHopNeighbors[1] {
		timer := time.NewTimer(500 * time.Millisecond)
		select {
		case cn.msgChan <- &m:
			continue
		case <-timer.C:
			n.f.WriteString(fmt.Sprintf("Failed to send message (bcast) to: (%d)\n", cn.Id))
			timer.Stop()
		}
	}
}
func (n *Node) sendMsg(msg *messages.Message) {
	if c, ok := n.DHopNeighbors[1][msg.To]; ok {
		timer := time.NewTimer(500 * time.Millisecond)
		select {
		case c.msgChan <- msg:
			break
		case <-timer.C:
			n.f.WriteString(fmt.Sprintf("[%d]: Failed to send message to: (%d)\n", n.Id, msg.To))
			timer.Stop()
		}
		return
	}
	n.bcast(msg)
}

func (n *Node) sendBeacons() {
	for _, ne := range n.DHopNeighbors[1] {
		msg := messages.NewMessage(n.Id, ne.Id, messages.DefaultTTL, &messages.BeaconMessage{
			Velocity: n.Velocity,
			PosX:     n.PosX,
			PosY:     n.PosY,
			SenderId: n.Id,
			Round:    1,
		})
		n.sendMsg(msg)
	}
}
func (n *Node) advertiseCNN() {
	for subId := range n.subscribers {
		for round, cnn := range n.CNN {
			if cnn == nil {
				break
			}
			msg := messages.NewMessage(n.Id, subId, messages.DefaultTTL, &messages.CNNMessage{
				SenderId: n.Id,
				Round:    round + 1,
				CNN:      n.CNN[round],
			})
			n.sendMsg(msg)
		}
	}
}

func (n *Node) advertiseCluster() {
	d := len(n.PCH) - 1
	if n.PCH[d] == nil {
		return
	}
	for subId := range n.DHopNeighbors[1] {
		msg := messages.NewMessage(n.Id, subId, messages.DefaultTTL, &messages.ClusterMessage{
			Sender:    n.Id,
			ClusterId: n.PCH[d].Id,
			IsCh:      n.isCH(d),
		})
		n.sendMsg(msg)
	}
}

// Beacon sends a beacon message to all neighbors.
func (n *Node) Beacon(ctx context.Context) {
	t := time.NewTicker(100 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-t.C:
			n.sendBeacons()
			n.advertiseCNN()
			n.advertiseCluster()
		case <-ctx.Done():
			n.f.WriteString(fmt.Sprintf("[%d]: Done sending messages\n", n.Id))
			return
		}

	}
}
func (n *Node) Start(ctx context.Context, d int) {
	var (
		messagesReceived = make(map[int]struct{})
	)

	// initialization
	n.CNN[0] = n
	n.PCH[0] = n

	for {
		select {
		case m := <-n.msgChan:
			if _, ok := messagesReceived[m.ID]; ok {
				continue
			}
			messagesReceived[m.ID] = struct{}{}
			if m.Ttl <= 0 {
				continue
			}
			if m.To != n.Id {
				n.bcast(m)
				continue
			}
			switch msg := m.Msg.(type) {
			case *messages.BeaconMessage:
				if msg.Round != n.round {
					continue
				}
			case *messages.CNNMessage:
				if msg.Round != n.round {
					continue
				}
			case *messages.SubscribeMsg:
				senderId := m.Msg.(*messages.SubscribeMsg).SenderId
				n.subscribers[senderId] = struct{}{}
				continue
			}
			n.internalChan <- m.Msg
		case <-ctx.Done():
			n.f.WriteString(fmt.Sprintf("[%d]: Terminating...\n", n.Id))
			return
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
}

// exceptions are not used for now.
func (n *Node) Exceptions(d int) {
	var (
		clusterMsg = messages.NewClusterMessage(n.PCH[d].Id, n.Id, n.isCH(d))
		clusters   = make(map[int][]int)
	)
	for _, neighbor := range n.DHopNeighbors[1] {
		msg := messages.NewMessage(n.Id, neighbor.Id, messages.DefaultTTL, clusterMsg)
		n.sendMsg(msg)
	}

	for range len(n.DHopNeighbors[1]) {
		message := <-n.internalChan
		chMsg, ok := message.(*messages.ClusterMessage)
		if !ok {
			// fmt.Printf("[%d]: Received invalid message: %+v. Expected ClusterMessage\n", n.Id, message)
			continue
		}

		if chMsg.IsCh {
			clusters[chMsg.ClusterId] = make([]int, 0)
		}
		clusters[chMsg.ClusterId] = append(clusters[chMsg.ClusterId], chMsg.Sender)
	}
	if _, ok := clusters[n.PCH[d].Id]; !ok {
		clusters[n.PCH[d].Id] = make([]int, 0)
	}
	clusters[n.PCH[d].Id] = append(clusters[n.PCH[d].Id], n.Id)

}
func (n *Node) RelativeMax(d int) {
	n.CNN[0] = n
	n.PCH[0] = n

	if len(n.DHopNeighbors[1]) == 0 {
		// node has no neighbors
		for i := 1; i <= d; i++ {
			n.CNN[i] = n
			n.PCH[i] = n
		}

		return
	}

	// In the first round, each node finds the CNN based on it's neighborhood.
	n.f.WriteString(fmt.Sprintf("Starting round: %d\n", n.round))
	// fmt.Printf("[%d]: Starting round: %d\n", n.Id, n.round)
	msgs := make(map[*messages.BeaconMessage]struct{})

	for len(msgs) < len(n.DHopNeighbors[1]) {
		newMsg := <-n.internalChan
		beaconMessage, ok := newMsg.(*messages.BeaconMessage)
		if !ok {
			continue
		}
		if beaconMessage.Round != 1 {
			continue
		}
		msgs[beaconMessage] = struct{}{}
	}

	minRelativeMob := math.MaxFloat64
	for msg := range msgs {
		cnn := n.DHopNeighbors[1][msg.SenderId]
		relativeMobility := n.CNN[n.round-1].GetRelativeMobility(msg.Velocity, msg.PosX, msg.PosY, cnn.Degree(), cnn.PCI())
		n.f.WriteString(fmt.Sprintf("Comparing CNN: %d with %d (%f)\n", n.CNN[n.round-1].Id, msg.SenderId, relativeMobility))
		// fmt.Printf("[%d]: Comparing CNN: %d with %d (%f)\n", n.Id, n.CNN[n.round-1].Id, msg.SenderId, relativeMobility)
		if (relativeMobility < minRelativeMob) || (relativeMobility == minRelativeMob && n.CNN[1].Degree() < cnn.Degree()) {
			n.CNN[1] = cnn
			minRelativeMob = relativeMobility
		}
	}
	if n.CNN[1].Degree() > n.PCH[0].Degree() || (n.CNN[1].Degree() == n.PCH[0].Degree() && n.CNN[1].Velocity < n.PCH[0].Velocity) {
		n.PCH[1] = n.CNN[1]
	} else {
		n.PCH[1] = n.PCH[0]
	}

	n.f.WriteString(fmt.Sprintf("[%d]: Round 1: CNN: %d, PCH: %d\n", n.Id, n.CNN[1].Id, n.PCH[1].Id))
	// fmt.Printf("[%d]: Round 1: CNN: %d, PCH: %d\n", n.Id, n.CNN[1].Id, n.PCH[1].Id)

	// Each node has to subscribe to its potential pch to be able to receive CNN messages
	subMsg := messages.NewMessage(n.Id, n.PCH[1].Id, messages.DefaultTTL, messages.NewSubscribeMessage(n.Id))
	n.sendMsg(subMsg)

	// for the rest of the rounds the CNN is selected based on what the previous CNN has selected
	for n.round = 2; n.round <= d; {
		n.f.WriteString(fmt.Sprintf("Starting round: %d\n", n.round))
		newMsg := <-n.internalChan
		cnnMessage, ok := newMsg.(*messages.CNNMessage)
		if !ok {
			// fmt.Printf("[%d]: Received invalid type message: %+v, expected: CNNMessage\n", n.Id, newMsg)
			continue
		}
		if cnnMessage.Round != n.round {
			continue
		}
		// fmt.Printf("[%d]: Received CNN message from: (%d) -> %+v\n", n.Id, cnnMessage.SenderId, cnnMessage)
		cnn, ok := cnnMessage.CNN.(*Node)
		if !ok {
			// fmt.Printf("[%d]: Received invalid cnn: %+v, expected: Node\n", n.Id, cnn)
			continue
		}
		n.CNN[n.round] = cnn
		if cnn.Degree() > n.PCH[n.round-1].Degree() || (cnn.Degree() == n.PCH[n.round-1].Degree() && cnn.Velocity < n.PCH[n.round-1].Velocity) {
			n.PCH[n.round] = cnn
		} else {
			n.PCH[n.round] = n.PCH[n.round-1]
		}
		n.f.WriteString(fmt.Sprintf("CNN: %d\tPCH: %d\n", n.CNN[n.round].Id, n.PCH[n.round].Id))
		// fmt.Printf("[%d]: CNN: %d\tPCH: %d\n", n.Id, n.CNN[n.round].Id, n.PCH[n.round].Id)
		n.round++
	}
	n.f.WriteString(fmt.Sprintf("CNN: %s\nPCH: %s\n", n.writeCNN(d), n.writePCH(d)))
	n.f.WriteString(fmt.Sprintf("Finished all rounds my CH: %d\n", n.PCH[d].Id))
	// fmt.Printf("[%d]: Finished all rounds my CH: %d\n", n.Id, n.PCH[d].Id)
}
func (n *Node) GetRelativeMobility(vel float64, x, y float64, degree, pci int) float64 {
	dx := math.Pow(n.PosX-x, 2)
	dy := math.Pow(n.PosY-y, 2)
	dxy := math.Sqrt(dx + dy)
	// Using degree
	return a*dxy + b*math.Abs(n.Velocity-vel) + c*(float64(n.Degree())-float64(degree))
	// Using PCI
	// return a*dxy + b*math.Abs(n.Velocity-vel) + c*(float64(n.PCI())-float64(n.PCI()))

}
func (n *Node) FindPath(tNode *Node) []*Node {
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
			return cState.path[1:]
		}
		visited[cState.node.Id] = struct{}{}
		q = q[1:]
		for _, n := range cState.node.DHopNeighbors[1] {
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

func (n *Node) PCI() int {
	pciTable := make([]int, len(n.DHopNeighbors[1]))
	idx := 0
	for _, node := range n.DHopNeighbors[1] {
		pciTable[idx] = node.Degree()
	}
	sort.Ints(pciTable)
	slices.Reverse(pciTable)
	for idx := range len(pciTable) {
		if idx+1 <= pciTable[idx] {
			return idx + 1
		}
	}

	return len(pciTable)
}
