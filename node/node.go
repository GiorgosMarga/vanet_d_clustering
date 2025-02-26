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

func NewNode(id int, posx, posy, velocity float64, filename string) *Node {
	f, err := os.OpenFile(filename, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal(err)
	}
	return &Node{
		Id:            id,
		Velocity:      velocity,
		PosX:          posx,
		PosY:          posy,
		CNN:           make([]*Node, 5),
		PCH:           make([]*Node, 5),
		msgChan:       make(chan *messages.Message, 1000), //TODO: change this
		internalChan:  make(chan any, 1000),
		round:         0,
		mtx:           &sync.Mutex{},
		finishChan:    make(chan struct{}),
		f:             f,
		DHopNeighbors: make(map[int]map[int]*Node), // TODO: I dont need to know all the graph, only the 1-hop neighbors
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
	if len(n.DHopNeighbors[1]) == 0 {
		n.DHopNeighbors[1] = make(map[int]*Node)
	}
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
			fmt.Printf("[%d]: Failed to send message to: (%d)\n", n.Id, msg.To)
			timer.Stop()
		}
		return
	}
	n.bcast(msg)
}

// beacon sends a beacon message to all neighbors.
func (n *Node) beacon(ctx context.Context) {
	t := time.NewTicker(100 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-t.C:
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
		case <-ctx.Done():
			fmt.Printf("[%d]: Done sending messages\n", n.Id)
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
	n.round = 1

	go n.beacon(ctx)

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
			fmt.Printf("[%d]: Terminating...\n", n.Id)
			// close(n.msgChan)
			return
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
}

// exceptions are not used for now.
func (n *Node) exceptions(d int) {
	cms := make([]*Node, 0)
	if n.isCH(d) {
		for i := 1; i <= d; i++ {
			for _, n := range n.DHopNeighbors[i] {
				if n.PCH[d] == n {
					cms = append(cms, n)
				}
			}
		}
		if len(cms) == 0 || len(cms) == 1 {
			fmt.Println("CMS is 0", cms)
		}
		return
	}

	potentialCh := n.PCH[d]
	pathToCh := n.FindPath(potentialCh)

	for i := 1; i < len(pathToCh); i++ {
		node := pathToCh[i]
		_ = node
	}

}
func (n *Node) RelativeMax(d int) {
	n.CNN[0] = n
	n.PCH[0] = n

	// In the first round, each node finds the CNN based on it's neighbor.
	// n.f.WriteString(fmt.Sprintf("Starting round: %d\n", n.round))
	fmt.Printf("[%d]: Starting round: %d\n", n.Id, n.round)
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
		relativeMobility := n.CNN[n.round-1].GetRelativeMobility(msg.Velocity, msg.PosX, msg.PosY, cnn.Degree())
		n.f.WriteString(fmt.Sprintf("Comparing CNN: %d with %d (%f)\n", n.CNN[n.round-1].Id, msg.SenderId, relativeMobility))
		fmt.Printf("[%d]: Comparing CNN: %d with %d (%f)\n", n.Id, n.CNN[n.round-1].Id, msg.SenderId, relativeMobility)
		if (relativeMobility < minRelativeMob) || (relativeMobility == minRelativeMob && n.CNN[1].Degree() < cnn.Degree()) {
			n.CNN[1] = cnn
			minRelativeMob = relativeMobility
		}
	}
	if n.CNN[1].Degree() > n.PCH[0].Degree() || (n.CNN[1].Degree() == n.PCH[0].Degree() && n.CNN[1].Velocity < n.PCH[0].Velocity) {
		n.PCH[1] = n.CNN[1]
	} else {
		n.PCH[1] = n.PCH[0]
		// chose itself
		// n.f.WriteString(fmt.Sprintf("[%d]: Round 1: CNN: %d, PCH: %d\n", n.Id, n.CNN[1].Id, n.PCH[1].Id))
		// fmt.Printf("[%d]: Round 1: CNN: %d, PCH: %d\n", n.Id, n.CNN[1].Id, n.PCH[1].Id)
		// n.searchNeighborhood(d)
		// return
	}

	n.f.WriteString(fmt.Sprintf("[%d]: Round 1: CNN: %d, PCH: %d\n", n.Id, n.CNN[1].Id, n.PCH[1].Id))
	fmt.Printf("[%d]: Round 1: CNN: %d, PCH: %d\n", n.Id, n.CNN[1].Id, n.PCH[1].Id)

	// Each node has to subscribe to its potential pch to be able to receive CNN messages
	subMsg := messages.NewMessage(n.Id, n.PCH[1].Id, messages.DefaultTTL, messages.NewSubscribeMessage(n.Id))
	n.sendMsg(subMsg)

	// for the rest of the rounds the CNN is selected based on what the previous CNN has selected
	for n.round = 2; n.round <= d; n.round++ {
		n.f.WriteString(fmt.Sprintf("Starting round: %d\n", n.round))
		newMsg := <-n.internalChan
		cnnMessage, ok := newMsg.(*messages.CNNMessage)
		if !ok {
			fmt.Printf("[%d]: Received invalid type message: %+v, expected: CNNMessage\n", n.Id, newMsg)
			continue
		}
		if cnnMessage.Round != n.round {
			continue
		}
		fmt.Printf("[%d]: Received CNN message from: (%d) -> %+v\n", n.Id, cnnMessage.SenderId, cnnMessage)
		cnn, ok := cnnMessage.CNN.(*Node)
		if !ok {
			fmt.Printf("[%d]: Received invalid cnn: %+v, expected: Node\n", n.Id, cnn)
			continue
		}
		n.CNN[n.round] = cnn
		if cnn.Degree() > n.PCH[n.round-1].Degree() || (cnn.Degree() == n.PCH[n.round-1].Degree() && cnn.Velocity < n.PCH[n.round-1].Velocity) {
			n.PCH[n.round] = cnn
		} else {
			n.PCH[n.round] = n.PCH[n.round-1]
		}
		n.f.WriteString(fmt.Sprintf("CNN: %d\tPCH: %d\n", n.CNN[n.round].Id, n.PCH[n.round].Id))
		fmt.Printf("[%d]: CNN: %d\tPCH: %d\n", n.Id, n.CNN[n.round].Id, n.PCH[n.round].Id)
	}
	n.f.WriteString(fmt.Sprintf("CNN: %s\nPCH: %s\n", n.writeCNN(d), n.writePCH(d)))
	n.f.WriteString(fmt.Sprintf("Finished all rounds my CH: %d\n", n.PCH[d].Id))
	fmt.Printf("[%d]: Finished all rounds my CH: %d\n", n.Id, n.PCH[d].Id)
}
func (n *Node) GetRelativeMobility(vel float64, x, y float64, degree int) float64 {
	dx := math.Pow(n.PosX-x, 2)
	dy := math.Pow(n.PosY-y, 2)
	dxy := math.Sqrt(dx + dy)
	return a*dxy + b*math.Abs(n.Velocity-vel) + c*(float64(n.Degree())-float64(degree))

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
			return cState.path
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
func printPath(path []*Node) string {
	s := ""
	for _, n := range path {
		s += fmt.Sprintf("%d ", n.Id)
	}
	return s + "\n"
}

func (n *Node) searchNeighborhood(d int) {
	for n.round = 2; n.round <= d; n.round++ {
		n.f.WriteString(fmt.Sprintf("Starting round: %d (searching neighborhood)\n", n.round))
		neighborhood := n.GetdHopNeighs(n.round)
		minRelativeMob := math.MaxFloat64
		for _, neighbor := range neighborhood {
			relativeMobility := n.GetRelativeMobility(neighbor.Velocity, neighbor.PosX, neighbor.PosY, neighbor.Degree())
			if (relativeMobility < minRelativeMob) || (relativeMobility == minRelativeMob && n.CNN[1].Degree() < neighbor.Degree()) {
				n.CNN[n.round] = neighbor
				minRelativeMob = relativeMobility
			}

		}
		cnn := n.CNN[n.round]
		if cnn.Degree() > n.PCH[n.round-1].Degree() || (cnn.Degree() == n.PCH[n.round-1].Degree() && cnn.Velocity < n.PCH[n.round-1].Velocity) {
			n.PCH[n.round] = cnn
		} else {
			n.PCH[n.round] = n.PCH[n.round-1]
		}
		n.f.WriteString(fmt.Sprintf("CNN: %d\tPCH: %d\n", n.CNN[n.round].Id, n.PCH[n.round].Id))
		fmt.Printf("[%d]: CNN: %d\tPCH: %d\n", n.Id, n.CNN[n.round].Id, n.PCH[n.round].Id)
	}
}
