package node

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"time"

	"github.com/GiorgosMarga/vanet_d_clustering/gru"
	"github.com/GiorgosMarga/vanet_d_clustering/messages"
	"github.com/GiorgosMarga/vanet_d_clustering/utils"
)

const (
	a                   = 0.1
	b                   = 0.9
	c                   = 1
	TrainSizePercentage = 0.8
	HiddenStateSize     = 16
	InputSize           = 4
	Epochs              = 30
	BatchSize           = 10
	d                   = 2
)

type Node struct {
	Id             int
	DHopNeighbors  map[int]*Node
	Velocity       float64
	PosX           float64
	PosY           float64
	Angle          float64
	CNN            []*Node
	PCH            []*Node
	msgChan        chan *messages.Message
	finishChan     chan struct{}
	internalChan   chan any
	f              *os.File
	round          int
	subscribers    map[int]struct{}
	gru            *gru.GRU
	weightMessages map[int]*messages.WeightsMessage
}

func NewNode(id, d int, posx, posy, velocity, angle float64, filename string) *Node {
	f, err := os.OpenFile(filename, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0644)
	if err != nil {
		log.Fatal(err)
	}

	gru := gru.NewGRU(HiddenStateSize, InputSize, 10, gru.MeanSquareError, 0.001)

	if err := gru.ParseFile(filepath.Join(utils.GetProjectRoot(), "data", fmt.Sprintf("car_%d.txt", id%60))); err != nil {
		panic(err)
	}

	return &Node{
		Id:             id,
		Velocity:       velocity,
		PosX:           posx,
		PosY:           posy,
		Angle:          angle,
		CNN:            make([]*Node, d+1),
		PCH:            make([]*Node, d+1),
		msgChan:        make(chan *messages.Message, 1000), //TODO: change this
		internalChan:   make(chan any, 1000),
		round:          1,
		finishChan:     make(chan struct{}),
		f:              f,
		DHopNeighbors:  make(map[int]*Node),
		subscribers:    make(map[int]struct{}),
		gru:            gru,
		weightMessages: make(map[int]*messages.WeightsMessage),
	}
}

func (n *Node) UpdateNode(posx, posy, velocity, angle float64) {
	n.PosX = posx
	n.PosY = posy
	n.Velocity = velocity
	n.Angle = angle
}

func (n *Node) ResetNode() {
	n.round = 1
	n.PCH = make([]*Node, len(n.PCH))
	n.CNN = make([]*Node, len(n.CNN))
	n.DHopNeighbors = make(map[int]*Node)
	n.subscribers = make(map[int]struct{})
}

func (n *Node) SendWeights() {
	n.sendMsg(messages.NewMessage(n.Id, n.PCH[len(n.PCH)-1].Id, messages.DefaultTTL, &messages.WeightsMessage{SenderId: n.Id, Weights: n.gru.GetWeights()}))
}
func PrintPath(path []*Node) string {
	s := ""
	for _, n := range path {
		s += fmt.Sprintf("%d ", n.Id)
	}
	s += "\n"
	return s
}

func (n *Node) Predict() error {
	n.f.WriteString(fmt.Sprintf("Predicting node %d\n", n.Id))

	predictSize := int(float64(len(n.gru.X)) * TrainSizePercentage)
	n.gru.X = n.gru.X[predictSize:]
	n.gru.Y = n.gru.Y[predictSize:]

	for i := range len(n.gru.X) {
		output, err := n.gru.Predict(n.gru.X[i])
		if err != nil {
			return err
		}
		n.f.WriteString(fmt.Sprintf("[%d]: Predicted: %f, Expected: %f\n", n.Id, n.gru.Sx.InverseTransform([][][]float64{output})[0][0], n.gru.Sx.InverseTransform([][][]float64{n.gru.Y[i]})[0][0]))
	}
	return nil
}
func (n *Node) Train() error {

	trainSize := int(float64(len(n.gru.X)) * TrainSizePercentage)

	n.f.WriteString(fmt.Sprintf("Training node %d\n", n.Id))
	if err := n.gru.Train(n.gru.X[:trainSize], n.gru.Y[:trainSize], Epochs, BatchSize); err != nil {
		return err
	}
	n.f.WriteString(fmt.Sprintf("Finished training node %d\n", n.Id))
	// print errors
	n.f.WriteString(fmt.Sprintf("Errors: %+v\n", n.gru.Errors))
	return nil
}

// TODO: change cluster size and fix function (no if/else) split ?
func (n *Node) HandleWeightsExchange(clusters map[int][]int) {
	myCluster := clusters[n.PCH[len(n.PCH)-1].Id]
	clusterSize := len(myCluster)
	if n.IsCH() {
		n.f.WriteString(fmt.Sprintf("[%d]: Expecting %d weights\n", n.Id, clusterSize-1))
		weights := make([][][][]float64, 1, clusterSize)
		weights[0] = n.gru.GetWeights()
		timer := time.NewTimer(100 * time.Millisecond)
		defer timer.Stop()
	membersLoop:
		for len(weights) < clusterSize {
			select {
			case msg := <-n.internalChan:
				weightMessage, ok := msg.(*messages.WeightsMessage)
				if !ok {
					continue
				}
				weights = append(weights, weightMessage.Weights)
			case <-timer.C:
				n.f.WriteString(fmt.Sprintf("[%d]: Received only %d weights\n", n.Id, len(weights)))
				break membersLoop
			}

		}
		averageWeights := gru.CalculateAverageWeights(weights)

		// send average weights to all cluster heads
		for clusterId := range clusters {
			// don't send to itself
			if clusterId == n.PCH[len(n.PCH)-1].Id {
				continue
			}
			n.sendMsg(messages.NewMessage(n.Id, clusterId, messages.DefaultTTL, &messages.ClusterWeightsMessage{
				SenderId:       n.Id,
				AverageWeights: averageWeights,
			}))
		}

		// receive average weights from other cluster heads
		averageWeightsFromClusters := make([][][][]float64, 1, len(clusters))
		averageWeightsFromClusters[0] = averageWeights

		// since there is no path to all cluster heads, if cluster doesnt receive
		// a message from a cluster head, in 100ms, it stops waiting
		// and calculates the average weights
	clustersLoop:
		for range len(clusters) {
			timer := time.NewTimer(100 * time.Millisecond)
			select {
			case msg := <-n.internalChan:
				weightMessage, ok := msg.(*messages.ClusterWeightsMessage)
				if !ok {
					continue
				}
				averageWeightsFromClusters = append(averageWeightsFromClusters, weightMessage.AverageWeights)
				if !timer.Stop() {
					<-timer.C
				}
			case <-timer.C:
				continue clustersLoop
			}
		}
		totalAverage := gru.CalculateAverageWeights(averageWeightsFromClusters)

		for _, nodeId := range myCluster {
			if nodeId != n.Id {
				n.sendMsg(messages.NewMessage(n.Id, nodeId, messages.DefaultTTL, &messages.WeightsMessage{SenderId: n.Id, Weights: totalAverage}))
			}
		}
		if err := n.gru.SetWeights(totalAverage); err != nil {
			panic(err)
		}
		n.f.WriteString(fmt.Sprintf("[%d]: Finished exchanging weights\n", n.Id))
		return
	}
	// this is executed only by the cluster members
	n.SendWeights()
	timer := time.NewTimer(1 * time.Second)
	defer timer.Stop()
outerLoop:
	for {
		select {
		case msg := <-n.internalChan:
			weightsMessage, ok := msg.(*messages.WeightsMessage)
			if !ok {
				continue
			}
			if err := n.gru.SetWeights(weightsMessage.Weights); err != nil {
				panic(err)
			}
			n.f.WriteString(fmt.Sprintf("CH: %d, Weights: %v\n", n.Id, weightsMessage.Weights))
			break outerLoop
		case <-timer.C:
			fmt.Printf("[%d]: Did not receive weights\n", n.Id)
			n.printPCH(len(n.PCH))
			break outerLoop

		}

	}
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
	fmt.Printf("[%d]: PCH: %d\n", n.Id, len(n.PCH))
	fmt.Printf("[%d]: ", n.Id)
	for i := range d {
		fmt.Printf("%d ", n.PCH[i].Id)
	}
	fmt.Println()
}

func (n *Node) AddNeighbor(neighbor *Node) {
	n.DHopNeighbors[neighbor.Id] = neighbor
	neighbor.DHopNeighbors[n.Id] = n
}

func (n *Node) Degree() int {
	return len(n.DHopNeighbors)
}
func (n *Node) IsCH() bool {
	if n.PCH[len(n.PCH)-1] == nil {
		panic(fmt.Sprintf("[%d]: %+v\n", n.Id, n.PCH))
	}
	return n.PCH[len(n.PCH)-1].Id == n.Id
}

func (n *Node) GetdHopNeighs(d int) map[int]*Node {
	visited := make(map[int]struct{})
	q := make([]*Node, 1)
	q[0] = n
	visited[n.Id] = struct{}{}
	for i := 0; i < d; i++ {
		for _, cn := range q {
			q = q[1:]
			for _, nn := range cn.DHopNeighbors {
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

	for _, cn := range n.DHopNeighbors {
		if m.From == cn.Id {
			continue
		}
		timer := time.NewTimer(500 * time.Millisecond)
		select {
		case cn.msgChan <- &m:
			if !timer.Stop() {
				<-timer.C
			}
		case <-timer.C:
			n.f.WriteString(fmt.Sprintf("Failed to send message (bcast) to: (%d)\n", cn.Id))
		}
	}
}
func (n *Node) sendMsg(msg *messages.Message) {
	timer := time.NewTimer(500 * time.Millisecond)
	defer timer.Stop()
	if c, ok := n.DHopNeighbors[msg.To]; ok {
		select {
		case c.msgChan <- msg:
		case <-timer.C:
			n.f.WriteString(fmt.Sprintf("[%d]: Failed to send message to: (%d)\n", n.Id, msg.To))
		}
		return
	}
	n.bcast(msg)
}

func (n *Node) sendBeacons() {
	for _, ne := range n.DHopNeighbors {
		msg := messages.NewMessage(n.Id, ne.Id, messages.DefaultTTL, &messages.BeaconMessage{
			Velocity: n.Velocity,
			PosX:     n.PosX,
			PosY:     n.PosY,
			Angle:    n.Angle,
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
	if n.PCH[d] == nil {
		return
	}
	for subId := range n.DHopNeighbors {
		msg := messages.NewMessage(n.Id, subId, messages.DefaultTTL, &messages.ClusterMessage{
			Sender:    n.Id,
			ClusterId: n.PCH[d].Id,
			IsCh:      n.IsCH(),
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
				// since we update the nodes, the channel may still have a messages from previous snapshot
				// the message may be from an old neighbor
				// if it is from an old neighbor we should ignore its
				_, ok := n.DHopNeighbors[msg.SenderId]
				if msg.Round != n.round || !ok {
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
			case *messages.WeightsMessage:
			case *messages.ClusterWeightsMessage:
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

func (n *Node) RelativeMax(d int) {
	n.CNN[0] = n
	n.PCH[0] = n

	if len(n.DHopNeighbors) == 0 {
		// node has no neighbors
		for i := 1; i <= d; i++ {
			n.CNN[i] = n
			n.PCH[i] = n
		}
		panic("No neighbors")
	}

	// In the first round, each node finds the CNN based on it's neighborhood.
	n.f.WriteString(fmt.Sprintf("Starting round: %d\n", n.round))
	msgs := make(map[*messages.BeaconMessage]struct{})

	for len(msgs) < len(n.DHopNeighbors) {
		newMsg := <-n.internalChan
		beaconMessage, ok := newMsg.(*messages.BeaconMessage)
		if !ok || beaconMessage.Round != 1 {
			continue
		}
		msgs[beaconMessage] = struct{}{}
	}

	minRelativeMob := math.MaxFloat64
	for msg := range msgs {
		cnn := n.DHopNeighbors[msg.SenderId]
		relativeMobility := n.CNN[0].GetRelativeMobility(msg.Velocity, msg.Angle, msg.PosX, msg.PosY, cnn.Degree(), cnn.PCI())
		n.f.WriteString(fmt.Sprintf("Comparing CNN: %d with %d (%f)\n", n.CNN[0].Id, msg.SenderId, relativeMobility))
		if (relativeMobility < minRelativeMob) || (relativeMobility == minRelativeMob && n.CNN[0].Degree() < cnn.Degree()) {
			n.CNN[1] = cnn
			minRelativeMob = relativeMobility
		}
	}
	if n.CNN[1].Degree() > n.PCH[0].Degree() || (n.CNN[1].Degree() == n.PCH[0].Degree() && n.CNN[1].Velocity < n.PCH[0].Velocity) {
		n.PCH[1] = n.CNN[1]
	} else {
		n.PCH[1] = n.PCH[0]

		for n.round = 2; n.round <= d; n.round++ {
			n.PCH[n.round] = n
			n.CNN[n.round] = n
			n.f.WriteString(fmt.Sprintf("Finished all rounds my CH: %d\n", n.PCH[d].Id))
		}
		return
	}

	n.f.WriteString(fmt.Sprintf("[%d]: Round 1: CNN: %d, PCH: %d\n", n.Id, n.CNN[1].Id, n.PCH[1].Id))

	// Each node has to subscribe to its potential pch to be able to receive CNN messages
	subMsg := messages.NewMessage(n.Id, n.PCH[1].Id, messages.DefaultTTL, &messages.SubscribeMsg{SenderId: n.Id})
	n.sendMsg(subMsg)

	// for the rest of the rounds the CNN is selected based on what the previous CNN has selected
	for n.round = 2; n.round <= d; {
		n.f.WriteString(fmt.Sprintf("Starting round: %d\n", n.round))
		newMsg := <-n.internalChan
		cnnMessage, ok := newMsg.(*messages.CNNMessage)
		if !ok || cnnMessage.Round != n.round || cnnMessage.SenderId != n.PCH[n.round-1].Id {
			continue
		}
		cnn, ok := cnnMessage.CNN.(*Node)
		if !ok {
			continue
		}
		n.CNN[n.round] = cnn
		if cnn.Degree() > n.PCH[n.round-1].Degree() || (cnn.Degree() == n.PCH[n.round-1].Degree() && cnn.Velocity < n.PCH[n.round-1].Velocity) {
			n.PCH[n.round] = cnn
		} else {
			n.PCH[n.round] = n.PCH[n.round-1]
		}
		n.f.WriteString(fmt.Sprintf("[%d]: Round %d: CNN: %d, PCH: %d\n", n.Id, n.round, n.CNN[n.round].Id, n.PCH[n.round].Id))
		n.round++
	}
	n.f.WriteString(fmt.Sprintf("Finished all rounds my CH: %d\n", n.PCH[d].Id))
}
func (n *Node) GetRelativeMobility(vel, angle, x, y float64, degree, pci int) float64 {
	dx := math.Pow(n.PosX-x, 2)
	dy := math.Pow(n.PosY-y, 2)
	dxy := math.Sqrt(dx + dy)

	dvelocityX := math.Abs(math.Cos(angle)*vel - math.Cos(n.Angle)*n.Velocity)

	// Using degree
	return a*dxy + b*dvelocityX + c*(float64(n.Degree())-float64(degree))
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
		for _, n := range cState.node.DHopNeighbors {
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
	pciTable := make([]int, len(n.DHopNeighbors))
	idx := 0
	for _, node := range n.DHopNeighbors {
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
