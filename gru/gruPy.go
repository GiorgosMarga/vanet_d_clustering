package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"log"
	"net"
)

func main() {
	randomWeights := [][][]float64{
		{
			{1, 1, 1, 1, 1},
		},
		{
			{2, 2, 2, 2, 2},
		},
		{
			{3, 3, 3, 3, 3},
		},
		{
			{4, 4, 4, 4, 4},
		},
		{
			{5, 5, 5, 5, 5},
		},
	}
	Train(randomWeights, 1)
}

func Train(targets [][][]float64, epochs int) error {
	// Serialize the data using gob
	var buffer bytes.Buffer
	encoder := gob.NewEncoder(&buffer)
	err := encoder.Encode(targets)
	if err != nil {
		log.Fatal("Error encoding data:", err)
	}

	// Send the serialized data over UDP
	serverAddr := "localhost:12345"
	conn, err := net.Dial("udp", serverAddr)
	if err != nil {
		log.Fatal("Error connecting to server:", err)
	}
	defer conn.Close()

	_, err = conn.Write(buffer.Bytes()) // Send the byte slice to the server
	if err != nil {
		log.Fatal("Error sending data:", err)
	}

	fmt.Println("Data sent to server.")

}
