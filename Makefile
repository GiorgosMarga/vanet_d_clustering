app/run:
	go run main.go

graph:
	dot -Tpng test.dot -o output.png

clean:
	rm -rf *.info
	rm -rf *.graph_*

.PHONY: graph