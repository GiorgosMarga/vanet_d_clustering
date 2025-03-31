app/run:
	go run main.go

graph:
	dot -Tpng test.dot -o output.png

clean:
	rm -rf *.png
	rm -rf ./sumo/*
	rm -rf ./graphviz/*
	rm -rf ./graph_info/*
	rm -rf ./cars_info/*
	rm -rf ./graph.info
clean/snapshots:
	rm -rf ./snapshots/*
clean/data:
	rm -rf ./data/*
.PHONY: graph