#ifndef BIPARTITE_GRAPH_H
#define BIPARTITE_GRAPH_H

#include <vector>
#include <set>
#include <map>

class Edge
{
public:

	int edge_idx;
	int dst;

	Edge(int _edge_idx, int _dst);
};

class BipartiteGraph
{
public:

	BipartiteGraph(int _num_entities);

	void Reset();
	void AddNodes(int num);
	void AddEvent(int subject, int object, bool init_ent_list);


	int num_entities;


	std::vector<int> entity_list;
	std::map<int, int> entity_idx;
	std::vector< std::set<int> > entity_neighbors;
	std::vector< std::vector<Edge> > entity_edge_list;

	std::vector< std::vector<Edge> > entity_neg_edge_list; 

	std::vector< std::pair<int, int> > edge_list;

};

#endif

