#include "bipartite_graph.h"
#include "config.h"
#include <iostream>

Edge::Edge(int _edge_idx, int _dst) : edge_idx(_edge_idx), dst(_dst)
{

}

BipartiteGraph::BipartiteGraph(int _num_entities) : num_entities(_num_entities)
{
	Reset();
}

void BipartiteGraph::Reset()
{
	entity_idx.clear();
	entity_list.clear();
	edge_list.clear();



	if ((int)entity_edge_list.size() != num_entities)
		entity_edge_list.resize(num_entities);
	if ((int)entity_neighbors.size() != num_entities)
		entity_neighbors.resize(num_entities);

	for (int i = 0; i < num_entities; ++i)
	{
		entity_edge_list[i].clear();
		entity_neighbors[i].clear();
	}

}

void BipartiteGraph::AddNodes(int num)
{
	for (int i=0; i < num; ++i)
	{
		entity_idx[i] = i;
		entity_list.push_back(i);
	}
}

void BipartiteGraph::AddEvent(int subject, int object, bool init_ent_list)
{

	if (init_ent_list)
	{
		if (entity_idx.count(subject) == 0)
		{
			int idx = entity_idx.size();
			entity_idx[subject] = idx;
			entity_list.push_back(subject);
		}
		if (entity_idx.count(object) == 0)
		{
			int idx = entity_idx.size();
			entity_idx[object] = idx;
			entity_list.push_back(object);
		}
	}

	int edge_idx = edge_list.size();

	if (entity_neighbors[subject].count(object))
			return;

	entity_edge_list[subject].push_back(Edge(edge_idx, object));
	entity_neighbors[subject].insert(object);
	entity_edge_list[object].push_back(Edge(edge_idx, subject));
	entity_neighbors[object].insert(subject);

	edge_list.push_back(std::make_pair(subject, object));

}

