#include "embedding_holder.h"
#include "layer_holder.h"

EmbeddingHolder::EmbeddingHolder(int num)
{
	embedding_list.clear();
	embedding_rel_list.clear();

	for (int i = 0; i < num; ++i)
	{
		embedding_list.push_back(std::deque< std::pair<Event*, ILayer<mode, Dtype>*> >());
		embedding_rel_list.push_back(std::deque< std::pair<Event*, ILayer<mode, Dtype>*> >());
	}
}

void EmbeddingHolder::KeepLatestOnly(int id)
{
	while (embedding_list[id].size() > 1)
	{
		LayerHolder::RecycleLayer(embedding_list[id].front().second);
		embedding_list[id].pop_front();
	}
}

void EmbeddingHolder::KeepLatestRelOnly(int id)
{
	while (embedding_rel_list[id].size() > 1)
	{
		LayerHolder::RecycleLayer(embedding_rel_list[id].front().second);
		embedding_rel_list[id].pop_front();
	}
}


void EmbeddingHolder::ClearLatest()
{
	for (size_t i = 0; i < embedding_list.size(); ++i)
	{
		assert(embedding_list[i].size() <= 1);
		if (embedding_list[i].size() == 0)
			continue;
		LayerHolder::RecycleLayer(embedding_list[i].front().second);
		embedding_list[i].clear();
	}
}

void EmbeddingHolder::ClearAll()
{
	for (size_t i = 0; i < embedding_list.size(); ++i)
		embedding_list[i].clear();
	for (size_t i = 0; i < embedding_rel_list.size(); ++i)
		embedding_rel_list[i].clear();
}

void EmbeddingHolder::ClearFull()
{

	for (size_t i = 0; i < embedding_list.size(); ++i){

		if (embedding_list[i].size() == 0)
			continue;
		LayerHolder::RecycleLayer(embedding_list[i].front().second);
		embedding_list[i].clear();
	}

	for (size_t i = 0; i < embedding_rel_list.size(); ++i){
		embedding_rel_list[i].clear();
	}

}



void EmbeddingHolder::InsertEmbedding(int id, Event* event, ILayer<mode, Dtype>* embed)
{
	if (embedding_list[id].size() > 1)
	{
		assert(event->global_idx >= embedding_list[id].back().first->global_idx);
	}
	embedding_list[id].push_back(std::make_pair(event, embed));
}

void EmbeddingHolder::InsertRelEmbedding(int id, Event* event, ILayer<mode, Dtype>* rel_embed)
{
	if (embedding_rel_list[id].size())
	{
		assert(event->global_idx >= embedding_rel_list[id].back().first->global_idx);
	}
	embedding_rel_list[id].push_back(std::make_pair(event, rel_embed));
}


ILayer<mode, Dtype>* EmbeddingHolder::GetLatestEmbedding(int id)
{
	if (embedding_list[id].size())
		return embedding_list[id].back().second;
	else
		return nullptr;
}

ILayer<mode, Dtype>* EmbeddingHolder::GetLatestRelEmbedding(int id)
{
	if (embedding_rel_list[id].size())
		return embedding_rel_list[id].back().second;
	else
		return nullptr;
}



ILayer<mode, Dtype>* EmbeddingHolder::GetLatestEmbeddingTill(int id, Event* event, bool inclusive)
{
	ILayer<mode, Dtype>* result = nullptr;
	if (embedding_list[id].size())
	{
		auto& list = embedding_list[id];
		for (size_t i = 0; i < list.size(); ++i)
		{
			if (list[i].first->global_idx > event->global_idx)
				break;
			if (inclusive && list[i].first->global_idx <= event->global_idx)
				result = list[i].second;
			if (!inclusive && list[i].first->global_idx < event->global_idx)
				result = list[i].second;
		}
	}
	return result;
}
