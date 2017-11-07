#ifndef EMBEDDING_HOLDER_H
#define EMBEDDING_HOLDER_H

#include <deque>
#include <vector>
#include "config.h"
#include "i_layer.h"
#include "dataset.h"

class EmbeddingHolder
{
public:
	EmbeddingHolder(int num);

	void ClearAll();
	void ClearFull();
	void ClearLatest();
	void KeepLatestOnly(int id);
	void KeepLatestRelOnly(int id);

	ILayer<mode, Dtype>* GetLatestEmbedding(int id);
	ILayer<mode, Dtype>* GetLatestRelEmbedding(int id);
	ILayer<mode, Dtype>* GetLatestEmbeddingTill(int id, Event* event, bool inclusive);


	void InsertEmbedding(int id, Event* event, ILayer<mode, Dtype>* embed);
	void InsertRelEmbedding(int id, Event* event, ILayer<mode, Dtype>* rel_embed);

	std::vector< std::deque< std::pair<Event*, ILayer<mode, Dtype>*> > > embedding_list;
	std::vector< std::deque< std::pair<Event*, ILayer<mode, Dtype>*> > > embedding_rel_list;
};

#endif
