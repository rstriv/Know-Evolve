#include "layer_holder.h"
#include "i_layer.h"
#include <vector>
#include <map>
#include "cppformat/format.h"

void LayerHolder::Init()
{
	layer_store.clear();
	layer_created = 0;
}

std::string GetPrefix(std::string name)
{
	auto pos = name.find('-');
	assert(pos != std::string::npos);
	return name.substr(0, pos);
}

void LayerHolder::RecycleLayer(ILayer<mode, Dtype>* layer)
{
	auto name_prefix = GetPrefix(layer->name);
	if (!layer_store.count(name_prefix))
		layer_store[name_prefix] = std::vector<ILayer<mode, Dtype>*>();
	layer_store[name_prefix].push_back(layer);
}

ILayer<mode, Dtype>* LayerHolder::GetLayer(std::string name_prefix)
{
	if (!layer_store.count(name_prefix) || layer_store[name_prefix].size() == 0)
		return nullptr;

	auto* result = layer_store[name_prefix].back();
	layer_store[name_prefix].pop_back();
	return result;
}

void LayerHolder::UseNewLayer(ILayer<mode, Dtype>* layer)
{
	layer->name = fmt::sprintf("%s-%d", layer->name.c_str(), layer_created);
	layer_created++;
}

int LayerHolder::layer_created;
std::map<std::string, std::vector<ILayer<mode, Dtype>*> > LayerHolder::layer_store;
