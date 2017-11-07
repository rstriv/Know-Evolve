#ifndef LAYER_HOLDER_H
#define LAYER_HOLDER_H

#include "config.h"
#include <deque>
#include <vector>
#include "i_layer.h"
#include "nngraph.h"

class LayerHolder
{
public:

	static void Init();

	static void RecycleLayer(ILayer<mode, Dtype>* layer);

	static ILayer<mode, Dtype>* GetLayer(std::string name_prefix);

	static void UseNewLayer(ILayer<mode, Dtype>* layer); 

	static int layer_created;
	static std::map<std::string, std::vector<ILayer<mode, Dtype>*> > layer_store;
};

template<template <MatMode, typename> class LayerType, MatMode mode, typename Dtype, typename... Args>
inline ILayer<mode, Dtype>* gl(const std::string name_prefix, 
                               NNGraph<mode, Dtype>& gnn,                             
                               std::vector< ILayer<mode, Dtype>* > operands, 
                               Args&&... args)
{        
		ILayer<mode, Dtype>* layer = LayerHolder::GetLayer(name_prefix);
		if (layer == nullptr)
		{
			layer = new LayerType<mode, Dtype>(name_prefix, std::forward<Args>(args)...);
			LayerHolder::UseNewLayer(layer);
		}
        gnn.InsertLayer(layer, operands);
        return layer;		
}

// workaround for deducting list
template<template <MatMode, typename> class LayerType, MatMode mode, typename Dtype, typename... Args>
inline ILayer<mode, Dtype>* gl(const std::string name_prefix,
                               NNGraph<mode, Dtype>& gnn,
                               std::vector< ILayer<mode, Dtype>* > operands,
                               std::vector< IParam<mode, Dtype>* > params, 
                               Args&&... args)
{
		ILayer<mode, Dtype>* layer = LayerHolder::GetLayer(name_prefix);
		if (layer == nullptr)
		{
			layer = new LayerType<mode, Dtype>(name_prefix, params, std::forward<Args>(args)...);
			LayerHolder::UseNewLayer(layer);
		}
        gnn.InsertLayer(layer, operands);
        return layer;				
}

#endif
