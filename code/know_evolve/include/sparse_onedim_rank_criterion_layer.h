#ifndef SPARSE_ONEDIM_RANK_CRITERION_LAYER_H
#define SPARSE_ONEDIM_RANK_CRITERION_LAYER_H

#include "i_criterion_layer.h"
#include "loss_func.h"
#include "linear_param.h"
#include "dense_matrix.h"
#include "bipartite_graph.h"
#include "recorder.h"

template<MatMode mode, typename Dtype>
class SparseOnedimRankCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			SparseOnedimRankCriterionLayer(std::string _name, int _num_entity, LinearParam<mode, Dtype>* _R, RankOrder _order, std::map<std::string, int> _entity_dict, std::map<std::string, int> _event_dict);
             				 
            static std::string str_type();

            void SetMutableInfo(BipartiteGraph* _bg, int _subject, int _object, int _rel, Dtype _t, Recorder& _cur_time, bool _rank_object);
            
            virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override; 
            
            virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override {}
                         
            virtual bool HasParam() override { return false; } 
    
protected:
            BipartiteGraph* bg;
			RankOrder order; 
			int subject, object, rel, num_entity;
			bool rank_object;
			LinearParam<mode, Dtype> *R;
			DenseMat<mode, Dtype> A, B, C, D;
			Dtype event_t;
			Recorder cur_time;
			Dtype LogLL(Dtype sim, Dtype dur);
			std::map<std::string, int> entity_dict;
			std::map<std::string, int> event_dict;
};

#endif
