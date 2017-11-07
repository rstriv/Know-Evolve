#ifndef SPARSE_SURVIVAL_NLL_LAYER_H
#define SPARSE_SURVIVAL_NLL_LAYER_H

#include "i_criterion_layer.h"
#include "linear_param.h"
#include "dense_matrix.h"
#include "recorder.h"
#include "bipartite_graph.h"

template<MatMode mode, typename Dtype>
class SparseSurvivalNllLayer : public ICriterionLayer<mode, Dtype>, public IParametric<mode, Dtype>
{
public:
      SparseSurvivalNllLayer(std::string _name, int _num_entity, int _num_rel, LinearParam<mode, Dtype>* _R, PropErr _properr = PropErr::T)
                : SparseSurvivalNllLayer<mode, Dtype>(_name, _num_entity, _num_rel, _R, 1.0, _properr) {}

      SparseSurvivalNllLayer(std::string _name, int _num_entity, int _num_rel, LinearParam<mode, Dtype>* _R, Dtype _lambda, PropErr _properr = PropErr::T);

            static std::string str_type();

            void SetMutableInfo(BipartiteGraph* _bg, int _subject, int _object, int _rel, Dtype _t_end, Recorder& _cur_time);

            virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override;

            virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override;

            virtual bool HasParam() override { return true; }

            virtual void AccDeriv(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx) override;
protected:
            void AccumulateGrad(int base_idx,
                            DenseMat<mode, Dtype>& coeff,
                            std::map<int, int>& idx_map,
							std::vector<int>& edge_list,
                            std::vector< ILayer<mode, Dtype>* >& operands,
                            bool use_edge_idx);

            void AccumulateGrad(int base_idx,
                            DenseMat<mode, Dtype>& coeff,
                            std::map<int, int>& idx_map,
                            std::vector<Edge>& edge_list,
                            std::vector< ILayer<mode, Dtype>* >& operands,
                            bool use_edge_idx);
			DenseMat<mode, Dtype> A, B, C;
			DenseMat<mode, Dtype> subj_centric, obj_centric, rel_centric;
			DenseMat<mode, Dtype> subj_grad, obj_grad;
			Dtype t_end;
			Dtype temp_dur;
			LinearParam<mode, Dtype> *R;
			Recorder cur_time;
			int current_subject, current_object, current_rel, num_entity, num_rel;
			double total_surv;
			BipartiteGraph* bg;
};
#endif
