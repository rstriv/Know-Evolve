#include "sparse_survival_nll_layer.h"
#include <random>
#include "mkl_helper.h"
#define max(x, y) (x > y ? x : y)
#define sqr(x) ((x) * (x))

using std::default_random_engine;
int seed = time(0);
default_random_engine RNG(seed);

template<MatMode mode, typename Dtype>
SparseSurvivalNllLayer<mode, Dtype>::SparseSurvivalNllLayer(std::string _name, int _num_entity, int _num_rel, LinearParam<mode, Dtype>* _R, Dtype _lambda, PropErr _properr)
                                    : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr), num_entity(_num_entity), num_rel(_num_rel)
{
		this->R = _R;
		this->grad = new DenseMat<mode, Dtype>();
        this->state = new DenseMat<mode, Dtype>();

}

template<MatMode mode, typename Dtype>
std::string SparseSurvivalNllLayer<mode, Dtype>::str_type()
{
        return "SparseSurvivalNll";
}

template<MatMode mode, typename Dtype>
void SparseSurvivalNllLayer<mode, Dtype>::SetMutableInfo(BipartiteGraph* _bg, int _subject, int _object, int _rel, Dtype _t_end, Recorder& _cur_time)
{
		this->bg = _bg;
        this->current_subject = _subject;
        this->current_object = _object;
        this->num_rel = _rel;
        this->t_end = _t_end;
        this->cur_time = _cur_time;
}

//Survival Loss computation module
template<MatMode mode, typename Dtype>
void SparseSurvivalNllLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
	assert(operands.size() == bg->entity_list.size());
    // survival of all
    if (current_subject < 0 || current_object < 0)
    {

    	auto& cur_state = this->state->DenseDerived();
    	cur_state.Resize(1, bg->edge_list.size());
    	auto& cur_rel_weight = R->p["weight"]->value;

    	#pragma omp parallel for
    	for (size_t i = 0; i < bg->edge_list.size(); ++i)
    	{
    		int subject = bg->edge_list[i].first;
    		int object = bg->edge_list[i].second;
    		auto& subject_feat = operands[bg->entity_idx[subject]]->state->DenseDerived();
    		auto& object_feat = operands[bg->entity_idx[object]]->state->DenseDerived();

    		A.CopyFrom(subject_feat);
    		B.GeMM(A,cur_rel_weight, Trans::N, Trans::N, 1.0, 0.0);
    		C.GeMM(B,object_feat,Trans::N, Trans::T, 1.0, 0.0);
    		temp_dur = sqr(t_end) - sqr(cur_time.GetCurTime(subject, object));
    		cur_state.data[i] = exp(C.data[0]) * (temp_dur / 2);
    	}
		this->loss = cur_state.Sum();

    } else
    {
    	auto& cur_subject_feat = operands[bg->entity_idx[current_subject]]->state->DenseDerived();
    	auto& cur_object_feat = operands[bg->entity_idx[current_object]]->state->DenseDerived();
    	auto& cur_rel_weight = R->p["weight"]->value;

    	std::uniform_int_distribution<int> s_dist(0,num_entity-1);

    	auto& batch_object_list = bg->entity_list;
    	subj_centric.Resize(1, batch_object_list.size());
	
	#pragma omp parallel for
    	for (size_t i=0; i < batch_object_list.size(); ++i)
    	{
    		int object = batch_object_list[i];
    		if (object == current_subject){
    			subj_centric.data[i]=0.0;
    			continue;
    		}

    		auto& object_feat = operands[bg->entity_idx[object]]->state->DenseDerived();
			B.GeMM(cur_subject_feat,cur_rel_weight, Trans::N, Trans::N, 1.0, 0.0);
			C.GeMM(B,object_feat,Trans::N, Trans::T, 1.0, 0.0);
			temp_dur = sqr(t_end) - sqr(cur_time.GetCurTime(current_subject, object));
			subj_centric.data[i] = exp(C.data[0]) * (temp_dur/ 2);
    	}

    	/* All about current object*/

    	Dtype dup = 0; //discard duplicate calculation for subject-object edge
    	auto& batch_subject_list = bg->entity_list;
    	obj_centric.Resize(1, batch_subject_list.size());
	#pragma omp parallel for
    	for (size_t i=0; i < batch_subject_list.size(); ++i)
    	{
    		int subject = batch_subject_list[i];
    		if (subject == current_object){
    			obj_centric.data[i]=0.0;
    			continue;
    		}
    		auto& subject_feat = operands[bg->entity_idx[subject]]->state->DenseDerived();
			B.GeMM(subject_feat,cur_rel_weight, Trans::N, Trans::N, 1.0, 0.0);
			C.GeMM(B,cur_object_feat,Trans::N, Trans::T, 1.0, 0.0);
			temp_dur = sqr(t_end) - sqr(cur_time.GetCurTime(subject, current_object));
			obj_centric.data[i] = exp(C.data[0]) * (temp_dur/ 2);
			
			if (subject == current_subject)
				dup = obj_centric.data[i];
    	}


    	/* All about current Relation*/

    	//We only need update relation gradient with sum of subj and obj centric.

    	total_surv = subj_centric.Sum() + obj_centric.Sum() -eps;
		this->loss = total_surv;
    }

    this->loss *= this->lambda;

}

//function overloading for accumulategrad as one uses edge list  for all survival while other uses
// just subject or object list

template<MatMode mode, typename Dtype>
void SparseSurvivalNllLayer<mode, Dtype>::AccumulateGrad(int base_idx,
                                                            DenseMat<mode, Dtype>& coeff,
                                                            std::map<int, int>& idx_map,
							    std::vector<int>& edge_list,
                                                            std::vector< ILayer<mode, Dtype>* >& operands,
                                                            bool use_edge_idx)
{
        auto& grad = this->grad->DenseDerived();
        auto& cur_rel_weight = R->p["weight"]->value;
        for (size_t i = 0; i < edge_list.size(); ++i)
        {
            int other_name = edge_list[i];
            auto& other_feat = operands[base_idx + idx_map[other_name]]->state->DenseDerived();

            A.GeMM(other_feat,cur_rel_weight,Trans::N, Trans::N, 1.0, 0.0); 
            Dtype factor = coeff.data[i];
            grad.Axpby(factor, A, 1.0);
        }
}


template<MatMode mode, typename Dtype>
void SparseSurvivalNllLayer<mode, Dtype>::AccumulateGrad(int base_idx,
                                                            DenseMat<mode, Dtype>& coeff,
                                                            std::map<int, int>& idx_map,
                                                            std::vector<Edge>& edge_list,
                                                            std::vector< ILayer<mode, Dtype>* >& operands,
                                                            bool use_edge_idx)
{
        auto& grad = this->grad->DenseDerived();
        auto& cur_rel_weight = R->p["weight"]->value;
        for (size_t i = 0; i < edge_list.size(); ++i)
        {
            int other_name = edge_list[i].dst;
            auto& other_feat = operands[idx_map[other_name]]->state->DenseDerived();
            A.GeMM(other_feat,cur_rel_weight,Trans::N, Trans::N, 1.0, 0.0); 
            Dtype factor = use_edge_idx ? coeff.data[edge_list[i].edge_idx] : coeff.data[i];
            grad.Axpby(factor, A, 1.0);
        }
}




//gradients for the survival term
template<MatMode mode, typename Dtype>
void SparseSurvivalNllLayer<mode, Dtype>::BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta)
{
        auto& prev_grad = operands[cur_idx]->grad->DenseDerived();
        auto& grad = this->grad->DenseDerived();
        grad.Zeros(1, operands[cur_idx]->state->cols);

        if (current_subject < 0 || current_object < 0)
        {

            auto& cur_state = this->state->DenseDerived();
            AccumulateGrad(0, cur_state, bg->entity_idx,
                            bg->entity_edge_list[bg->entity_list[cur_idx]],
                            operands, true);

        } else {

        	if (cur_idx == bg->entity_idx[current_subject])
        		AccumulateGrad(0, subj_centric, bg->entity_idx,
        				bg->entity_list,operands, false);
        	else if (cur_idx == bg->entity_idx[current_object])
                AccumulateGrad(0, obj_centric, bg->entity_idx,
                            bg->entity_list,operands, false);
        	else {

				auto& object_edge_list = bg->entity_edge_list[current_object];
				for (size_t i = 0; i < object_edge_list.size(); ++i)
					if (object_edge_list[i].dst == bg->entity_list[cur_idx])
					{
						grad.CopyFrom(operands[bg->entity_idx[current_object]]->state->DenseDerived());
						grad.Scale(obj_centric.data[i]);
						break;
					}

				auto& subject_edge_list = bg->entity_edge_list[current_subject];
				for (size_t i = 0; i < subject_edge_list.size(); ++i)
					if (subject_edge_list[i].dst == bg->entity_list[cur_idx])
					{
						grad.CopyFrom(operands[bg->entity_idx[current_subject]]->state->DenseDerived());
						grad.Scale(subj_centric.data[i]);
						break;
					}

        	}

        }

        if (beta == 0)
            prev_grad.CopyFrom(grad);
        else
            prev_grad.Axpby(1.0, grad, beta);
}

template<MatMode mode, typename Dtype>
void SparseSurvivalNllLayer<mode, Dtype>::AccDeriv(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx)
{
		R->p["weight"]->grad.Scale(total_surv);
}


template class SparseSurvivalNllLayer<CPU, float>;
template class SparseSurvivalNllLayer<CPU, double>;

