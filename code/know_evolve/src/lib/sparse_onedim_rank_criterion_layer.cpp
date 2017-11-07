#include "sparse_onedim_rank_criterion_layer.h"
#include "dense_matrix.h"
#include "mkl_helper.h"

#define max(x, y) (x > y ? x : y)
#define sqr(x) ((x) * (x))

template<MatMode mode, typename Dtype>
SparseOnedimRankCriterionLayer<mode, Dtype>::SparseOnedimRankCriterionLayer(std::string _name, int _num_entity, LinearParam<mode, Dtype>* _R, RankOrder _order, std::map<std::string, int> _entity_dict, std::map<std::string, int> _event_dict)
                                    : ICriterionLayer<mode, Dtype>(_name, 1.0, PropErr::N), order(_order), num_entity(_num_entity), entity_dict(_entity_dict), event_dict(_event_dict)
{
		this->R = _R;
		this->grad = new DenseMat<mode, Dtype>();
        this->state = new DenseMat<mode, Dtype>();
}

template<MatMode mode, typename Dtype>
void SparseOnedimRankCriterionLayer<mode, Dtype>::SetMutableInfo(BipartiteGraph* _bg, int _subject, int _object, int _rel, Dtype _t, Recorder& _cur_time, bool _rank_object)
{
		this->bg = _bg;
		this->subject = _subject;
		this->object = _object;
		this->rel = _rel;
		this->rank_object = _rank_object;
		this->event_t = _t;
		this->cur_time = _cur_time;
}

template<MatMode mode, typename Dtype>
std::string SparseOnedimRankCriterionLayer<mode, Dtype>::str_type()
{
        return "SparseOnedimRank";
}

template<MatMode mode, typename Dtype>
Dtype SparseOnedimRankCriterionLayer<mode, Dtype>::LogLL(Dtype sim, Dtype dur)
{
        Dtype intensity = exp(sim);
        return log(dur) + sim - 0.5 * intensity * sqr(dur);
}

//Compute rank of entity for evaluation
template<MatMode mode, typename Dtype>
void SparseOnedimRankCriterionLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
		assert(rank_object); 
		auto& cur_rel_weight = R->p["weight"]->value;
		auto& cur_feat = operands[bg->entity_idx[subject]]->state->DenseDerived();
		this->loss = 1.0;


		B.GeMM(cur_feat,cur_rel_weight, Trans::N, Trans::N, 1.0, 0.0);
		C.GeMM(B,operands[bg->entity_idx[object]]->state->DenseDerived(),Trans::N, Trans::T, 1.0, 0.0);
		Dtype sim = C.data[0]; 
		sim = LogLL(sim, this->event_t - this->cur_time.GetCurTime(subject, object));
		#pragma omp parallel for
		for (size_t i = 0; i < bg->entity_list.size(); ++i)
		{
			int pred_object = bg->entity_list[i];
			if (pred_object == this->object || pred_object == this->subject)
				continue;
			if (entity_dict.count(std::to_string(pred_object))==0)
				continue;

			/*Needed for filtering rank report
			std::string e1 =  std::to_string(this->subject);
			std::string e2 =  std::to_string(this->object);
			std::string r = std::to_string(this->rel);
			std::string key = e1 + "_" + r + "_" + e2;

			if (event_dict.count(key) > 0)
			{
				continue;
			}
			*/

			auto& other_feat = operands[bg->entity_idx[pred_object]]->state->DenseDerived();
			D.GeMM(B,other_feat,Trans::N, Trans::T, 1.0, 0.0);
			Dtype cur_sim = D.data[0]; 
			cur_sim = LogLL(cur_sim, this->event_t - this->cur_time.GetCurTime(subject, object));

			if (cur_sim > sim && order == RankOrder::DESC)
				this->loss++;
			if (cur_sim < sim && order == RankOrder::ASCE)
				this->loss++;

		}
}


template class SparseOnedimRankCriterionLayer<CPU, float>;
template class SparseOnedimRankCriterionLayer<CPU, double>;
