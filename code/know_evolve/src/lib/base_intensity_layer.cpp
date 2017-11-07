#include "base_intensity_layer.h"

template<MatMode mode, typename Dtype>
BaseIntensityLayer<mode, Dtype>::BaseIntensityLayer(std::string _name, LinearParam<mode, Dtype>* _R, Dtype _lambda, PropErr _properr)
        : ICriterionLayer<mode, Dtype>(_name,_lambda, _properr)
{
        this->R = _R;
		this->state = new DenseMat<mode, Dtype>();
		this->grad = new DenseMat<mode, Dtype>();
}

template<MatMode mode, typename Dtype>
std::string BaseIntensityLayer<mode, Dtype>::str_type()
{
        return "BaseIntensity";
}


template<MatMode mode, typename Dtype>
void BaseIntensityLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
        assert(operands.size() == 2);

        auto& cur_output = this->state->DenseDerived();
        auto& cur_subject = operands[0]->state->DenseDerived();
        auto& cur_object = operands[1]->state->DenseDerived();
        auto& cur_relation = R->p["weight"]->value;
        buf.GeMM(cur_subject,cur_relation,Trans::N, Trans::N, 1.0, 0.0);
        cur_output.GeMM(buf, cur_object, Trans::N, Trans::T, 1.0, 0.0);

}

template<MatMode mode, typename Dtype>
void BaseIntensityLayer<mode, Dtype>::BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta)
{
        assert(operands.size() == 2);

        auto& cur_grad = this->grad->DenseDerived();
        auto& prev_grad = operands[cur_idx]->grad->DenseDerived();
        auto& another_operand = operands[1 - cur_idx]->state->DenseDerived();
        auto& rel_weight = R->p["weight"]->value;

        buf2.GeMM(another_operand,rel_weight,Trans::N, Trans::N, 1.0, 0.0);
        buf.MulColVec(buf2, cur_grad);
        if (beta == 0)
        	prev_grad.CopyFrom(buf);
        else
        	prev_grad.Axpby(1.0, buf, beta);
}


template<MatMode mode, typename Dtype>
void BaseIntensityLayer<mode, Dtype>::AccDeriv(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx)
{
        assert(operands.size() == 2);
        if (cur_idx)
            return;


        auto& subj = operands[0]->state->DenseDerived();
        auto& obj = operands[1]->state->DenseDerived();

        bufR.GeMM(obj,subj,Trans::T, Trans::N, 1.0, 0.0);
        R->p["weight"]->grad.Scale(bufR.data[0]);
}

template class BaseIntensityLayer<CPU, float>;
template class BaseIntensityLayer<CPU, double>;

