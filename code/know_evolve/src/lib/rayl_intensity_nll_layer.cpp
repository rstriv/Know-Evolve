#include "rayl_intensity_nll_layer.h"
#include "dense_matrix.h"

template<MatMode mode, typename Dtype>
RaylIntensityNllLayer<mode, Dtype>::RaylIntensityNllLayer(std::string _name, Dtype _lambda, PropErr _properr)
                                    : ICriterionLayer<mode, Dtype>(_name, _lambda, _properr)
{
        this->grad = new DenseMat<mode, Dtype>();    
        this->state = new DenseMat<mode, Dtype>();    
}

template<MatMode mode, typename Dtype>
std::string RaylIntensityNllLayer<mode, Dtype>::str_type()
{
        return "RaylIntensityNll";
}

template<MatMode mode, typename Dtype>
void RaylIntensityNllLayer<mode, Dtype>::UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase)
{
        assert(operands.size() == 2);                

        auto& state = this->state->DenseDerived();
        auto& base_intensity = operands[0]->state->DenseDerived();
        Dtype dur = operands[1]->state->DenseDerived().AsScalar();

        state.CopyFrom(base_intensity);
        if (!(dur==0))
        	state.Add(log(dur));

        this->loss = -state.Sum(); 
}

template<MatMode mode, typename Dtype>
void RaylIntensityNllLayer<mode, Dtype>::BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta)
{
        assert(operands.size() == 2 && cur_idx == 0);
        
        auto& prev_grad = operands[0]->grad->DenseDerived();

        prev_grad.Scale(beta);

        prev_grad.Add(-this->lambda);
}

template class RaylIntensityNllLayer<CPU, float>;
template class RaylIntensityNllLayer<CPU, double>;
