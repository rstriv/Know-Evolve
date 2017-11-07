#ifndef BASE_INTENSITY_LAYER_H_
#define BASE_INTENSITY_LAYER_H_

#include "i_layer.h"
#include "linear_param.h"
#include "i_criterion_layer.h"


template<MatMode mode, typename Dtype>
class BaseIntensityLayer : public ICriterionLayer<mode, Dtype>, public IParametric<mode, Dtype>
{
public:
			BaseIntensityLayer(std::string _name,  LinearParam<mode, Dtype>* _R, PropErr _properr = PropErr::T)
                : BaseIntensityLayer<mode, Dtype>(_name, _R, 1.0, _properr) {}

			BaseIntensityLayer(std::string _name, LinearParam<mode, Dtype>* _R, Dtype _lambda, PropErr _properr = PropErr::T);

            static std::string str_type();

            virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override;

            virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override;

            virtual bool HasParam() override { return true; }

            virtual void AccDeriv(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx) override;

protected:
            DenseMat<mode, Dtype> buf, buf2,bufR, ones;
            LinearParam<mode, Dtype> *R;
};

#endif
