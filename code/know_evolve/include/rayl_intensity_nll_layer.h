#ifndef RAYL_INTENSITY_NLL_LAYER_H
#define RAYL_INTENSITY_NLL_LAYER_H

#include "i_criterion_layer.h"

template<MatMode mode, typename Dtype>
class RaylIntensityNllLayer : public ICriterionLayer<mode, Dtype>
{
public:
			RaylIntensityNllLayer(std::string _name, PropErr _properr = PropErr::T)
                : RaylIntensityNllLayer<mode, Dtype>(_name, 1.0, _properr) {}
                
			RaylIntensityNllLayer(std::string _name, Dtype _lambda, PropErr _properr = PropErr::T);
             				 
            static std::string str_type();
            
            virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override; 
            
            virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override;    
                         
            virtual bool HasParam() override { return false; } 
    
protected:
};

#endif
