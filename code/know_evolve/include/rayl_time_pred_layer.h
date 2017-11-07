#ifndef INCLUDE_RAYL_TIME_PRED_LAYER_H_
#define INCLUDE_RAYL_TIME_PRED_LAYER_H_

#include "i_layer.h"
#include "linear_param.h"
#include "dense_matrix.h"

#include <iostream>

using namespace std;

#define M_PI 3.1415

template<MatMode mode, typename Dtype>
class RaylTimePredLayer : public ILayer<mode, Dtype>
{
public:
			RaylTimePredLayer(std::string _name)
				: ILayer<mode, Dtype>(_name, PropErr::N)
				{
					this->state = new DenseMat<mode, Dtype>();
				}

            static std::string str_type()
            {
            	return "RaylTimePred";
            }

            virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
            {
            	auto& op = operands[0]->state->DenseDerived();

            	auto& cur_out = this->state->DenseDerived();
            	cur_out.CopyFrom(op);
            	cur_out.Exp(op);

            	Dtype temp_val = 1.0 / cur_out.data[0];
            	Dtype pi_by_2 = M_PI / 2.0;

            	cur_out.data[0]  = temp_val * pi_by_2;
            	cur_out.Sqrt();

            }

            virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override {};

protected:
};


#endif
