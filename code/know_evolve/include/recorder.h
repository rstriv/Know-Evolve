#ifndef RECORDER_H
#define RECORDER_H

#include <map>
#include "config.h"

class Recorder 
{
public:

		Recorder();

		void UpdateEvent(int subject, int object, Dtype t);

		Dtype GetCurTime(int subject, int object);

		Dtype GetLastInteractTime(int subject, int object);

		void Init(int _n_entity, Dtype _t_begin);

private:
		std::map< int, Dtype > cur_ent_time;
		std::vector< std::map<int, Dtype> > cur_edge_time;
		int n_entity;
		Dtype t_begin;
};

#endif
