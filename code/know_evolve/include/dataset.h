#ifndef DATASET_H
#define DATASET_H
#include <vector>
#include "config.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <ctime>
#include <sstream>

struct Event
{
	std::vector< std::pair<int, Dtype> > feat;
	int subject, object, rel;
	Dtype t;
	Event* subj_prev_event;
	Event* obj_prev_event;
	Event* subj_next_event;
	Event* obj_next_event;
	int global_idx;
	Phase phase;
	Event(Phase _phase) 
	{
		phase = _phase;
		subj_next_event = subj_prev_event = obj_next_event = obj_prev_event;
	}

	Event(int _subject, int _object, int _rel, Dtype _t, Phase _phase)
		: subject(_subject), object(_object), t(_t), rel(_rel), phase(_phase)
		{ 
			feat.clear(); 
			subj_next_event = subj_prev_event = obj_next_event = obj_prev_event = nullptr;
		}

	bool operator<(const Event& other) const;

	void LoadFeat(std::ifstream& ss);
};

struct Dataset
{
	Dataset();
	
	void Clear();

	void LoadEvents(std::string filename, Phase phase); 

	std::vector< std::vector< Event* > > entity_event_lists;
	std::vector< Event* > ordered_events;
};

#endif
