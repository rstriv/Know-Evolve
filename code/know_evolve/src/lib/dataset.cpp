#include "dataset.h"
#include <vector>
#include "config.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <ctime>
#include <sstream>

void Event::LoadFeat(std::ifstream& ss)
{
	feat.clear();
	int n_feat, feat_id;
	Dtype feat_val;
	ss >> n_feat;
	feat.resize(n_feat);
	for (int i = 0; i < n_feat; ++i)
	{
		ss >> feat_id >> feat_val;
		feat.push_back(std::make_pair(feat_id - 1, feat_val));
	}
}

bool Event::operator<(const Event& other) const
{
	return this->t < other.t;
}

void Dataset::LoadEvents(std::string filename, Phase phase)
{
	Clear();
	for (int i = 0; i < cfg::num_entities; ++i) 
	{
		entity_event_lists.push_back(std::vector<Event*>());
	}
	std::ifstream fs(filename);
	int subject, object, rel;
	Dtype t;
	while (fs >> subject >> rel >> object >> t)
	{
		assert(subject >= 0 && subject < cfg::num_entities); 
		assert(object >= 0 && object < cfg::num_entities); 
		assert(rel >= 0 && rel < cfg::num_rels);
		auto* cur_event = new Event(subject, object, rel, t * cfg::time_scale, phase);
		cur_event->LoadFeat(fs);
		ordered_events.push_back(cur_event);
	}

	if (phase == 0)
		std::cerr << "Train size:" << (int)ordered_events.size() << std::endl;
	else
		std::cerr << "Test size:" << (int)ordered_events.size() << std::endl;

	for (int i = 0; i < (int)ordered_events.size(); ++i)
	{
		auto* cur_event = ordered_events[i];
		cur_event->global_idx = i;
		subject = cur_event->subject;
		object = cur_event->object;
		rel = cur_event->rel;
		cur_event->subj_prev_event = (entity_event_lists[subject].size()) ? entity_event_lists[subject].back() : nullptr;
		cur_event->obj_prev_event = (entity_event_lists[object].size()) ? entity_event_lists[object].back() : nullptr;
		if (cur_event->subj_prev_event)
		{
			cur_event->subj_prev_event->subj_next_event = cur_event; 
		}
		if (cur_event->obj_prev_event)
		{
			cur_event->obj_prev_event->obj_next_event = cur_event;
		}

		entity_event_lists[subject].push_back(cur_event);
		entity_event_lists[object].push_back(cur_event);
	}
}

Dataset::Dataset()
{
	Clear();
}

void Dataset::Clear()
{
	entity_event_lists.clear();

	ordered_events.clear();
}
