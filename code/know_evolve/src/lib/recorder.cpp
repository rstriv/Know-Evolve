#include "recorder.h"
#define max(x, y) (x > y ? x : y)

Recorder::Recorder()
{
	cur_ent_time.clear();
	cur_edge_time.clear();
}

void Recorder::Init(int _n_entity, Dtype _t_begin)
{
	this->n_entity = _n_entity;
	this->t_begin = _t_begin;

	cur_ent_time.clear();
	cur_edge_time.resize(this->n_entity);

	for (int i = 0; i < this->n_entity; ++i)
			cur_edge_time[i].clear();

}

void Recorder::UpdateEvent(int subject, int object, Dtype t)
{
	if (cur_ent_time.count(subject))
		assert(t >= cur_ent_time[subject]);
	if (cur_ent_time.count(object))
		assert(t >= cur_ent_time[object]);

	cur_ent_time[subject] = t;
	cur_ent_time[object] = t;
	cur_edge_time[subject][object] = t;
}

Dtype Recorder::GetCurTime(int subject, int object)
{
	Dtype t = t_begin;
	if (cur_ent_time.count(subject))
		t = max(t, cur_ent_time[subject]);
	if (cur_ent_time.count(object))
		t = max(t, cur_ent_time[object]);

	return t;
}

Dtype Recorder::GetLastInteractTime(int subject, int object)
{
	if (cur_edge_time[subject].count(object))
		return cur_edge_time[subject][object];
	return 0;
}
